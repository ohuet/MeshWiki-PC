"""Index Wikipedia ZIM files into ChromaDB for semantic search."""

import json
import logging
import queue
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import chromadb
import yaml
from lxml import html as lxml_html

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("data/indexing_checkpoint.json")

_indexing_eta: datetime | None = None
_indexing_lock = threading.Lock()


def get_indexing_eta() -> datetime | None:
    """Return the estimated completion time of an ongoing indexation, or None."""
    with _indexing_lock:
        return _indexing_eta


def _set_indexing_eta(eta: datetime | None) -> None:
    """Update the estimated completion time of the current indexation."""
    global _indexing_eta
    with _indexing_lock:
        _indexing_eta = eta


def _format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 0:
        return "?"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}min"
    if minutes > 0:
        return f"{minutes}min{secs:02d}s"
    return f"{secs}s"


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _clean_html(html: str) -> str:
    """Extract plain text from HTML, removing tags and extra whitespace."""
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    for el in doc.iter("script", "style"):
        el.drop_tree()
    text = doc.text_content()
    return re.sub(r"\s+", " ", text).strip()


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens with overlap.

    Uses whitespace-based token approximation.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start >= len(words):
            break

    return chunks


def _is_content_article(entry) -> bool:
    """Check if a ZIM entry is a real content article (not redirect/meta)."""
    try:
        path = entry.path
        # Skip non-article entries
        if not path.startswith("A/") and not path.startswith("C/"):
            # In newer ZIM format, articles may not have A/ prefix
            pass

        # Skip if it's a redirect
        if entry.is_redirect:
            return False

        # Skip meta pages, categories, etc.
        title = entry.title
        skip_prefixes = ("Catégorie:", "Portail:", "Projet:", "Aide:", "Modèle:", "Module:", "Wikipédia:", "MediaWiki:", "Spécial:", "Fichier:")
        if any(title.startswith(prefix) for prefix in skip_prefixes):
            return False

        return True
    except Exception:
        return False


def _load_checkpoint(collection_name: str, zim_path: Path) -> tuple[int, int, int] | None:
    """Load a checkpoint file if it matches the current collection and ZIM file.

    Returns (last_entry_index, article_count, chunk_count) or None.
    """
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        if (data["collection_name"] == collection_name
                and data["zim_filename"] == Path(zim_path).name):
            return (data["last_entry_index"], data["article_count"], data["chunk_count"])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _save_checkpoint(
    collection_name: str,
    zim_path: Path,
    last_entry_index: int,
    article_count: int,
    chunk_count: int,
) -> None:
    """Save indexing progress to checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({
            "collection_name": collection_name,
            "zim_filename": Path(zim_path).name,
            "last_entry_index": last_entry_index,
            "article_count": article_count,
            "chunk_count": chunk_count,
        }),
        encoding="utf-8",
    )


def index_zim(zim_path: Path, collection_name: str = "wikipedia") -> dict:
    """Read a ZIM file and index its content into ChromaDB.

    Uses a producer-consumer pipeline: the producer thread reads ZIM entries,
    cleans HTML and chunks text into batches; the main thread (consumer) encodes
    embeddings and inserts into ChromaDB. This overlaps I/O with encoding.

    Returns stats: {"article_count": int, "chunk_count": int}
    """
    from libzim.reader import Archive
    from sentence_transformers import SentenceTransformer

    config = _load_config()
    embeddings_config = config["embeddings"]
    vectordb_path = config["vectordb"]["path"]

    logger.info("Loading embedding model: %s", embeddings_config["model"])
    model = SentenceTransformer(embeddings_config["model"])

    logger.info("Opening ZIM file: %s", zim_path)
    archive = Archive(str(zim_path))

    client = chromadb.PersistentClient(path=vectordb_path)

    checkpoint = _load_checkpoint(collection_name, zim_path)
    if checkpoint is not None:
        start_index, article_count, chunk_count = checkpoint
        start_index += 1  # Resume after the last processed entry
        logger.info(
            "Reprise de l'indexation à l'entrée %d/%d (%.1f%%) — %d articles, %d chunks déjà indexés",
            start_index, archive.entry_count, start_index / archive.entry_count * 100,
            article_count, chunk_count,
        )
    else:
        start_index = 0
        article_count = 0
        chunk_count = 0
        # Delete existing collection for fresh indexing
        try:
            client.delete_collection(collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    chunk_size = embeddings_config["chunk_size"]
    chunk_overlap = embeddings_config["chunk_overlap"]
    batch_size = 5000
    batch_queue: queue.Queue = queue.Queue(maxsize=2)

    entry_count = archive.entry_count
    logger.info("Fichier ZIM : %d entrées à parcourir", entry_count)

    start_time = time.monotonic()
    _set_indexing_eta(datetime.now() + timedelta(hours=1))

    # --- Producer: reads ZIM, cleans HTML, chunks, puts batches in queue ---
    def _producer():
        nonlocal article_count, chunk_count
        p_article_count = article_count
        p_chunk_count = chunk_count
        batch_ids = []
        batch_docs = []
        batch_metadatas = []
        last_entry_idx = start_index

        try:
            for i in range(start_index, entry_count):
                try:
                    entry = archive._get_entry_by_id(i)
                except Exception:
                    continue

                if not _is_content_article(entry):
                    continue

                try:
                    item = entry.get_item()
                    content = bytes(item.content).decode("utf-8", errors="ignore")
                except Exception:
                    continue

                title = entry.title
                text = _clean_html(content)

                if len(text) < 50:
                    continue

                chunks = _chunk_text(text, chunk_size, chunk_overlap)
                if not chunks:
                    continue

                p_article_count += 1

                for j, chunk in enumerate(chunks):
                    doc_id = f"{collection_name}_{p_article_count}_{j}"
                    batch_ids.append(doc_id)
                    batch_docs.append(chunk)
                    batch_metadatas.append({
                        "title": title,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                    })
                    p_chunk_count += 1

                if len(batch_ids) >= batch_size:
                    # Progress reporting
                    abs_progress = i / entry_count
                    entries_done = i - start_index
                    elapsed = time.monotonic() - start_time
                    if entries_done > 0:
                        remaining_s = elapsed / entries_done * (entry_count - i)
                        _set_indexing_eta(datetime.now() + timedelta(seconds=remaining_s))
                        eta_dur = _format_eta(remaining_s)
                        eta_time = (datetime.now() + timedelta(seconds=remaining_s)).strftime("%H:%M")
                    else:
                        eta_dur = "?"
                        eta_time = "?"
                    logger.info(
                        "Lecture des articles : %.1f%% — %d articles — reste %s (fin ~%s)",
                        abs_progress * 100, p_article_count, eta_dur, eta_time,
                    )

                    batch_queue.put((
                        batch_ids, batch_docs, batch_metadatas,
                        i, p_article_count, p_chunk_count,
                    ))
                    batch_ids, batch_docs, batch_metadatas = [], [], []
                    last_entry_idx = i

            # Flush remaining partial batch
            if batch_ids:
                batch_queue.put((
                    batch_ids, batch_docs, batch_metadatas,
                    last_entry_idx, p_article_count, p_chunk_count,
                ))
        finally:
            # Sentinel: signals consumer that production is done
            batch_queue.put(None)

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    # --- Consumer (main thread): encodes + inserts into ChromaDB ---
    while True:
        batch = batch_queue.get()
        if batch is None:
            break

        batch_ids, batch_docs, batch_metadatas, last_entry_idx, article_count, chunk_count = batch
        logger.info("Encodage et insertion de %d chunks dans ChromaDB...", len(batch_ids))
        batch_embeddings = model.encode(batch_docs, batch_size=256).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )
        _save_checkpoint(collection_name, zim_path, last_entry_idx, article_count, chunk_count)

    producer_thread.join()

    # Indexation complete — remove checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    _set_indexing_eta(None)

    elapsed = time.monotonic() - start_time
    stats = {"article_count": article_count, "chunk_count": chunk_count}
    logger.info(
        "Indexation terminée : %d articles, %d chunks en %s",
        article_count, chunk_count, _format_eta(elapsed),
    )
    return stats
