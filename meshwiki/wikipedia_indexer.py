"""Index Wikipedia ZIM files into ChromaDB for semantic search."""

import json
import logging
import re
from pathlib import Path

import chromadb
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("data/indexing_checkpoint.json")


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _clean_html(html: str) -> str:
    """Extract plain text from HTML, removing tags and extra whitespace."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
            "Resuming indexation from entry %d (%d articles, %d chunks already indexed)",
            start_index, article_count, chunk_count,
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
    batch_ids = []
    batch_docs = []
    batch_metadatas = []
    batch_size = 5000

    entry_count = archive.entry_count
    logger.info("ZIM contains %d entries, processing articles...", entry_count)

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

        if len(text) < 50:  # Skip very short articles
            continue

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            continue

        article_count += 1

        for j, chunk in enumerate(chunks):
            doc_id = f"{collection_name}_{article_count}_{j}"
            batch_ids.append(doc_id)
            batch_docs.append(chunk)
            batch_metadatas.append({
                "title": title,
                "chunk_index": j,
                "total_chunks": len(chunks),
            })
            chunk_count += 1

        # Flush batch: encode + insert together
        if len(batch_ids) >= batch_size:
            batch_embeddings = model.encode(batch_docs).tolist()
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            batch_ids, batch_docs, batch_metadatas = [], [], []
            _save_checkpoint(collection_name, zim_path, i, article_count, chunk_count)

        # Progress reporting
        if article_count % 1000 == 0:
            progress = (i / entry_count) * 100
            logger.info("Progress: %.1f%% — %d articles, %d chunks", progress, article_count, chunk_count)

    # Flush remaining
    if batch_ids:
        batch_embeddings = model.encode(batch_docs).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    # Indexation complete — remove checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    stats = {"article_count": article_count, "chunk_count": chunk_count}
    logger.info("Indexing complete: %d articles, %d chunks", article_count, chunk_count)
    return stats
