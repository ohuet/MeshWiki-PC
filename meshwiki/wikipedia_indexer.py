"""Index Wikipedia ZIM files into ChromaDB for semantic search."""

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import chromadb
from lxml import html as lxml_html

from meshwiki import config
from meshwiki.progress import ProgressDisplay, format_bar

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = Path("data/indexing_checkpoint.json")

_indexing_eta: datetime | None = None
_force_fresh = False
_indexing_lock = threading.Lock()


def has_checkpoint() -> bool:
    """Return True if an indexing checkpoint file exists (partial index in progress)."""
    return CHECKPOINT_FILE.exists() or CHECKPOINT_FILE.with_suffix(".tmp").exists()


def clear_checkpoints() -> None:
    """Delete all indexing checkpoint files to force a fresh reindexation."""
    for suffix in (".json", ".tmp"):
        path = CHECKPOINT_FILE.with_suffix(suffix)
        if path.exists():
            path.unlink()
            logger.info("Checkpoint supprimé : %s", path)


def get_indexing_eta() -> datetime | None:
    """Return the estimated completion time of an ongoing indexation, or None."""
    with _indexing_lock:
        return _indexing_eta


def clear_indexing_eta() -> None:
    """Clear the indexing ETA (e.g. after a failure)."""
    _set_indexing_eta(None)


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


def _gpu_lock_clocks() -> bool:
    """Lock GPU clocks to max performance to prevent sleep throttling.

    Returns True if clocks were successfully locked.
    """
    try:
        subprocess.run(
            ["nvidia-smi", "-pm", "1"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["nvidia-smi", "--lock-gpu-clocks=300,9999"],
            capture_output=True, check=True,
        )
        logger.info("GPU clocks locked to max performance")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _gpu_unlock_clocks() -> None:
    """Release GPU clock lock."""
    try:
        subprocess.run(
            ["nvidia-smi", "--reset-gpu-clocks"],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["nvidia-smi", "-pm", "0"],
            capture_output=True, check=True,
        )
        logger.info("GPU clocks unlocked")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def _clean_html(html: str) -> str:
    """Extract plain text from HTML, removing tags and extra whitespace."""
    try:
        doc = lxml_html.fromstring(html)
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    for el in doc.iter("script", "style"):
        el.drop_tree()
    # Ensure a space between adjacent block/table elements to avoid
    # cell values being concatenated (e.g. "<td>1</td><td>2</td>" → "1 2")
    for el in doc.iter("td", "th", "caption", "li", "p", "h1", "h2", "h3",
                       "h4", "h5", "h6", "br", "div", "tr"):
        if el.tail is None or not el.tail.strip():
            el.tail = " " + (el.tail or "")
    text = doc.text_content()
    text = re.sub(r"\s+", " ", text).strip()
    # Strip the Kiwix/Wikipedia license footer
    text = re.sub(r"\s*Cet article est issu de Wikipédia\..*$", "", text)
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


def _try_read_checkpoint(path: Path, collection_name: str, zim_path: Path) -> tuple[int, int, int] | None:
    """Try to read and validate a single-ZIM checkpoint from the given path."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("multi"):
            return None  # Multi-ZIM checkpoint — not for us
        if (data["collection_name"] == collection_name
                and data["zim_filename"] == Path(zim_path).name):
            return (data["last_entry_index"], data["article_count"], data["chunk_count"])
        logger.info(
            "Checkpoint ignoré (collection=%s vs %s, zim=%s vs %s)",
            data.get("collection_name"), collection_name,
            data.get("zim_filename"), Path(zim_path).name,
        )
    except json.JSONDecodeError as e:
        logger.warning("Checkpoint corrompu dans %s (%s), ignoré", path, e)
    except (KeyError, TypeError) as e:
        logger.warning("Checkpoint corrompu dans %s (%s), ignoré", path, e)
    return None


def _load_checkpoint(collection_name: str, zim_path: Path) -> tuple[int, int, int] | None:
    """Load a checkpoint file if it matches the current collection and ZIM file.

    Tries .json first, then .tmp as fallback (survives if .json was lost
    during a non-atomic replace on Windows).
    Returns (last_entry_index, article_count, chunk_count) or None.
    """
    result = _try_read_checkpoint(CHECKPOINT_FILE, collection_name, zim_path)
    if result is not None:
        return result

    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    result = _try_read_checkpoint(tmp, collection_name, zim_path)
    if result is not None:
        logger.info("Checkpoint récupéré depuis %s (le .json était absent)", tmp)
        return result

    if not CHECKPOINT_FILE.exists() and not tmp.exists():
        logger.info("Pas de checkpoint trouvé (%s n'existe pas)", CHECKPOINT_FILE)
    return None


def _save_checkpoint(
    collection_name: str,
    zim_path: Path,
    last_entry_index: int,
    article_count: int,
    chunk_count: int,
) -> None:
    """Save indexing progress to checkpoint file.

    Writes to .tmp first (with fsync), then copies to .json (with fsync).
    Both files are kept on disk so _load_checkpoint can recover from either
    if the process is killed mid-write. This avoids the non-atomic
    Path.replace() on Windows (which deletes target before renaming source).
    """
    import shutil
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    data = json.dumps({
        "collection_name": collection_name,
        "zim_filename": Path(zim_path).name,
        "last_entry_index": last_entry_index,
        "article_count": article_count,
        "chunk_count": chunk_count,
    })
    # Step 1: write .tmp with fsync
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    # Step 2: copy .tmp → .json with fsync (both files survive)
    shutil.copy2(tmp, CHECKPOINT_FILE)
    with open(CHECKPOINT_FILE, "r+b") as f:
        os.fsync(f.fileno())


def _encode_with_retry(
    texts: list[str],
    remote_encode_fn,
    local_encode_fn,
    label: str = "",
) -> list[list[float]]:
    """Encode texts with resilient retry strategy.

    Retry cycle:
      1. Remote attempt 1
      2. Wait 30s → Remote attempt 2
      3. Wait 60s → Remote attempt 3
      4. Local attempt 1
      5. Wait 30s → Local attempt 2
      6. Wait 60s → Local attempt 3
      7. Wait 30 min → restart from step 1

    Args:
        texts: Texts to encode.
        remote_encode_fn: Callable returning embeddings list or None on failure.
        local_encode_fn: Callable returning embeddings list (may raise on CUDA error).
                         None if no local model available.
        label: Description for log messages.
    """
    cycle = 0
    while True:
        cycle += 1
        if cycle > 1:
            logger.warning(
                "Encodage %s — nouveau cycle de retry (#%d) après 30 min d'attente",
                label, cycle,
            )

        # Remote attempts
        for attempt, wait in [(1, 0), (2, 30), (3, 60)]:
            if wait:
                logger.info("Attente %ds avant retry distant #%d %s...", wait, attempt, label)
                time.sleep(wait)
            result = remote_encode_fn(texts)
            if result is not None:
                return result
            logger.warning("Échec distant #%d %s (%d chunks)", attempt, label, len(texts))

        # Local attempts
        if local_encode_fn is not None:
            for attempt, wait in [(1, 0), (2, 30), (3, 60)]:
                if wait:
                    logger.info("Attente %ds avant retry local #%d %s...", wait, attempt, label)
                    time.sleep(wait)
                try:
                    return local_encode_fn(texts)
                except Exception as e:
                    logger.warning("Échec local #%d %s : %s", attempt, label, e)

        # All failed — wait 30 minutes and restart
        logger.error(
            "Tous les encodages ont échoué %s — attente de 30 min avant nouveau cycle...",
            label,
        )
        time.sleep(30 * 60)


def index_zim(
    zim_path: Path,
    collection_name: str = "wikipedia",
    db_path: str | None = None,
    skip_titles: set[str] | None = None,
    append: bool = False,
    on_progress: 'Callable[[int, int, int], None] | None' = None,
    resume_from: tuple[int, int, int] | None = None,
) -> dict:
    """Read a ZIM file and index its content into ChromaDB.

    Uses a producer-consumer pipeline: the producer thread reads ZIM entries,
    cleans HTML and chunks text into batches; the main thread (consumer) encodes
    embeddings and inserts into ChromaDB. This overlaps I/O with encoding.

    Args:
        db_path: ChromaDB database directory. Defaults to config vectordb.path.
        skip_titles: Set of article titles to skip (already indexed by a prior ZIM).
        append: If True, do not delete the existing collection on fresh start.
        on_progress: Callback(last_entry_index, article_count, chunk_count) called
            after each batch. Replaces _save_checkpoint when provided (multi-ZIM mode).
        resume_from: (last_entry_index, article_count, chunk_count) to resume from.
            Replaces _load_checkpoint when provided (multi-ZIM mode).

    Returns stats: {"article_count": int, "chunk_count": int, "indexed_titles": set[str]}
    """
    from libzim.reader import Archive
    from sentence_transformers import SentenceTransformer

    from meshwiki import remote_embeddings

    cfg = config.load_config()
    tz = ZoneInfo(cfg.get("meshtastic_timezone", "UTC"))
    embeddings_config = cfg["embeddings"]
    vectordb_path = db_path or cfg["vectordb"]["path"]

    model_name = embeddings_config["model"]
    truncate_dim = embeddings_config.get("truncate_dim")
    model_kwargs = {"dtype": "float16"}

    remote_configured = remote_embeddings.is_configured(cfg)
    _local_model = None

    def _get_local_model():
        nonlocal _local_model
        if _local_model is None:
            logger.info("Loading local embedding model: %s", model_name)
            try:
                _local_model = SentenceTransformer(
                    model_name, truncate_dim=truncate_dim, local_files_only=True,
                    model_kwargs=model_kwargs,
                )
            except OSError:
                logger.info("Downloading embedding model: %s (first time)", model_name)
                _local_model = SentenceTransformer(
                    model_name, truncate_dim=truncate_dim,
                    model_kwargs=model_kwargs,
                )
        return _local_model

    local_workers = 0
    if not remote_configured:
        _get_local_model()  # Load immediately (original behaviour)
    else:
        remote_config = embeddings_config["remote"]
        local_workers = remote_config.get("local_workers", 0)
        max_concurrent = remote_config.get("max_concurrent", 4)
        logger.info("Remote embedding configured (%s)", remote_config["base_url"])
        remote_embeddings.reset()
        if local_workers > 0:
            logger.info(
                "Hybrid mode: %d remote + %d local workers",
                max_concurrent, local_workers,
            )
            _get_local_model()  # Also need local model in hybrid mode

    logger.info("Opening ZIM file: %s", zim_path)
    archive = Archive(str(zim_path))

    client = chromadb.PersistentClient(path=vectordb_path)

    # Determine resume state: from parameter (multi-ZIM) or from checkpoint file (standalone)
    checkpoint = resume_from or _load_checkpoint(collection_name, zim_path)
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
        # Delete existing collection for fresh indexing (unless appending)
        if not append:
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
    remaining_entries = entry_count - start_index
    if remaining_entries > 0 and start_index > 0:
        # Rough initial ETA: assume ~200 entries/s (adjusted after first batch)
        _set_indexing_eta(datetime.now(tz) + timedelta(seconds=remaining_entries / 200))
    else:
        _set_indexing_eta(datetime.now(tz) + timedelta(hours=1))

    gpu_locked = _gpu_lock_clocks()

    progress = ProgressDisplay()
    progress.start()
    if start_index > 0:
        progress.set_progress(
            start_index / entry_count * 100, article_count, "?", "?",
        )

    # --- Producer: reads ZIM, cleans HTML, chunks, puts batches in queue ---
    producer_completed = threading.Event()
    p_indexed_titles: set[str] = set()

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

                title = entry.title

                # Skip titles already indexed by a higher-priority ZIM
                if skip_titles is not None and title in skip_titles:
                    continue

                try:
                    item = entry.get_item()
                    content = bytes(item.content).decode("utf-8", errors="ignore")
                except Exception:
                    continue

                text = _clean_html(content)

                if len(text) < 50:
                    continue

                chunks = _chunk_text(text, chunk_size, chunk_overlap)
                if not chunks:
                    continue

                p_article_count += 1
                p_indexed_titles.add(title)

                for j, chunk in enumerate(chunks):
                    doc_id = f"{collection_name}_{p_article_count}_{j}"
                    batch_ids.append(doc_id)
                    batch_docs.append(f"{title} — {chunk}")
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
                        _set_indexing_eta(datetime.now(tz) + timedelta(seconds=remaining_s))
                        eta_dur = _format_eta(remaining_s)
                        eta_time = (datetime.now() + timedelta(seconds=remaining_s)).strftime("%H:%M")
                    else:
                        eta_dur = "?"
                        eta_time = "?"
                    progress.set_progress(abs_progress * 100, p_article_count, eta_dur, eta_time)

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
            producer_completed.set()
        finally:
            # Sentinel: signals consumer that production is done
            batch_queue.put(None)

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    # --- Writer thread: inserts into ChromaDB while consumer encodes next batch ---
    write_queue: queue.Queue = queue.Queue(maxsize=2)
    writer_error: list[Exception] = []

    # Choose checkpoint save function: callback (multi-ZIM) or file (standalone)
    _progress_fn = on_progress if on_progress is not None else (
        lambda idx, art, chk: _save_checkpoint(collection_name, zim_path, idx, art, chk)
    )

    def _writer():
        while True:
            item = write_queue.get()
            if item is None:
                break
            w_ids, w_docs, w_embeddings, w_metadatas, w_entry_idx, w_art, w_chunk = item
            try:
                progress.set_info("Insertion de %d chunks dans ChromaDB..." % len(w_ids))
                collection.add(
                    ids=w_ids,
                    documents=w_docs,
                    embeddings=w_embeddings,
                    metadatas=w_metadatas,
                )
                _progress_fn(w_entry_idx, w_art, w_chunk)
                progress.set_info("")
            except Exception as e:
                logger.error("Writer error: %s", e)
                writer_error.append(e)
                break

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    # --- Consumer (main thread): encodes embeddings, sends to writer ---
    consumer_error = None
    while True:
        progress.set_info("Extraction des articles...")
        batch = batch_queue.get()
        if batch is None:
            break
        progress.set_info("")

        if writer_error:
            break

        try:
            batch_ids, batch_docs, batch_metadatas, last_entry_idx, article_count, chunk_count = batch
            progress.set_info("Encodage de %d chunks..." % len(batch_ids))

            if remote_configured and local_workers > 0:
                # Hybrid mode: work-stealing between remote and local GPU
                from concurrent.futures import ThreadPoolExecutor
                remote_batch_size = embeddings_config["remote"].get("batch_size", 256)
                remote_params = remote_embeddings.prepare(cfg)

                # Split into sub-batches
                sb_list = []
                for sb_start in range(0, len(batch_docs), remote_batch_size):
                    sb_list.append(batch_docs[sb_start : sb_start + remote_batch_size])

                work_queue = queue.Queue()
                for i, sb in enumerate(sb_list):
                    work_queue.put((i, sb))

                sb_results: list[list[list[float]] | None] = [None] * len(sb_list)
                local_model = _get_local_model()
                local_lock = threading.Lock()

                total_sb = len(sb_list)
                remote_done = 0
                local_done = 0
                sb_done_lock = threading.Lock()
                SUB_BAR_W = 10

                def _update_hybrid_progress(worker_type: str):
                    nonlocal remote_done, local_done
                    with sb_done_lock:
                        if worker_type == "remote":
                            remote_done += 1
                        else:
                            local_done += 1
                        r, l = remote_done, local_done
                    r_pct = r / total_sb * 100 if total_sb else 0
                    l_pct = l / total_sb * 100 if total_sb else 0
                    progress.set_sub_progress(
                        "Embedding distant : Batch %d/%d [%s] | Embedding local : Batch %d/%d [%s]"
                        % (r, total_sb, format_bar(r_pct, SUB_BAR_W),
                           l, total_sb, format_bar(l_pct, SUB_BAR_W))
                    )

                def _remote_worker():
                    while True:
                        try:
                            idx, docs = work_queue.get_nowait()
                        except queue.Empty:
                            return

                        def _remote_single(texts):
                            return remote_embeddings.encode_single(texts, remote_params)

                        def _local_single(texts):
                            with local_lock:
                                return local_model.encode(
                                    texts, batch_size=64, show_progress_bar=False,
                                ).tolist()

                        sb_results[idx] = _encode_with_retry(
                            docs, _remote_single, _local_single,
                            label=f"(sub-batch {idx})",
                        )
                        _update_hybrid_progress("remote")

                def _local_worker():
                    while True:
                        try:
                            idx, docs = work_queue.get_nowait()
                        except queue.Empty:
                            return
                        with local_lock:
                            embs = local_model.encode(
                                docs, batch_size=64, show_progress_bar=False,
                            ).tolist()
                        sb_results[idx] = embs
                        _update_hybrid_progress("local")

                with ThreadPoolExecutor(max_workers=max_concurrent + local_workers) as pool:
                    futs = []
                    for _ in range(max_concurrent):
                        futs.append(pool.submit(_remote_worker))
                    for _ in range(local_workers):
                        futs.append(pool.submit(_local_worker))
                    for f in futs:
                        f.result()

                batch_embeddings = []
                for r in sb_results:
                    batch_embeddings.extend(r)

            elif remote_configured:
                # Remote-only mode (with retry)
                SUB_BAR_W = 10

                def _on_remote_sub_batch(done: int, total: int):
                    pct = done / total * 100 if total else 0
                    progress.set_sub_progress(
                        "Embedding distant : Batch %d/%d [%s]" % (done, total, format_bar(pct, SUB_BAR_W))
                    )

                def _remote_batch(texts):
                    return remote_embeddings.encode_batch(
                        texts, cfg, on_sub_batch=_on_remote_sub_batch,
                    )

                def _local_batch(texts):
                    m = _get_local_model()
                    return m.encode(texts, batch_size=64, show_progress_bar=False).tolist()

                batch_embeddings = _encode_with_retry(
                    batch_docs, _remote_batch, _local_batch,
                )

            else:
                # Local-only mode
                local_model = _get_local_model()
                local_batch_size = 64
                SUB_BAR_W = 10
                total_sb = (len(batch_docs) + local_batch_size - 1) // local_batch_size
                batch_embeddings = []
                for sb_idx in range(0, len(batch_docs), local_batch_size):
                    sb_docs = batch_docs[sb_idx : sb_idx + local_batch_size]
                    sb_embs = local_model.encode(
                        sb_docs, batch_size=local_batch_size, show_progress_bar=False,
                    ).tolist()
                    batch_embeddings.extend(sb_embs)
                    done = sb_idx // local_batch_size + 1
                    pct = done / total_sb * 100 if total_sb else 0
                    progress.set_sub_progress(
                        "Embedding local : Batch %d/%d [%s]" % (done, total_sb, format_bar(pct, SUB_BAR_W))
                    )

            progress.set_info("")
            progress.set_sub_progress("")

            write_queue.put((
                batch_ids, batch_docs, batch_embeddings, batch_metadatas,
                last_entry_idx, article_count, chunk_count,
            ))
        except Exception as e:
            consumer_error = e
            logger.error("Erreur dans la boucle d'encodage : %s", e)
            break

    # Signal writer to stop and wait for it
    write_queue.put(None)
    writer_thread.join()

    producer_thread.join()

    # Always cleanup progress, GPU, and ETA
    progress.stop()
    if gpu_locked:
        _gpu_unlock_clocks()
    _set_indexing_eta(None)

    # Check if consumer encountered an error
    if consumer_error is not None:
        raise consumer_error

    # Check if writer or producer failed
    if writer_error:
        logger.error("Indexation échouée — erreur d'écriture ChromaDB")
        raise writer_error[0]

    if not producer_completed.is_set():
        logger.warning("Indexation interrompue — checkpoint conservé pour reprise")
        raise RuntimeError("Indexation interrupted")

    # Indexation complete — archive checkpoint files (standalone mode only;
    # in multi-ZIM mode the caller manages the checkpoint lifecycle)
    if on_progress is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for cp in (CHECKPOINT_FILE, CHECKPOINT_FILE.with_suffix(".tmp")):
            if cp.exists():
                done_name = f"indexing_checkpoint_done_at_{timestamp}{cp.suffix}"
                dest = cp.parent / done_name
                cp.rename(dest)
                logger.info("Checkpoint archivé : %s → %s", cp.name, done_name)

    elapsed = time.monotonic() - start_time
    stats = {"article_count": article_count, "chunk_count": chunk_count, "indexed_titles": p_indexed_titles}
    logger.info(
        "Indexation terminée : %d articles, %d chunks en %s",
        article_count, chunk_count, _format_eta(elapsed),
    )
    return stats


def _collect_titles(zim_path: Path) -> set[str]:
    """Quickly collect all content article titles from a ZIM file (no content read)."""
    from libzim.reader import Archive

    archive = Archive(str(zim_path))
    titles: set[str] = set()
    for i in range(archive.entry_count):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue
        if _is_content_article(entry):
            titles.add(entry.title)
    logger.info("Titres collectés depuis %s : %d articles", zim_path.name, len(titles))
    return titles


def _load_multi_checkpoint(
    collection_name: str, zim_names: list[str],
) -> dict | None:
    """Load a multi-ZIM checkpoint if it matches the current ZIM list."""
    for path in (CHECKPOINT_FILE, CHECKPOINT_FILE.with_suffix(".tmp")):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data.get("multi"):
                continue
            if data["collection_name"] != collection_name:
                continue
            if data["zim_files"] != zim_names:
                logger.info(
                    "Checkpoint multi-ZIM ignoré (fichiers changés : %s vs %s)",
                    data["zim_files"], zim_names,
                )
                continue
            return data
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Checkpoint multi-ZIM corrompu (%s) : %s", path, e)
    return None


def _save_multi_checkpoint(
    collection_name: str,
    zim_names: list[str],
    completed_zims: list[str],
    current_zim: str | None,
    current_entry_index: int,
    current_article_count: int,
    current_chunk_count: int,
    total_article_count: int,
    total_chunk_count: int,
) -> None:
    """Save multi-ZIM indexing progress (including current ZIM state)."""
    import shutil
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    data = json.dumps({
        "multi": True,
        "collection_name": collection_name,
        "zim_files": zim_names,
        "completed_zims": completed_zims,
        "current_zim": current_zim,
        "current_entry_index": current_entry_index,
        "current_article_count": current_article_count,
        "current_chunk_count": current_chunk_count,
        "total_article_count": total_article_count,
        "total_chunk_count": total_chunk_count,
    })
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    shutil.copy2(tmp, CHECKPOINT_FILE)
    with open(CHECKPOINT_FILE, "r+b") as f:
        os.fsync(f.fileno())


def index_all_zims(
    zim_paths: list[Path],
    collection_name: str = "wikipedia",
    db_path: str | None = None,
) -> dict:
    """Index multiple ZIM files with deduplication (smaller ZIMs have priority).

    Args:
        zim_paths: ZIM files in priority order (smallest first).
        collection_name: ChromaDB collection name.
        db_path: ChromaDB database directory.

    Returns aggregated stats: {"article_count": int, "chunk_count": int}
    """
    if not zim_paths:
        logger.warning("Aucun fichier ZIM à indexer")
        return {"article_count": 0, "chunk_count": 0}

    if len(zim_paths) == 1:
        stats = index_zim(zim_paths[0], collection_name, db_path)
        stats.pop("indexed_titles", None)
        return stats

    zim_names = [p.name for p in zim_paths]
    checkpoint = _load_multi_checkpoint(collection_name, zim_names)

    completed_zims: list[str] = []
    seen_titles: set[str] = set()
    total_articles = 0
    total_chunks = 0
    is_first_zim = True

    if checkpoint is not None:
        completed_zims = list(checkpoint["completed_zims"])
        total_articles = checkpoint["total_article_count"]
        total_chunks = checkpoint["total_chunk_count"]
        logger.info(
            "Reprise multi-ZIM : %d/%d ZIM complétés, %d articles, %d chunks",
            len(completed_zims), len(zim_names), total_articles, total_chunks,
        )
        # Rebuild seen_titles from completed ZIMs
        for zim_path in zim_paths:
            if zim_path.name in completed_zims:
                logger.info("Collecte des titres depuis %s (déjà indexé)...", zim_path.name)
                seen_titles |= _collect_titles(zim_path)
                is_first_zim = False
    else:
        # Fresh start: delete the collection
        cfg = config.load_config()
        vectordb_path = db_path or cfg["vectordb"]["path"]
        client = chromadb.PersistentClient(path=vectordb_path)
        try:
            client.delete_collection(collection_name)
            logger.info("Collection '%s' supprimée pour indexation multi-ZIM", collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    for zim_path in zim_paths:
        if zim_path.name in completed_zims:
            continue

        logger.info(
            "=== Indexation de %s (%d/%d) — %d titres à ignorer ===",
            zim_path.name,
            len(completed_zims) + 1, len(zim_names),
            len(seen_titles),
        )

        # Resume state for current ZIM (from multi checkpoint)
        resume_from = None
        if checkpoint and checkpoint.get("current_zim") == zim_path.name:
            resume_from = (
                checkpoint["current_entry_index"],
                checkpoint["current_article_count"],
                checkpoint["current_chunk_count"],
            )

        # Callback: save multi checkpoint with current ZIM progress
        def _on_progress(entry_idx, art_count, chunk_count, _zim=zim_path.name):
            _save_multi_checkpoint(
                collection_name, zim_names, completed_zims,
                _zim, entry_idx, art_count, chunk_count,
                total_articles + art_count,
                total_chunks + chunk_count,
            )

        stats = index_zim(
            zim_path,
            collection_name,
            db_path,
            skip_titles=seen_titles if seen_titles else None,
            append=not is_first_zim,
            on_progress=_on_progress,
            resume_from=resume_from,
        )

        total_articles += stats["article_count"]
        total_chunks += stats["chunk_count"]
        seen_titles |= _collect_titles(zim_path)
        completed_zims.append(zim_path.name)
        is_first_zim = False

        _save_multi_checkpoint(
            collection_name, zim_names, completed_zims,
            None, 0, 0, 0,
            total_articles, total_chunks,
        )

        logger.info(
            "ZIM %s terminé — total cumulé : %d articles, %d chunks",
            zim_path.name, total_articles, total_chunks,
        )

    # Archive multi checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for cp in (CHECKPOINT_FILE, CHECKPOINT_FILE.with_suffix(".tmp")):
        if cp.exists():
            done_name = f"indexing_checkpoint_done_at_{timestamp}{cp.suffix}"
            dest = cp.parent / done_name
            cp.rename(dest)

    logger.info(
        "Indexation multi-ZIM terminée : %d ZIM, %d articles, %d chunks",
        len(zim_paths), total_articles, total_chunks,
    )
    return {"article_count": total_articles, "chunk_count": total_chunks}
