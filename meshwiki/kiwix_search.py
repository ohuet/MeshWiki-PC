"""Kiwix ZIM full-text search fallback for use during ChromaDB indexation."""

import logging
import re
import threading

logger = logging.getLogger(__name__)

_zim_path: str | None = None
_zim_lock = threading.Lock()
_archive = None
_archive_path: str | None = None


def set_zim_path(path) -> None:
    """Set the ZIM file path (thread-safe). Pass None to clear."""
    global _zim_path, _archive, _archive_path
    with _zim_lock:
        _zim_path = str(path) if path is not None else None
        # Invalidate cached archive if path changed
        if _archive_path != _zim_path:
            _archive = None
            _archive_path = None


def get_zim_path() -> str | None:
    """Return the current ZIM file path, or None if not set."""
    with _zim_lock:
        return _zim_path


def _get_archive():
    """Lazy-open and cache the libzim Archive."""
    global _archive, _archive_path
    path = get_zim_path()
    if path is None:
        return None
    with _zim_lock:
        if _archive is not None and _archive_path == path:
            return _archive
        from libzim.reader import Archive
        _archive = Archive(path)
        _archive_path = path
        return _archive


def _clean_html(html: str) -> str:
    """Extract plain text from HTML, removing tags and extra whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def search(query: str, num_results: int = 3, max_chars_per_result: int = 500) -> list[dict]:
    """Search the ZIM full-text index for articles matching the query.

    Returns a list of {"title": str, "content": str} dicts.
    Returns an empty list if no ZIM is available or search fails.
    """
    archive = _get_archive()
    if archive is None:
        return []

    try:
        from libzim.search import Query, Searcher

        searcher = Searcher(archive)
        zim_query = Query().set_query(query)
        results = searcher.search(zim_query)

        output = []
        for path in results.getResults(0, num_results):
            try:
                entry = archive.get_entry_by_path(path)
                title = entry.title
                item = entry.get_item()
                html = bytes(item.content).decode("utf-8", errors="ignore")
                text = _clean_html(html)
                if len(text) > max_chars_per_result:
                    # Truncate at word boundary
                    truncated = text[:max_chars_per_result]
                    last_space = truncated.rfind(" ")
                    if last_space > max_chars_per_result // 2:
                        truncated = truncated[:last_space]
                    text = truncated + "..."
                if text:
                    output.append({"title": title, "content": text})
            except Exception as e:
                logger.debug("Failed to read ZIM entry %s: %s", path, e)
                continue

        return output

    except Exception as e:
        logger.warning("Kiwix search failed: %s", e)
        return []
