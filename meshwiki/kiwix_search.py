"""Kiwix ZIM full-text search fallback for use when ChromaDB index is unavailable."""

import html as html_lib
import logging
import re
import threading

logger = logging.getLogger(__name__)

_zim_path: str | None = None
_zim_lock = threading.Lock()
_archive = None
_archive_path: str | None = None

# French stop words — stripped from queries before searching
_STOP_WORDS = frozenset(
    "le la les l un une des du de d et en est au aux"
    " que qui qu ce c cette ces quel quelle quels quelles"
    " je tu il elle on nous vous ils elles"
    " mon ton son ma ta sa mes tes ses notre votre leur nos vos leurs"
    " me te se"
    " ne pas plus"
    " dans par pour sur avec sans"
    " a à y"
    " être avoir fait faire"
    " tout tous toute toutes"
    " où ou comment combien pourquoi quand quel quelle quels quelles"
    .split()
)


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


def _clean_html(raw_html: str) -> str:
    """Extract plain text from HTML, removing tags, entities, and extra whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_keywords(query: str) -> str:
    """Extract meaningful keywords from a natural-language query.

    Strips punctuation, stop words, and short tokens to improve
    Xapian full-text search results.
    """
    # Remove punctuation except hyphens (for compound words)
    cleaned = re.sub(r"[^\w\s-]", " ", query)
    words = cleaned.lower().split()
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(keywords)


def _read_entry(archive, path: str, max_chars: int) -> dict | None:
    """Read a ZIM entry and return a cleaned result dict, or None on failure."""
    try:
        entry = archive.get_entry_by_path(path)
        title = entry.title
        item = entry.get_item()
        raw_html = bytes(item.content).decode("utf-8", errors="ignore")
        text = _clean_html(raw_html)
        if len(text) > max_chars:
            truncated = text[:max_chars]
            last_space = truncated.rfind(" ")
            if last_space > max_chars // 2:
                truncated = truncated[:last_space]
            text = truncated + "..."
        if text:
            return {"title": title, "content": text}
    except Exception as e:
        logger.debug("Failed to read ZIM entry %s: %s", path, e)
    return None


def search(query: str, num_results: int = 3, max_chars_per_result: int = 500) -> list[dict]:
    """Search the ZIM file for articles matching the query.

    Combines title-based suggestion search with full-text search
    for better recall. Keywords are extracted from the query to
    improve Xapian matching on natural-language questions.

    Returns a list of {"title": str, "content": str} dicts.
    Returns an empty list if no ZIM is available or search fails.
    """
    archive = _get_archive()
    if archive is None:
        return []

    keywords = _extract_keywords(query)
    logger.debug("Kiwix search: query=%r → keywords=%r", query, keywords)
    if not keywords:
        keywords = query  # Fallback to raw query if all words were filtered

    seen_paths: set[str] = set()
    output: list[dict] = []

    # 1. Title/suggestion search — best for direct article matches
    try:
        from libzim.suggestion import SuggestionSearcher

        suggestion_searcher = SuggestionSearcher(archive)
        suggestion = suggestion_searcher.suggest(keywords)
        for path in suggestion.getResults(0, num_results):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            result = _read_entry(archive, path, max_chars_per_result)
            if result:
                output.append(result)
                logger.debug("Kiwix suggestion hit: %s", result["title"])
    except Exception as e:
        logger.debug("Kiwix suggestion search failed: %s", e)

    # 2. Full-text search — complements title search with content matches
    remaining = num_results - len(output)
    if remaining > 0:
        try:
            from libzim.search import Query, Searcher

            searcher = Searcher(archive)
            zim_query = Query().set_query(keywords)
            results = searcher.search(zim_query)
            for path in results.getResults(0, remaining + 3):  # Fetch extra to account for dedup
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                result = _read_entry(archive, path, max_chars_per_result)
                if result:
                    output.append(result)
                    logger.debug("Kiwix fulltext hit: %s", result["title"])
                if len(output) >= num_results:
                    break
        except Exception as e:
            logger.debug("Kiwix full-text search failed: %s", e)

    if not output:
        logger.info("Kiwix search: no results for %r (keywords: %r)", query, keywords)
    else:
        logger.info("Kiwix search: %d results for %r", len(output), query)

    return output
