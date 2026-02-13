"""Kiwix ZIM full-text search fallback for use when ChromaDB index is unavailable."""

import html as html_lib
import logging
import re
import threading

logger = logging.getLogger(__name__)

_zim_paths: list[str] = []
_zim_lock = threading.Lock()
_archives: dict[str, object] = {}  # path → Archive cache
_disabled = False

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


def set_zim_paths(paths: list) -> None:
    """Set the list of ZIM file paths (thread-safe), ordered by priority."""
    global _zim_paths, _archives
    with _zim_lock:
        new_paths = [str(p) for p in paths]
        # Close archives that are no longer in the list
        removed = set(_archives.keys()) - set(new_paths)
        for r in removed:
            _archives.pop(r, None)
        _zim_paths = new_paths


def set_zim_path(path) -> None:
    """Legacy single-path setter. Pass None to clear."""
    if path is None:
        set_zim_paths([])
    else:
        set_zim_paths([path])


def set_disabled(value: bool) -> None:
    """Disable Kiwix search entirely (used by --nowiki flag)."""
    global _disabled
    _disabled = value


def has_zim_paths() -> bool:
    """Return True if at least one ZIM file is configured and not disabled."""
    if _disabled:
        return False
    with _zim_lock:
        return len(_zim_paths) > 0


def get_zim_path() -> str | None:
    """Legacy getter — return the first ZIM path, or None if empty/disabled."""
    if _disabled:
        return None
    with _zim_lock:
        return _zim_paths[0] if _zim_paths else None


def _get_archive(path: str):
    """Lazy-open and cache a libzim Archive for the given path."""
    with _zim_lock:
        if path in _archives:
            return _archives[path]
    from libzim.reader import Archive
    archive = Archive(path)
    with _zim_lock:
        _archives[path] = archive
    return archive


def _get_archives() -> list[tuple[str, object]]:
    """Return list of (path, archive) for all current ZIM paths, in priority order."""
    if _disabled:
        return []
    with _zim_lock:
        paths = list(_zim_paths)
    result = []
    for path in paths:
        try:
            archive = _get_archive(path)
            result.append((path, archive))
        except Exception as e:
            logger.warning("Impossible d'ouvrir le ZIM %s : %s", path, e)
    return result


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


def _search_single_archive(
    archive, keywords: str, num_results: int, max_chars: int,
    seen_titles: set[str],
) -> list[dict]:
    """Search a single ZIM archive, skipping titles already seen."""
    output: list[dict] = []
    seen_paths: set[str] = set()

    # 1. Title/suggestion search — progressively drop leading keywords
    #    if no results (e.g. "hauteur piton neiges" → "piton neiges")
    try:
        from libzim.suggestion import SuggestionSearcher

        suggestion_searcher = SuggestionSearcher(archive)
        words = keywords.split()
        for start in range(len(words)):
            sub_keywords = " ".join(words[start:])
            if not sub_keywords:
                break
            suggestion = suggestion_searcher.suggest(sub_keywords)
            paths = list(suggestion.getResults(0, num_results))
            if paths:
                if start > 0:
                    logger.debug("Kiwix suggestion: %r failed, %r matched", keywords, sub_keywords)
                for path in paths:
                    if path in seen_paths:
                        continue
                    seen_paths.add(path)
                    result = _read_entry(archive, path, max_chars)
                    if result and result["title"] not in seen_titles:
                        seen_titles.add(result["title"])
                        output.append(result)
                        logger.debug("Kiwix suggestion hit: %s", result["title"])
                break  # Found results, stop trying
    except Exception as e:
        logger.debug("Kiwix suggestion search failed: %s", e)

    # 2. Full-text search
    remaining = num_results - len(output)
    if remaining > 0:
        try:
            from libzim.search import Query, Searcher

            searcher = Searcher(archive)
            zim_query = Query().set_query(keywords)
            results = searcher.search(zim_query)
            for path in results.getResults(0, remaining + 3):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                result = _read_entry(archive, path, max_chars)
                if result and result["title"] not in seen_titles:
                    seen_titles.add(result["title"])
                    output.append(result)
                    logger.debug("Kiwix fulltext hit: %s", result["title"])
                if len(output) >= remaining:
                    break
        except Exception as e:
            logger.debug("Kiwix full-text search failed: %s", e)

    return output


def get_article_content(title: str, max_chars: int = 0) -> str | None:
    """Read the full content of an article by exact title from ZIM archives.

    Args:
        title: Exact article title to look up.
        max_chars: If > 0, truncate the result to this many characters.

    Returns:
        Cleaned article text, or None if not found.
    """
    archives = _get_archives()
    if not archives:
        return None

    for _path, archive in archives:
        try:
            from libzim.suggestion import SuggestionSearcher

            searcher = SuggestionSearcher(archive)
            suggestion = searcher.suggest(title)
            for path in suggestion.getResults(0, 5):
                entry = archive.get_entry_by_path(path)
                if entry.title == title:
                    item = entry.get_item()
                    raw_html = bytes(item.content).decode("utf-8", errors="ignore")
                    text = _clean_html(raw_html)
                    if max_chars > 0 and len(text) > max_chars:
                        truncated = text[:max_chars]
                        last_space = truncated.rfind(" ")
                        if last_space > max_chars // 2:
                            truncated = truncated[:last_space]
                        text = truncated + "..."
                    return text
        except Exception as e:
            logger.debug("get_article_content failed for %r in %s: %s", title, _path, e)

    return None


def search(query: str, num_results: int = 3, max_chars_per_result: int = 500) -> list[dict]:
    """Search all ZIM files for articles matching the query.

    Always searches ALL ZIMs so that a fallback ZIM can contribute articles
    that the priority ZIM doesn't have. Deduplication by title ensures that
    when both ZIMs have the same article, the priority ZIM's version wins.
    The final result list is capped at num_results.

    Returns a list of {"title": str, "content": str} dicts.
    Returns an empty list if no ZIM is available or search fails.
    """
    archives = _get_archives()
    if not archives:
        return []

    keywords = _extract_keywords(query)
    logger.debug("Kiwix search: query=%r → keywords=%r", query, keywords)
    if not keywords:
        keywords = query  # Fallback to raw query if all words were filtered

    seen_titles: set[str] = set()
    output: list[dict] = []

    # Search ALL ZIMs — each contributes up to num_results, dedup by title
    for path, archive in archives:
        results = _search_single_archive(
            archive, keywords, num_results, max_chars_per_result, seen_titles,
        )
        output.extend(results)

    # Cap at num_results (priority ZIM results come first in the list)
    output = output[:num_results]

    if not output:
        logger.info("Kiwix search: no results for %r (keywords: %r)", query, keywords)
    else:
        logger.info("Kiwix search: %d results for %r", len(output), query)

    return output
