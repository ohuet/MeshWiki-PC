"""Tests for multi-ZIM search with priority and deduplication."""

from unittest.mock import MagicMock, patch

import meshwiki.kiwix_search as ks


def setup_function():
    """Reset module state before each test."""
    ks.set_zim_paths([])
    ks._archives.clear()
    ks.set_disabled(False)


def _make_mock_entry(title: str, html: str) -> MagicMock:
    """Create a mock ZIM entry with given title and HTML content."""
    mock_item = MagicMock()
    mock_item.content = html.encode("utf-8")
    mock_entry = MagicMock()
    mock_entry.title = title
    mock_entry.get_item.return_value = mock_item
    return mock_entry


def test_set_zim_paths_and_has_zim_paths():
    """set_zim_paths / has_zim_paths work correctly."""
    assert ks.has_zim_paths() is False
    ks.set_zim_paths(["/a.zim", "/b.zim"])
    assert ks.has_zim_paths() is True
    ks.set_zim_paths([])
    assert ks.has_zim_paths() is False


def test_has_zim_paths_respects_disabled():
    """has_zim_paths returns False when disabled."""
    ks.set_zim_paths(["/a.zim"])
    assert ks.has_zim_paths() is True
    ks.set_disabled(True)
    assert ks.has_zim_paths() is False
    ks.set_disabled(False)
    assert ks.has_zim_paths() is True


def test_legacy_set_zim_path():
    """Legacy set_zim_path maps to set_zim_paths."""
    ks.set_zim_path("/tmp/test.zim")
    assert ks.has_zim_paths() is True
    assert ks.get_zim_path() == "/tmp/test.zim"
    ks.set_zim_path(None)
    assert ks.has_zim_paths() is False


def test_set_zim_paths_cleans_removed_archives():
    """Archives no longer in the path list are removed from cache."""
    ks._archives["/old.zim"] = MagicMock()
    ks._archives["/keep.zim"] = MagicMock()
    ks.set_zim_paths(["/keep.zim"])
    assert "/old.zim" not in ks._archives
    assert "/keep.zim" in ks._archives


def _make_mock_archive(entries: dict[str, MagicMock]) -> MagicMock:
    """Create a mock archive that returns entries by path."""
    archive = MagicMock()
    archive.get_entry_by_path.side_effect = lambda p: entries[p]
    return archive


@patch("meshwiki.kiwix_search._get_archives")
def test_multi_zim_priority_dedup(mock_get_archives):
    """Results from higher-priority ZIM take precedence; duplicates are skipped."""
    # Priority ZIM (small, custom): has "Paris" with detailed content
    priority_entries = {
        "A/Paris": _make_mock_entry("Paris", "<p>Paris, article complet et détaillé.</p>"),
    }
    priority_archive = _make_mock_archive(priority_entries)

    # Fallback ZIM (large, Wikipedia): also has "Paris" (should be skipped)
    fallback_entries = {
        "A/Paris": _make_mock_entry("Paris", "<p>Paris, version réduite.</p>"),
        "A/Lyon": _make_mock_entry("Lyon", "<p>Lyon est une ville de France.</p>"),
    }
    fallback_archive = _make_mock_archive(fallback_entries)

    mock_get_archives.return_value = [
        ("/small/custom.zim", priority_archive),
        ("/large/wikipedia.zim", fallback_archive),
    ]

    # Mock search for priority archive: finds "Paris"
    mock_sugg1 = MagicMock()
    mock_sugg1.getResults.return_value = ["A/Paris"]
    mock_sugg_searcher1 = MagicMock()
    mock_sugg_searcher1.suggest.return_value = mock_sugg1

    # Mock search for fallback archive: finds "Paris" + "Lyon"
    mock_sugg2 = MagicMock()
    mock_sugg2.getResults.return_value = ["A/Paris", "A/Lyon"]
    mock_sugg_searcher2 = MagicMock()
    mock_sugg_searcher2.suggest.return_value = mock_sugg2

    # SuggestionSearcher returns different searchers for each archive
    call_count = [0]
    def suggestion_searcher_factory(archive):
        result = mock_sugg_searcher1 if call_count[0] == 0 else mock_sugg_searcher2
        call_count[0] += 1
        return result

    # No fulltext results needed
    mock_ft = MagicMock()
    mock_ft.search.return_value = MagicMock(getResults=MagicMock(return_value=[]))
    mock_query = MagicMock()
    mock_query.set_query.return_value = mock_query

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(side_effect=suggestion_searcher_factory),
        ),
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=mock_query),
            Searcher=MagicMock(return_value=mock_ft),
        ),
    }):
        results = ks.search("Paris", num_results=3)

    # Paris from priority ZIM + Lyon from fallback (Paris deduped)
    assert len(results) == 2
    assert results[0]["title"] == "Paris"
    assert "complet" in results[0]["content"]  # From priority ZIM
    assert results[1]["title"] == "Lyon"


@patch("meshwiki.kiwix_search._get_archives")
def test_multi_zim_caps_at_num_results(mock_get_archives):
    """Results from all ZIMs are combined but capped at num_results."""
    entries1 = {
        "A/A": _make_mock_entry("Article A", "<p>Content A</p>"),
        "A/B": _make_mock_entry("Article B", "<p>Content B</p>"),
    }
    archive1 = _make_mock_archive(entries1)

    entries2 = {
        "A/C": _make_mock_entry("Article C", "<p>Content C</p>"),
    }
    archive2 = _make_mock_archive(entries2)

    mock_get_archives.return_value = [
        ("/small.zim", archive1),
        ("/large.zim", archive2),
    ]

    call_count = [0]
    def make_sugg_searcher(archive):
        sugg = MagicMock()
        if call_count[0] == 0:
            sugg.suggest.return_value = MagicMock(getResults=MagicMock(return_value=["A/A", "A/B"]))
        else:
            sugg.suggest.return_value = MagicMock(getResults=MagicMock(return_value=["A/C"]))
        call_count[0] += 1
        return sugg

    mock_ft = MagicMock()
    mock_ft.search.return_value = MagicMock(getResults=MagicMock(return_value=[]))
    mock_query = MagicMock()
    mock_query.set_query.return_value = mock_query

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(side_effect=make_sugg_searcher),
        ),
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=mock_query),
            Searcher=MagicMock(return_value=mock_ft),
        ),
    }):
        # 3 total results available, but cap at 2
        results = ks.search("test", num_results=2)

    assert len(results) == 2
    # Priority ZIM results come first
    assert results[0]["title"] == "Article A"
    assert results[1]["title"] == "Article B"


@patch("meshwiki.kiwix_search._get_archives")
def test_search_no_archives_returns_empty(mock_get_archives):
    """When no archives are available, search returns empty list."""
    mock_get_archives.return_value = []
    assert ks.search("test") == []


@patch("meshwiki.kiwix_search._get_archives")
def test_get_article_content_exact_match(mock_get_archives):
    """get_article_content returns cleaned text for an exact title match."""
    entry = _make_mock_entry(
        "Piton des Neiges",
        "<p>Le Piton des Neiges culmine à 3 070 mètres.</p>",
    )
    archive = MagicMock()
    archive.get_entry_by_path.return_value = entry

    mock_get_archives.return_value = [("/test.zim", archive)]

    mock_sugg = MagicMock()
    mock_sugg.getResults.return_value = ["A/Piton_des_Neiges"]
    mock_searcher = MagicMock()
    mock_searcher.suggest.return_value = mock_sugg

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_searcher),
        ),
    }):
        result = ks.get_article_content("Piton des Neiges")

    assert result is not None
    assert "3 070" in result
    assert "<p>" not in result  # HTML cleaned


@patch("meshwiki.kiwix_search._get_archives")
def test_get_article_content_not_found(mock_get_archives):
    """get_article_content returns None for a title that doesn't exist."""
    # Archive where suggestion returns no matching titles
    entry = _make_mock_entry("Autre Article", "<p>Contenu</p>")
    archive = MagicMock()
    archive.get_entry_by_path.return_value = entry

    mock_get_archives.return_value = [("/test.zim", archive)]

    mock_sugg = MagicMock()
    mock_sugg.getResults.return_value = ["A/Autre"]
    mock_searcher = MagicMock()
    mock_searcher.suggest.return_value = mock_sugg

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_searcher),
        ),
    }):
        result = ks.get_article_content("Piton des Neiges")

    assert result is None


@patch("meshwiki.kiwix_search._get_archives")
def test_get_article_content_no_archives(mock_get_archives):
    """get_article_content returns None when no archives are available."""
    mock_get_archives.return_value = []
    assert ks.get_article_content("Test") is None


@patch("meshwiki.kiwix_search._get_archives")
def test_get_article_content_with_max_chars(mock_get_archives):
    """get_article_content truncates content when max_chars is set."""
    long_text = "Le Piton des Neiges est un volcan. " * 50
    entry = _make_mock_entry("Piton des Neiges", f"<p>{long_text}</p>")
    archive = MagicMock()
    archive.get_entry_by_path.return_value = entry

    mock_get_archives.return_value = [("/test.zim", archive)]

    mock_sugg = MagicMock()
    mock_sugg.getResults.return_value = ["A/Piton_des_Neiges"]
    mock_searcher = MagicMock()
    mock_searcher.suggest.return_value = mock_sugg

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_searcher),
        ),
    }):
        result = ks.get_article_content("Piton des Neiges", max_chars=100)

    assert result is not None
    assert result.endswith("...")
    assert len(result) <= 104  # 100 + "..."
