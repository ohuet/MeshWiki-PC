"""Tests for meshwiki.kiwix_search — Kiwix ZIM full-text search fallback."""

from unittest.mock import MagicMock, patch

import meshwiki.kiwix_search as ks


def setup_function():
    """Reset module state before each test."""
    ks.set_zim_path(None)
    ks._archive = None
    ks._archive_path = None


def test_set_and_get_zim_path():
    assert ks.get_zim_path() is None
    ks.set_zim_path("/tmp/test.zim")
    assert ks.get_zim_path() == "/tmp/test.zim"
    ks.set_zim_path(None)
    assert ks.get_zim_path() is None


def test_search_no_zim_returns_empty():
    """Without a ZIM path set, search returns an empty list."""
    result = ks.search("test query")
    assert result == []


def test_clean_html_strips_tags():
    assert ks._clean_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert ks._clean_html("<script>alert('x')</script><p>Text</p>") == "Text"
    assert ks._clean_html("<style>.x{}</style><div>Content</div>") == "Content"
    assert ks._clean_html("Plain text") == "Plain text"


def test_clean_html_decodes_entities():
    """HTML entities like &nbsp; and &amp; are decoded to plain text."""
    assert ks._clean_html("<p>mot&nbsp;:&nbsp;test</p>") == "mot : test"
    assert ks._clean_html("A &amp; B") == "A & B"
    assert ks._clean_html("1 &lt; 2 &gt; 0") == "1 < 2 > 0"


def test_extract_keywords_removes_stop_words():
    """_extract_keywords strips French stop words and punctuation."""
    assert ks._extract_keywords("Quelle est la hauteur du piton des neiges ?") == "hauteur piton neiges"
    assert ks._extract_keywords("Dangerosité du requin tigre") == "dangerosité requin tigre"
    assert ks._extract_keywords("Qui est le président de la France ?") == "président france"


def test_extract_keywords_preserves_compound_words():
    """Hyphens in compound words are preserved."""
    result = ks._extract_keywords("Saint-Denis de la Réunion")
    assert "saint-denis" in result


def test_extract_keywords_empty_result_handled():
    """If all words are stop words, the raw query is used as fallback in search()."""
    # All stop words → empty string
    assert ks._extract_keywords("est ce que") == ""


def _make_mock_entry(title: str, html: str) -> MagicMock:
    """Create a mock ZIM entry with given title and HTML content."""
    mock_item = MagicMock()
    mock_item.content = html.encode("utf-8")
    mock_entry = MagicMock()
    mock_entry.title = title
    mock_entry.get_item.return_value = mock_item
    return mock_entry


@patch("meshwiki.kiwix_search._get_archive")
def test_search_returns_results(mock_get_archive):
    """Search with mocked Archive returns expected results."""
    ks.set_zim_path("/tmp/test.zim")

    mock_entry = _make_mock_entry("Paris", "<p>Paris est la capitale de la France.</p>")
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archive.return_value = mock_archive

    # Mock suggestion search returning the result
    mock_suggestion = MagicMock()
    mock_suggestion.getResults.return_value = ["A/Paris"]
    mock_suggestion_searcher = MagicMock()
    mock_suggestion_searcher.suggest.return_value = mock_suggestion

    # Mock full-text search (no additional results needed)
    mock_search_result = MagicMock()
    mock_search_result.getResults.return_value = []
    mock_ft_searcher = MagicMock()
    mock_ft_searcher.search.return_value = mock_search_result
    mock_query = MagicMock()
    mock_query.set_query.return_value = mock_query

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_suggestion_searcher),
        ),
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=mock_query),
            Searcher=MagicMock(return_value=mock_ft_searcher),
        ),
    }):
        results = ks.search("Paris")

    assert len(results) == 1
    assert results[0]["title"] == "Paris"
    assert "Paris est la capitale de la France" in results[0]["content"]


@patch("meshwiki.kiwix_search._get_archive")
def test_search_combines_suggestion_and_fulltext(mock_get_archive):
    """Search combines results from suggestion and full-text, deduplicating."""
    ks.set_zim_path("/tmp/test.zim")

    entries = {
        "A/Piton": _make_mock_entry("Piton des Neiges", "<p>Le piton des Neiges culmine à 3070 m.</p>"),
        "A/Reunion": _make_mock_entry("La Réunion", "<p>La Réunion est une île de l'océan Indien.</p>"),
    }
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.side_effect = lambda p: entries[p]
    mock_get_archive.return_value = mock_archive

    # Suggestion returns "Piton"
    mock_suggestion = MagicMock()
    mock_suggestion.getResults.return_value = ["A/Piton"]
    mock_suggestion_searcher = MagicMock()
    mock_suggestion_searcher.suggest.return_value = mock_suggestion

    # Full-text returns "Piton" (dup) + "Reunion" (new)
    mock_search_result = MagicMock()
    mock_search_result.getResults.return_value = ["A/Piton", "A/Reunion"]
    mock_ft_searcher = MagicMock()
    mock_ft_searcher.search.return_value = mock_search_result
    mock_query = MagicMock()
    mock_query.set_query.return_value = mock_query

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_suggestion_searcher),
        ),
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=mock_query),
            Searcher=MagicMock(return_value=mock_ft_searcher),
        ),
    }):
        results = ks.search("piton neiges", num_results=3)

    assert len(results) == 2
    assert results[0]["title"] == "Piton des Neiges"
    assert results[1]["title"] == "La Réunion"


@patch("meshwiki.kiwix_search._get_archive")
def test_search_truncates_long_content(mock_get_archive):
    """Content longer than max_chars_per_result is truncated."""
    ks.set_zim_path("/tmp/test.zim")

    long_text = "mot " * 200  # ~800 chars
    mock_entry = _make_mock_entry("Long Article", f"<p>{long_text}</p>")
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archive.return_value = mock_archive

    mock_suggestion = MagicMock()
    mock_suggestion.getResults.return_value = ["A/Long"]
    mock_suggestion_searcher = MagicMock()
    mock_suggestion_searcher.suggest.return_value = mock_suggestion

    with patch.dict("sys.modules", {
        "libzim.suggestion": MagicMock(
            SuggestionSearcher=MagicMock(return_value=mock_suggestion_searcher),
        ),
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=MagicMock(set_query=MagicMock(return_value=MagicMock()))),
            Searcher=MagicMock(return_value=MagicMock(search=MagicMock(return_value=MagicMock(getResults=MagicMock(return_value=[]))))),
        ),
    }):
        results = ks.search("test", max_chars_per_result=100)

    assert len(results) == 1
    assert len(results[0]["content"]) <= 104  # 100 + "..."
    assert results[0]["content"].endswith("...")
