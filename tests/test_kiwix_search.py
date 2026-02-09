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


@patch("meshwiki.kiwix_search._get_archive")
def test_search_returns_results(mock_get_archive):
    """Search with mocked Archive + Searcher returns expected results."""
    ks.set_zim_path("/tmp/test.zim")

    # Mock entry
    mock_item = MagicMock()
    mock_item.content = b"<html><body><p>Paris est la capitale de la France.</p></body></html>"

    mock_entry = MagicMock()
    mock_entry.title = "Paris"
    mock_entry.get_item.return_value = mock_item

    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archive.return_value = mock_archive

    mock_search_result = MagicMock()
    mock_search_result.getResults.return_value = ["A/Paris"]

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = mock_search_result

    mock_query = MagicMock()
    mock_query.set_query.return_value = mock_query

    mock_libzim_search = MagicMock(
        Query=MagicMock(return_value=mock_query),
        Searcher=MagicMock(return_value=mock_searcher),
    )

    with patch.dict("sys.modules", {"libzim.search": mock_libzim_search}):
        results = ks.search("Paris")

    assert len(results) == 1
    assert results[0]["title"] == "Paris"
    assert "Paris est la capitale de la France" in results[0]["content"]


@patch("meshwiki.kiwix_search._get_archive")
def test_search_truncates_long_content(mock_get_archive):
    """Content longer than max_chars_per_result is truncated."""
    ks.set_zim_path("/tmp/test.zim")

    long_text = "mot " * 200  # ~800 chars
    mock_item = MagicMock()
    mock_item.content = f"<p>{long_text}</p>".encode()

    mock_entry = MagicMock()
    mock_entry.title = "Long Article"
    mock_entry.get_item.return_value = mock_item

    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archive.return_value = mock_archive

    mock_search = MagicMock()
    mock_search.getResults.return_value = ["A/Long"]

    mock_searcher = MagicMock()
    mock_searcher.search.return_value = mock_search

    with patch.dict("sys.modules", {
        "libzim.search": MagicMock(
            Query=MagicMock(return_value=MagicMock(set_query=MagicMock(return_value=MagicMock()))),
            Searcher=MagicMock(return_value=mock_searcher),
        ),
    }):
        results = ks.search("test", max_chars_per_result=100)

    assert len(results) == 1
    assert len(results[0]["content"]) <= 104  # 100 + "..."
    assert results[0]["content"].endswith("...")


def test_clean_html_strips_tags():
    assert ks._clean_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert ks._clean_html("<script>alert('x')</script><p>Text</p>") == "Text"
    assert ks._clean_html("<style>.x{}</style><div>Content</div>") == "Content"
    assert ks._clean_html("Plain text") == "Plain text"
