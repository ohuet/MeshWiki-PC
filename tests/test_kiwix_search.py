"""Tests for meshwiki.kiwix_search — Kiwix ZIM full-text search fallback."""

from unittest.mock import MagicMock, patch

import meshwiki.kiwix_search as ks


def setup_function():
    """Reset module state before each test."""
    ks.set_zim_paths([])
    ks._archives.clear()
    ks.set_disabled(False)


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


def test_disabled_hides_zim_path():
    """When disabled, get_zim_path returns None even if a path is set."""
    ks.set_zim_path("/tmp/test.zim")
    assert ks.get_zim_path() == "/tmp/test.zim"
    ks.set_disabled(True)
    assert ks.get_zim_path() is None
    ks.set_disabled(False)
    assert ks.get_zim_path() == "/tmp/test.zim"


def test_clean_html_strips_tags():
    assert ks._clean_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert ks._clean_html("<script>alert('x')</script><p>Text</p>") == "Text"
    assert ks._clean_html("<style>.x{}</style><div>Content</div>") == "Content"
    assert ks._clean_html("Plain text") == "Plain text"


def test_clean_html_removes_hatnote():
    """Hatnote/disambiguation banners are removed from cleaned text."""
    html = (
        '<div class="hatnote navigation-not-searchable">'
        'Pour le groupe rock, voir <a href="/Dengue_Fever">Dengue Fever</a>.'
        '</div>'
        '<p>La dengue est une maladie tropicale.</p>'
    )
    result = ks._clean_html(html)
    assert "groupe rock" not in result
    assert "Dengue Fever" not in result
    assert "dengue est une maladie tropicale" in result


def test_clean_html_removes_dablink():
    """Dablink disambiguation banners are removed."""
    html = (
        '<div class="dablink">'
        'Pour les articles homonymes, voir <a href="/Test">Test (homonymie)</a>.'
        '</div>'
        '<p>Contenu principal.</p>'
    )
    result = ks._clean_html(html)
    assert "homonymes" not in result
    assert "Contenu principal" in result


def test_clean_html_removes_homonymie():
    """Homonymie banners are removed."""
    html = (
        '<div class="homonymie">'
        'Cette page est une page de désambiguïsation.'
        '</div>'
        '<p>Article réel.</p>'
    )
    result = ks._clean_html(html)
    assert "désambiguïsation" not in result
    assert "Article réel" in result


def test_clean_html_removes_bandeau_container():
    """Warning banners (Mise en garde médicale, etc.) are removed."""
    html = (
        '<div class="bandeau-container">'
        '<div class="bandeau-cell">Mise en garde médicale</div>'
        '</div>'
        '<p>La dengue est une maladie.</p>'
    )
    result = ks._clean_html(html)
    assert "Mise en garde" not in result
    assert "dengue est une maladie" in result


def test_clean_html_removes_navbox():
    """Navigation boxes at the bottom of articles are removed."""
    html = (
        '<p>Contenu principal.</p>'
        '<div class="navbox">'
        '<table><tr><td>Liens internes très longs...</td></tr></table>'
        '</div>'
    )
    result = ks._clean_html(html)
    assert "Liens internes" not in result
    assert "Contenu principal" in result


def test_clean_html_removes_classification_section():
    """'Classification et ressources externes' block is removed from infoboxes."""
    html = (
        '<table class="infobox">'
        '<tr><th>Causes</th><td>Virus</td></tr>'
        '<tr><th colspan="2">Classification et ressources externes</th></tr>'
        '<tr><td>CIM-10</td><td>A90</td></tr>'
        '<tr><td>OMIM</td><td>614371</td></tr>'
        '</tbody></table>'
        '<p>Article principal.</p>'
    )
    result = ks._clean_html(html)
    assert "Causes" in result
    assert "Virus" in result
    assert "CIM-10" not in result
    assert "OMIM" not in result
    assert "614371" not in result
    assert "Article principal" in result


def test_clean_html_removes_edit_links():
    """Edit section links (modifier - modifier le code) are removed."""
    html = (
        '<h2>Histoire'
        '<span class="mw-editsection">'
        '<a href="/edit">modifier</a> | <a href="/edit">modifier le code</a>'
        '</span>'
        '</h2>'
        '<p>Texte historique.</p>'
    )
    result = ks._clean_html(html)
    assert "modifier" not in result.lower()
    assert "Histoire" in result
    assert "Texte historique" in result


def test_clean_html_removes_wikidata_markers():
    """Wikidata/interlanguage markers like ( d ) and ( en ) are removed."""
    html = "<p>hémostatique ( en ) , transfusion ( d ) et diurétique</p>"
    result = ks._clean_html(html)
    assert "( en )" not in result
    assert "( d )" not in result
    assert "hémostatique" in result
    assert "transfusion" in result
    assert "diurétique" in result


def test_clean_html_removes_wikidata_edit_text():
    """Leftover 'modifier - modifier le code - voir Wikidata' text is removed."""
    # With spaces: ( aide )
    html = "<p>Titre modifier - modifier le code - voir Wikidata ( aide ) Suite du texte.</p>"
    result = ks._clean_html(html)
    assert "modifier" not in result
    assert "Wikidata" not in result
    assert "Titre" in result
    assert "Suite du texte" in result
    # Without spaces: (aide)
    html2 = "<p>Titre modifier - modifier le code - voir Wikidata (aide) Suite.</p>"
    result2 = ks._clean_html(html2)
    assert "modifier" not in result2
    assert "Wikidata" not in result2


def test_clean_html_removes_mise_en_garde_text():
    """'Mise en garde médicale' text is removed even if HTML filter missed it."""
    html = "<p>Mise en garde médicale La dengue est une maladie.</p>"
    result = ks._clean_html(html)
    assert "Mise en garde" not in result
    assert "dengue est une maladie" in result


def test_clean_html_removes_hatnote_text_fallback():
    """Hatnote text at the start is removed even without HTML class."""
    # Pattern: "Pour le/la/les/l' ... voir ..."
    html = "<p>Pour le groupe rock qui porte ce nom, voir Dengue Fever. La dengue est une maladie.</p>"
    result = ks._clean_html(html)
    assert "groupe rock" not in result
    assert "Dengue Fever" not in result
    assert "dengue est une maladie" in result

    # "Pour l'article..." variant
    html2 = "<p>Pour l'article principal, voir France. La France est un pays.</p>"
    result2 = ks._clean_html(html2)
    assert "article principal" not in result2
    assert "France est un pays" in result2

    # Does NOT remove "Pour" in the middle of text
    html3 = "<p>Ceci est un texte. Pour le moment, rien à voir ici.</p>"
    result3 = ks._clean_html(html3)
    assert "Pour le moment" in result3


def test_clean_html_decodes_entities():
    """HTML entities like &nbsp; and &amp; are decoded to plain text."""
    assert ks._clean_html("<p>mot&nbsp;:&nbsp;test</p>") == "mot : test"
    assert ks._clean_html("A &amp; B") == "A & B"
    assert ks._clean_html("1 &lt; 2 &gt; 0") == "1 < 2 > 0"


def test_clean_html_removes_reference_markers():
    """Wikipedia reference markers [1], [ 2 ], etc. are removed."""
    html = "<p>La dengue est une maladie<sup>[1]</sup> tropicale<sup>[2]</sup>.</p>"
    assert ks._clean_html(html) == "La dengue est une maladie tropicale ."
    # Also works with spaced markers after tag stripping
    assert "[ 3 ]" not in ks._clean_html("<p>Texte [ 3 ] suite</p>")


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


@patch("meshwiki.kiwix_search._get_archives")
def test_search_returns_results(mock_get_archives):
    """Search with mocked Archive returns expected results."""
    mock_entry = _make_mock_entry("Paris", "<p>Paris est la capitale de la France.</p>")
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archives.return_value = [("/tmp/test.zim", mock_archive)]

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


@patch("meshwiki.kiwix_search._get_archives")
def test_search_combines_suggestion_and_fulltext(mock_get_archives):
    """Search combines results from suggestion and full-text, deduplicating."""
    entries = {
        "A/Piton": _make_mock_entry("Piton des Neiges", "<p>Le piton des Neiges culmine à 3070 m.</p>"),
        "A/Reunion": _make_mock_entry("La Réunion", "<p>La Réunion est une île de l'océan Indien.</p>"),
    }
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.side_effect = lambda p: entries[p]
    mock_get_archives.return_value = [("/tmp/test.zim", mock_archive)]

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


@patch("meshwiki.kiwix_search._get_archives")
def test_search_truncates_long_content(mock_get_archives):
    """Content longer than max_chars_per_result is truncated."""
    long_text = "mot " * 200  # ~800 chars
    mock_entry = _make_mock_entry("Long Article", f"<p>{long_text}</p>")
    mock_archive = MagicMock()
    mock_archive.get_entry_by_path.return_value = mock_entry
    mock_get_archives.return_value = [("/tmp/test.zim", mock_archive)]

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
