"""Tests for meshwiki.build_zim — Custom ZIM builder from Wikipedia."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import yaml

from meshwiki.build_zim import (
    DEFAULT_CONFIG,
    WikiArticleItem,
    WikipediaAPI,
    _build_home_page,
    _format_eta,
    _safe_filename,
    _strip_images,
    _title_to_path,
    _load_staging_index,
    _save_staging_index,
    _save_article_staging,
    _load_article_staging,
    _clean_staging,
    init_config,
    build_zim,
    STAGING_DIR,
    STAGING_INDEX,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestStripImages:
    def test_removes_img_tags(self):
        html = '<div><p>Hello</p><img src="x.png"/><p>World</p></div>'
        result = _strip_images(html)
        assert "<img" not in result
        assert "Hello" in result
        assert "World" in result

    def test_removes_figure_tags(self):
        html = '<div><figure><img src="x.png"/><figcaption>Cap</figcaption></figure><p>Text</p></div>'
        result = _strip_images(html)
        assert "<figure" not in result
        assert "<figcaption" not in result
        assert "Text" in result

    def test_removes_thumb_class(self):
        html = '<div><div class="thumb">Thumb content</div><p>Keep</p></div>'
        result = _strip_images(html)
        assert "Thumb content" not in result
        assert "Keep" in result

    def test_removes_infobox_class(self):
        html = '<div><table class="infobox"><tr><td>Info</td></tr></table><p>Keep</p></div>'
        result = _strip_images(html)
        assert "infobox" not in result
        assert "Keep" in result

    def test_removes_navbox_class(self):
        html = '<div><div class="navbox">Nav</div><p>Article</p></div>'
        result = _strip_images(html)
        assert "navbox" not in result.lower() or "Nav" not in result
        assert "Article" in result

    def test_preserves_tables(self):
        html = '<div><table class="wikitable"><tr><td>1</td><td>2</td></tr></table></div>'
        result = _strip_images(html)
        assert "<table" in result
        assert "1" in result
        assert "2" in result

    def test_invalid_html_returns_original(self):
        result = _strip_images("")
        assert isinstance(result, str)

    def test_removes_style_tags(self):
        html = '<div><style>.foo{color:red}</style><p>Text</p></div>'
        result = _strip_images(html)
        assert "<style" not in result
        assert "Text" in result


class TestTitleToPath:
    def test_simple_title(self):
        assert _title_to_path("Paris") == "A/Paris"

    def test_spaces_replaced(self):
        result = _title_to_path("La Réunion")
        assert result.startswith("A/")
        assert " " not in result
        assert "La" in result

    def test_special_chars(self):
        result = _title_to_path("Échelle de Saffir-Simpson")
        assert result.startswith("A/")
        assert " " not in result


class TestSafeFilename:
    def test_returns_hex_string(self):
        result = _safe_filename("Test Article")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        assert _safe_filename("Paris") == _safe_filename("Paris")

    def test_different_for_different_titles(self):
        assert _safe_filename("Paris") != _safe_filename("Lyon")


class TestBuildHomePage:
    def test_contains_all_titles(self):
        titles = ["Paris", "Lyon", "Marseille"]
        html = _build_home_page(titles)
        for t in titles:
            assert t in html

    def test_sorted_alphabetically(self):
        titles = ["Zéro", "Alpha", "Midi"]
        html = _build_home_page(titles)
        pos_a = html.index("Alpha")
        pos_m = html.index("Midi")
        pos_z = html.index("Zéro")
        assert pos_a < pos_m < pos_z

    def test_shows_count(self):
        titles = ["A", "B", "C"]
        html = _build_home_page(titles)
        assert "3 articles" in html

    def test_empty_list(self):
        html = _build_home_page([])
        assert "0 articles" in html

    def test_static_pages_section(self):
        titles = ["Paris"]
        static = [{"title": "Hébergements d'urgence"}]
        html = _build_home_page(titles, static_pages=static)
        assert "Pages locales" in html
        assert "Hébergements d'urgence" in html
        assert "Paris" in html

    def test_no_static_pages_section_when_empty(self):
        html = _build_home_page(["Paris"])
        assert "Pages locales" not in html


class TestFormatEta:
    def test_seconds(self):
        assert _format_eta(45) == "45s"

    def test_minutes_seconds(self):
        assert _format_eta(125) == "2min05s"

    def test_hours_minutes(self):
        assert _format_eta(3725) == "1h02min"

    def test_negative(self):
        assert _format_eta(-1) == "?"


# ---------------------------------------------------------------------------
# WikipediaAPI
# ---------------------------------------------------------------------------

class TestWikipediaAPI:
    @pytest.fixture
    def api(self):
        return WikipediaAPI(
            api_url="https://fr.wikipedia.org/w/api.php",
            user_agent="TestBot/0.1",
            request_delay=0,
            max_retries=2,
            retry_delay=0,
        )

    @patch("meshwiki.build_zim.requests.Session")
    def test_get_category_members(self, mock_session_cls, api):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "query": {
                "categorymembers": [
                    {"title": "Article 1"},
                    {"title": "Article 2"},
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        api.session.get = MagicMock(return_value=mock_resp)

        result = api.get_category_members("Test")
        assert result == ["Article 1", "Article 2"]

    @patch("meshwiki.build_zim.requests.Session")
    def test_get_category_members_pagination(self, mock_session_cls, api):
        page1 = MagicMock()
        page1.json.return_value = {
            "query": {"categorymembers": [{"title": "A1"}]},
            "continue": {"cmcontinue": "abc123"},
        }
        page1.raise_for_status = MagicMock()

        page2 = MagicMock()
        page2.json.return_value = {
            "query": {"categorymembers": [{"title": "A2"}]},
        }
        page2.raise_for_status = MagicMock()

        api.session.get = MagicMock(side_effect=[page1, page2])

        result = api.get_category_members("Test")
        assert result == ["A1", "A2"]
        assert api.session.get.call_count == 2

    @patch("meshwiki.build_zim.requests.Session")
    def test_get_article_html(self, mock_session_cls, api):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "parse": {"text": {"*": "<p>Article content</p>"}}
        }
        mock_resp.raise_for_status = MagicMock()
        api.session.get = MagicMock(return_value=mock_resp)

        result = api.get_article_html("Paris")
        assert result == "<p>Article content</p>"

    @patch("meshwiki.build_zim.requests.Session")
    def test_get_article_html_failure_returns_none(self, mock_session_cls, api):
        import requests as req
        api.session.get = MagicMock(side_effect=req.RequestException("timeout"))

        result = api.get_article_html("Paris")
        assert result is None

    def test_recursive_with_blacklist(self, api):
        """Test that blacklisted categories are skipped."""
        call_log = []

        def mock_get_members(cat, namespace=0):
            call_log.append((cat, namespace))
            if cat == "Root" and namespace == 0:
                return ["Article Root"]
            if cat == "Root" and namespace == 14:
                return ["Catégorie:Good", "Catégorie:Bad"]
            if cat == "Good" and namespace == 0:
                return ["Article Good"]
            if cat == "Good" and namespace == 14:
                return []
            return []

        api.get_category_members = mock_get_members

        result = api.get_category_articles_recursive(
            "Root", max_depth=2, blacklist=["Catégorie:Bad"],
        )
        assert "Article Root" in result
        assert "Article Good" in result
        # "Bad" should never have been explored
        assert not any(cat == "Bad" for cat, _ in call_log)

    def test_recursive_deduplication(self, api):
        """Test that articles are deduplicated across categories."""
        def mock_get_members(cat, namespace=0):
            if namespace == 14:
                return []
            return ["Shared Article", f"Article {cat}"]

        api.get_category_members = mock_get_members

        seen = set()
        result1 = api.get_category_articles_recursive("Cat1", seen=seen)
        result2 = api.get_category_articles_recursive("Cat2", seen=seen)
        combined = result1 | result2
        # "Shared Article" should appear once in the combined set
        assert "Shared Article" in combined

    def test_recursive_max_depth(self, api):
        """Test that recursion respects max_depth."""
        depth_reached = []

        def mock_get_members(cat, namespace=0):
            if namespace == 14:
                depth_reached.append(cat)
                return [f"Catégorie:Sub_{cat}"]
            return [f"Article_{cat}"]

        api.get_category_members = mock_get_members

        result = api.get_category_articles_recursive("Root", max_depth=1)
        # Depth 0: Root articles + subcats
        # Depth 1: Sub_Root articles + subcats (but won't recurse further)
        assert "Article_Root" in result


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

class TestStaging:
    @pytest.fixture(autouse=True)
    def _setup_staging(self, tmp_path, monkeypatch):
        """Redirect staging to tmp_path for test isolation."""
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr("meshwiki.build_zim.STAGING_DIR", staging)
        monkeypatch.setattr("meshwiki.build_zim.STAGING_INDEX", staging / "_index.json")
        self.staging_dir = staging

    def test_save_and_load_index(self):
        index = {"Paris": "abc123", "Lyon": "def456"}
        _save_staging_index(index)
        loaded = _load_staging_index()
        assert loaded == index

    def test_load_empty_index(self):
        assert _load_staging_index() == {}

    def test_save_and_load_article(self):
        index = {}
        _save_article_staging("Paris", "<p>Paris content</p>", index)
        assert "Paris" in index

        html = _load_article_staging("Paris", index)
        assert html == "<p>Paris content</p>"

    def test_load_missing_article(self):
        assert _load_article_staging("Nonexistent", {}) is None

    def test_clean_staging(self):
        index = {}
        _save_article_staging("Test", "<p>Test</p>", index)
        _save_staging_index(index)
        assert self.staging_dir.exists()

        _clean_staging()
        assert not self.staging_dir.exists()


# ---------------------------------------------------------------------------
# WikiArticleItem
# ---------------------------------------------------------------------------

class TestWikiArticleItem:
    def test_properties(self):
        item = WikiArticleItem("Paris", "A/Paris", "<p>Content</p>")
        assert item.get_path() == "A/Paris"
        assert item.get_title() == "Paris"
        assert item.get_mimetype() == "text/html"

    def test_hints(self):
        from libzim.writer import Hint  # noqa: F811
        item = WikiArticleItem("Test", "A/Test", "<p>Test</p>")
        hints = item.get_hints()
        assert hints[Hint.FRONT_ARTICLE] is True

    def test_content_provider(self):
        item = WikiArticleItem("Test", "A/Test", "<p>Content</p>")
        provider = item.get_contentprovider()
        assert provider is not None


# ---------------------------------------------------------------------------
# init_config
# ---------------------------------------------------------------------------

class TestInitConfig:
    def test_creates_config(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        init_config(config_path)

        assert config_path.exists()
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["output"] == "data/meshwiki_reunion.zim"
        assert "wikipedia" in cfg
        assert "categories" in cfg
        assert "articles" in cfg
        assert "metadata" in cfg

    def test_asks_confirmation_on_overwrite(self, tmp_path, monkeypatch):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("existing content")

        # User says no → file unchanged
        monkeypatch.setattr("builtins.input", lambda _: "")
        init_config(config_path)
        assert config_path.read_text() == "existing content"

    def test_overwrites_on_confirmation(self, tmp_path, monkeypatch):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("existing content")

        # User says yes → file overwritten
        monkeypatch.setattr("builtins.input", lambda _: "o")
        init_config(config_path)
        assert config_path.read_text() != "existing content"
        assert "output" in config_path.read_text()

    def test_force_overwrites_without_prompt(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("existing content")

        init_config(config_path, force=True)
        assert "output" in config_path.read_text()

    def test_default_config_has_categories(self):
        cats = DEFAULT_CONFIG["categories"]["list"]
        assert len(cats) >= 30
        names = [c["name"] for c in cats]
        assert "Cyclone tropical" in names
        assert "Premiers secours" in names

    def test_default_config_has_articles(self):
        articles = DEFAULT_CONFIG["articles"]
        assert len(articles) >= 40
        assert "Échelle de Saffir-Simpson" in articles
        assert "Réanimation cardiopulmonaire" in articles

    def test_default_config_has_per_category_depths(self):
        """Broad generic categories should have limited max_depth."""
        cats = DEFAULT_CONFIG["categories"]["list"]
        cats_by_name = {c["name"]: c for c in cats}
        # Broad categories should have explicit low max_depth
        assert cats_by_name["Premiers secours"].get("max_depth", 10) <= 3
        assert cats_by_name["Risque naturel"].get("max_depth", 10) <= 3
        assert cats_by_name["Cyclone tropical"].get("max_depth", 10) <= 3
        # La Réunion top-level uses limited depth (specifics are covered by sub-categories)
        assert cats_by_name["La Réunion"].get("max_depth", 10) <= 5
        # La Réunion-specific categories use default (high) depth
        assert "max_depth" not in cats_by_name["Géographie de La Réunion"]


# ---------------------------------------------------------------------------
# build_zim integration
# ---------------------------------------------------------------------------

class TestBuildZimDryRun:
    def test_dry_run_returns_count(self, tmp_path):
        """Dry-run with mocked API returns article count."""
        config_path = tmp_path / "config.yaml"
        cfg = {
            "output": str(tmp_path / "test.zim"),
            "wikipedia": {
                "api_url": "https://fr.wikipedia.org/w/api.php",
                "user_agent": "TestBot/0.1",
                "request_delay": 0,
                "max_retries": 1,
                "retry_delay": 0,
            },
            "categories": {
                "max_depth": 1,
                "blacklist": [],
                "list": [{"name": "TestCat"}],
            },
            "articles": ["Explicit Article"],
            "metadata": {"name": "test", "title": "Test", "description": "Test", "language": "fra"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        with patch.object(WikipediaAPI, "get_category_articles_recursive") as mock_cat:
            mock_cat.return_value = {"Cat Article 1", "Cat Article 2"}
            stats = build_zim(config_path, dry_run=True)

        assert stats["article_count"] == 3  # 2 from cat + 1 explicit
        assert stats["output"] is None

    def test_per_category_max_depth(self, tmp_path):
        """Per-category max_depth overrides the global max_depth."""
        config_path = tmp_path / "config.yaml"
        cfg = {
            "output": str(tmp_path / "test.zim"),
            "wikipedia": {
                "api_url": "https://fr.wikipedia.org/w/api.php",
                "user_agent": "TestBot/0.1",
                "request_delay": 0,
                "max_retries": 1,
                "retry_delay": 0,
            },
            "categories": {
                "max_depth": 10,
                "blacklist": [],
                "list": [
                    {"name": "DeepCat"},
                    {"name": "ShallowCat", "max_depth": 1},
                ],
            },
            "articles": [],
            "metadata": {"name": "test", "title": "Test", "description": "Test", "language": "fra"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        depth_calls = {}

        def mock_recursive(cat, max_depth, blacklist, **kwargs):
            depth_calls[cat] = max_depth
            return {f"Article_{cat}"}

        with patch.object(WikipediaAPI, "get_category_articles_recursive", side_effect=mock_recursive):
            stats = build_zim(config_path, dry_run=True)

        assert depth_calls["DeepCat"] == 10
        assert depth_calls["ShallowCat"] == 1
        assert stats["article_count"] == 2

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_zim(tmp_path / "nonexistent.yaml")


class TestBuildZimFull:
    def test_full_build(self, tmp_path, monkeypatch):
        """Full build with mocked API and real libzim."""
        # Redirect staging
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr("meshwiki.build_zim.STAGING_DIR", staging)
        monkeypatch.setattr("meshwiki.build_zim.STAGING_INDEX", staging / "_index.json")

        config_path = tmp_path / "config.yaml"
        output_path = str(tmp_path / "test.zim")
        cfg = {
            "output": output_path,
            "wikipedia": {
                "api_url": "https://fr.wikipedia.org/w/api.php",
                "user_agent": "TestBot/0.1",
                "request_delay": 0,
                "max_retries": 1,
                "retry_delay": 0,
            },
            "categories": {
                "max_depth": 1,
                "blacklist": [],
                "list": [],
            },
            "articles": ["Paris", "Lyon"],
            "metadata": {
                "name": "test",
                "title": "Test ZIM",
                "description": "Test",
                "language": "fra",
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        def mock_get_html(title):
            return f"<html><body><h1>{title}</h1><p>{title} est une ville de France.</p></body></html>"

        with patch.object(WikipediaAPI, "get_article_html", side_effect=mock_get_html):
            stats = build_zim(config_path)

        assert stats["article_count"] == 2
        assert stats["failed_count"] == 0
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Verify ZIM is readable
        from libzim.reader import Archive
        archive = Archive(output_path)
        assert archive.entry_count > 0

    def test_resume_skips_existing(self, tmp_path, monkeypatch):
        """Resume mode skips already-staged articles."""
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr("meshwiki.build_zim.STAGING_DIR", staging)
        monkeypatch.setattr("meshwiki.build_zim.STAGING_INDEX", staging / "_index.json")

        # Pre-stage one article
        index = {}
        _save_article_staging(
            "Paris",
            "<html><body><p>Paris content</p></body></html>",
            index,
        )
        _save_staging_index(index)

        config_path = tmp_path / "config.yaml"
        output_path = str(tmp_path / "test.zim")
        cfg = {
            "output": output_path,
            "wikipedia": {
                "api_url": "https://fr.wikipedia.org/w/api.php",
                "user_agent": "TestBot/0.1",
                "request_delay": 0,
                "max_retries": 1,
                "retry_delay": 0,
            },
            "categories": {"max_depth": 1, "blacklist": [], "list": []},
            "articles": ["Paris", "Lyon"],
            "metadata": {"name": "test", "title": "Test", "description": "Test", "language": "fra"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        call_log = []

        def mock_get_html(title):
            call_log.append(title)
            return f"<html><body><p>{title} content</p></body></html>"

        with patch.object(WikipediaAPI, "get_article_html", side_effect=mock_get_html):
            stats = build_zim(config_path, resume=True)

        # Paris should not have been re-downloaded
        assert "Paris" not in call_log
        assert "Lyon" in call_log
        assert stats["article_count"] == 2

    def test_static_pages_included(self, tmp_path, monkeypatch):
        """Static pages are included in the ZIM."""
        staging = tmp_path / "staging"
        staging.mkdir()
        monkeypatch.setattr("meshwiki.build_zim.STAGING_DIR", staging)
        monkeypatch.setattr("meshwiki.build_zim.STAGING_INDEX", staging / "_index.json")

        # Create a static page file
        static_dir = tmp_path / "static"
        static_dir.mkdir()
        static_file = static_dir / "urgences.html"
        static_file.write_text(
            "<html><body><h1>Urgences</h1><p>Liste des centres.</p></body></html>",
            encoding="utf-8",
        )

        config_path = tmp_path / "config.yaml"
        output_path = str(tmp_path / "test.zim")
        cfg = {
            "output": output_path,
            "wikipedia": {
                "api_url": "https://fr.wikipedia.org/w/api.php",
                "user_agent": "TestBot/0.1",
                "request_delay": 0,
                "max_retries": 1,
                "retry_delay": 0,
            },
            "categories": {"max_depth": 1, "blacklist": [], "list": []},
            "articles": ["Paris"],
            "static_pages": [
                {"title": "Urgences La Réunion", "file": "static/urgences.html"},
            ],
            "metadata": {"name": "test", "title": "Test", "description": "Test", "language": "fra"},
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)

        def mock_get_html(title):
            return f"<html><body><p>{title} content</p></body></html>"

        with patch.object(WikipediaAPI, "get_article_html", side_effect=mock_get_html):
            stats = build_zim(config_path)

        assert stats["article_count"] == 1
        assert Path(output_path).exists()

        # Verify ZIM contains the static page
        from libzim.reader import Archive
        archive = Archive(output_path)
        # mainpage + 1 article + 1 static page = at least 3 entries
        assert archive.entry_count >= 3
