"""Tests for meshwiki.wikipedia_updater — download and re-indexation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import meshwiki.wikipedia_updater as updater_module
from meshwiki.wikipedia_updater import WikipediaUpdater


MOCK_CONFIG = {
    "updater": {
        "kiwix_url": "https://download.kiwix.org/zim/wikipedia/?C=M;O=D",
        "dump_pattern": "wikipedia_fr_all_mini_",
        "temp_dir": "./data/tmp",
        "max_download_retries": 3,
        "download_timeout_hours": 24,
    },
    "vectordb": {"path": "./data/chroma_db"},
    "embeddings": {"model": "test-model", "chunk_size": 500, "chunk_overlap": 50},
}

KIWIX_HTML = """
<html><body>
<a href="wikipedia_fr_all_mini_2025-01.zim">wikipedia_fr_all_mini_2025-01.zim</a>
<a href="wikipedia_fr_all_mini_2024-12.zim">wikipedia_fr_all_mini_2024-12.zim</a>
<a href="wikipedia_en_all_2025-01.zim">wikipedia_en_all_2025-01.zim</a>
</body></html>
"""


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_finds_match(mock_get, mock_config):
    mock_response = MagicMock()
    mock_response.text = KIWIX_HTML
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is not None
    url, filename = result
    assert filename == "wikipedia_fr_all_mini_2025-01.zim"
    assert "wikipedia_fr_all_mini_2025-01.zim" in url


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_no_match(mock_get, mock_config):
    mock_response = MagicMock()
    mock_response.text = "<html><a href='other_file.txt'>nope</a></html>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is None


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_network_error(mock_get, mock_config):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError()

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is None


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
def test_download_dump_skips_if_up_to_date(mock_file, mock_config):
    mock_file.exists.return_value = True
    mock_file.__fspath__ = lambda self: "data/last_update.json"

    updater = WikipediaUpdater()

    with patch.object(updater, "get_latest_dump_url", return_value=("http://example.com/file.zim", "wikipedia_fr_all_mini_2025-01.zim")), \
         patch("builtins.open", mock_open(read_data=json.dumps({"last_filename": "wikipedia_fr_all_mini_2025-01.zim"}))):
        result = updater.download_dump()

    assert result is None


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
def test_download_dump_returns_none_when_no_url(mock_config):
    updater = WikipediaUpdater()

    with patch.object(updater, "get_latest_dump_url", return_value=None):
        result = updater.download_dump()

    assert result is None


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
@patch("meshwiki.wikipedia_updater.requests.get")
def test_download_skips_if_zim_already_exists(mock_get, mock_file, mock_config, tmp_path):
    """If the .zim file is already on disk, skip download entirely."""
    mock_file.exists.return_value = False  # LAST_UPDATE_FILE missing

    zim_file = tmp_path / "wiki_2025.zim"
    zim_file.write_bytes(b"existing zim")

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)

    with patch.object(updater, "get_latest_dump_url",
                      return_value=("http://example.com/wiki_2025.zim", "wiki_2025.zim")):
        result = updater.download_dump()

    assert result == zim_file
    mock_get.assert_not_called()  # No HTTP request made


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.reset_collection")
@patch("meshwiki.wikipedia_updater.index_zim")
@patch("meshwiki.wikipedia_updater.collection_state")
@patch("meshwiki.wikipedia_updater.discover_zims")
def test_reindex_success(mock_discover, mock_cs, mock_index_zim, mock_reset, mock_config):
    mock_index_zim.return_value = {"article_count": 100, "chunk_count": 500}
    mock_cs.get_inactive_db_path.return_value = "./data/chroma_db_b"
    mock_cs.get_inactive_slot.return_value = "b"
    mock_cs.get_active_db_path.return_value = "./data/chroma_db_a"

    zim_path = MagicMock(spec=Path)
    zim_path.name = "test.zim"
    mock_discover.return_value = [zim_path]

    updater = WikipediaUpdater()

    with patch("builtins.open", mock_open()):
        result = updater.reindex(zim_path)

    assert result is True
    # Indexes into the inactive database (single ZIM path)
    mock_index_zim.assert_called_once_with(zim_path, db_path="./data/chroma_db_b")
    # Pointer is updated to new slot
    mock_cs.set_active_slot.assert_called_once_with("b")
    # RAG cache is invalidated after swap
    mock_reset.assert_called_once()
    # Old database cleanup is deferred to next startup (_cleanup_inactive_db)


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.index_zim")
@patch("meshwiki.wikipedia_updater.collection_state")
@patch("meshwiki.wikipedia_updater.discover_zims", return_value=[])
def test_reindex_failure_preserves_old_index(mock_discover, mock_cs, mock_index_zim, mock_config):
    mock_index_zim.side_effect = Exception("Indexation error")
    mock_cs.get_inactive_db_path.return_value = "./data/chroma_db_b"
    mock_cs.get_inactive_slot.return_value = "b"

    zim_path = MagicMock(spec=Path)
    mock_discover.return_value = [zim_path]

    updater = WikipediaUpdater()
    result = updater.reindex(zim_path)

    assert result is False
    # Pointer is NOT updated on failure — old index stays active
    mock_cs.set_active_slot.assert_not_called()


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
def test_cleanup_keeps_zim_file(mock_config, tmp_path):
    """cleanup() must NOT delete .zim files (kept as Kiwix fallback)."""
    zim_file = tmp_path / "test.zim"
    zim_file.write_bytes(b"x" * 1024)

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)
    updater.cleanup()

    assert zim_file.exists()


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
def test_cleanup_removes_temp_files_but_not_zim(mock_config, tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    (temp_dir / "partial.zim.download").write_bytes(b"partial")
    zim_file = temp_dir / "wiki.zim"
    zim_file.write_bytes(b"zim data")

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(temp_dir)
    updater.cleanup()

    remaining = [f.name for f in temp_dir.iterdir()]
    assert "wiki.zim" in remaining
    assert "partial.zim.download" not in remaining


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
@patch("meshwiki.wikipedia_updater.requests.get")
def test_download_replaces_old_zim_only_after_completion(mock_get, mock_file, mock_config, tmp_path):
    """Download writes to .zim.download, then replaces the .zim atomically."""
    mock_file.exists.return_value = False

    # Simulate a small download
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "4"}
    mock_response.iter_content.return_value = [b"data"]
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)

    with patch.object(updater, "get_latest_dump_url",
                      return_value=("http://example.com/wiki.zim", "wiki_2025.zim")):
        result = updater.download_dump()

    # Final .zim exists, .download does not
    assert result == tmp_path / "wiki_2025.zim"
    assert result.exists()
    assert not (tmp_path / "wiki_2025.zim.download").exists()


@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
@patch("meshwiki.wikipedia_updater.requests.get")
def test_download_failure_does_not_create_partial_zim(mock_get, mock_file, mock_config, tmp_path):
    """If download fails for a new version, no .zim is created."""
    mock_file.exists.return_value = False
    import requests as req
    mock_get.side_effect = req.exceptions.ConnectionError("network down")

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)

    with patch.object(updater, "get_latest_dump_url",
                      return_value=("http://example.com/wiki_2026.zim", "wiki_2026.zim")):
        result = updater.download_dump()

    assert result is None
    assert not (tmp_path / "wiki_2026.zim").exists()


# --- Removal of older dump versions after a successful download ---


def _touch(directory, *names):
    for name in names:
        (directory / name).write_bytes(b"zim")


@patch("meshwiki.wikipedia_updater.discover_zims", return_value=[])
@patch("meshwiki.wikipedia_updater.kiwix_search")
def test_remove_older_versions_only_same_name_older_date(mock_kiwix, mock_discover, tmp_path):
    """Only files with the exact same name and an earlier date are deleted."""
    _touch(
        tmp_path,
        "wikipedia_fr_all_mini_2026-05.zim",       # the new download
        "wikipedia_fr_all_mini_2026-01.zim",       # older version → deleted
        "wikipedia_fr_all_mini_2026-02.zim",       # older version → deleted
        "wikipedia_fr_all_mini_2026-09.zim",       # newer date → kept
        "wikipedia_fr_all_maxi_2026-01.zim",       # different variant → kept
        "wikipedia_fr_all_2026-01.zim",            # shorter name → kept
        "xwikipedia_fr_all_mini_2026-01.zim",      # longer name → kept
        "wikipedia_fr_all_mini_2026-01.zim.download",  # not a .zim → kept
        "meshwiki_reunion.zim",                    # undated → kept
    )

    updater_module._remove_older_versions(tmp_path / "wikipedia_fr_all_mini_2026-05.zim")

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == sorted([
        "wikipedia_fr_all_mini_2026-05.zim",
        "wikipedia_fr_all_mini_2026-09.zim",
        "wikipedia_fr_all_maxi_2026-01.zim",
        "wikipedia_fr_all_2026-01.zim",
        "xwikipedia_fr_all_mini_2026-01.zim",
        "wikipedia_fr_all_mini_2026-01.zim.download",
        "meshwiki_reunion.zim",
    ])


@patch("meshwiki.wikipedia_updater.discover_zims")
@patch("meshwiki.wikipedia_updater.kiwix_search")
def test_remove_older_versions_releases_kiwix_archives_first(mock_kiwix, mock_discover, tmp_path):
    """Kiwix stops using the old dumps before they are deleted (open files can't be deleted on Windows)."""
    _touch(tmp_path, "wiki_fr_2026-05.zim", "wiki_fr_2026-02.zim", "other.zim")
    old, new, other = tmp_path / "wiki_fr_2026-02.zim", tmp_path / "wiki_fr_2026-05.zim", tmp_path / "other.zim"
    mock_discover.return_value = [other, old, new]

    def _check_still_present(paths):
        assert old.exists()
        assert [str(p) for p in paths] == [str(other), str(new)]

    mock_kiwix.set_zim_paths.side_effect = _check_still_present

    updater_module._remove_older_versions(new)

    mock_kiwix.set_zim_paths.assert_called_once()
    assert not old.exists()


@patch("meshwiki.wikipedia_updater.kiwix_search")
def test_remove_older_versions_ignores_undated_name(mock_kiwix, tmp_path):
    _touch(tmp_path, "wiki.zim", "wiki_2026-01.zim")

    updater_module._remove_older_versions(tmp_path / "wiki.zim")

    assert (tmp_path / "wiki_2026-01.zim").exists()
    mock_kiwix.set_zim_paths.assert_not_called()


@patch("meshwiki.wikipedia_updater.discover_zims", return_value=[])
@patch("meshwiki.wikipedia_updater.kiwix_search")
@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
@patch("meshwiki.wikipedia_updater.requests.get")
def test_download_success_removes_previous_dump(mock_get, mock_file, mock_config, mock_kiwix, mock_discover, tmp_path):
    mock_file.exists.return_value = False
    _touch(tmp_path, "wikipedia_fr_all_mini_2026-02.zim")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": "4"}
    mock_response.iter_content.return_value = [b"data"]
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)

    with patch.object(updater, "get_latest_dump_url",
                      return_value=("http://example.com/x.zim", "wikipedia_fr_all_mini_2026-05.zim")):
        result = updater.download_dump()

    assert result == tmp_path / "wikipedia_fr_all_mini_2026-05.zim"
    assert [p.name for p in tmp_path.iterdir()] == ["wikipedia_fr_all_mini_2026-05.zim"]


@patch("meshwiki.wikipedia_updater.kiwix_search")
@patch("meshwiki.config.load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
@patch("meshwiki.wikipedia_updater.requests.get")
def test_download_failure_keeps_previous_dump(mock_get, mock_file, mock_config, mock_kiwix, tmp_path):
    mock_file.exists.return_value = False
    _touch(tmp_path, "wikipedia_fr_all_mini_2026-02.zim")
    import requests as req
    mock_get.side_effect = req.exceptions.ConnectionError("network down")

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)

    with patch.object(updater, "get_latest_dump_url",
                      return_value=("http://example.com/x.zim", "wikipedia_fr_all_mini_2026-05.zim")):
        assert updater.download_dump() is None

    assert (tmp_path / "wikipedia_fr_all_mini_2026-02.zim").exists()
