"""Tests for meshwiki.main — startup logic."""

import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import meshwiki.main as main_module
import meshwiki.kiwix_search as ks


def test_wants_indexation_with_flag():
    """_wants_indexation returns True when --index is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "--index"]):
        assert main_module._wants_indexation() is True


def test_wants_indexation_with_slash_flag():
    """_wants_indexation returns True when /index is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "/index"]):
        assert main_module._wants_indexation() is True


def test_wants_indexation_with_dash_flag():
    """_wants_indexation returns True when -index is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "-index"]):
        assert main_module._wants_indexation() is True


def test_wants_indexation_without_flag():
    """_wants_indexation returns False when no index flag is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki"]):
        assert main_module._wants_indexation() is False


def test_wants_update_with_flag():
    """_wants_update returns True when --update is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "--update"]):
        assert main_module._wants_update() is True


def test_wants_update_with_slash_flag():
    """_wants_update returns True when /update is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "/update"]):
        assert main_module._wants_update() is True


def test_wants_update_without_flag():
    """_wants_update returns False when no update flag is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki"]):
        assert main_module._wants_update() is False


def test_wants_offline_with_flag():
    """_wants_offline returns True when --offline is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "--offline"]):
        assert main_module._wants_offline() is True


def test_wants_offline_with_slash_flag():
    """_wants_offline returns True when /offline is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "/offline"]):
        assert main_module._wants_offline() is True


def test_wants_offline_without_flag():
    """_wants_offline returns False when no offline flag is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki"]):
        assert main_module._wants_offline() is False


def test_wants_noindex_with_flag():
    """_wants_noindex returns True when --noindex is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "--noindex"]):
        assert main_module._wants_noindex() is True


def test_wants_noindex_with_slash_flag():
    """_wants_noindex returns True when /noindex is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "/noindex"]):
        assert main_module._wants_noindex() is True


def test_wants_noindex_without_flag():
    """_wants_noindex returns False when no noindex flag is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki"]):
        assert main_module._wants_noindex() is False


def test_wants_nowiki_with_flag():
    """_wants_nowiki returns True when --nowiki is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "--nowiki"]):
        assert main_module._wants_nowiki() is True


def test_wants_nowiki_with_slash_flag():
    """_wants_nowiki returns True when /nowiki is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki", "/nowiki"]):
        assert main_module._wants_nowiki() is True


def test_wants_nowiki_without_flag():
    """_wants_nowiki returns False when no nowiki flag is in sys.argv."""
    with patch.object(sys, "argv", ["meshwiki"]):
        assert main_module._wants_nowiki() is False


def test_zim_watcher_detects_new_file(tmp_path):
    """ZIM watcher detects new files and updates kiwix_search paths."""
    ks.set_zim_paths([])
    ks._archives.clear()

    # Create initial ZIM
    (tmp_path / "a.zim").write_bytes(b"x" * 100)

    with patch("meshwiki.main.discover_zims") as mock_discover:
        # First call: initial state
        mock_discover.return_value = [tmp_path / "a.zim"]
        stop = main_module._start_zim_watcher(interval=0)

        # Simulate a new ZIM appearing
        (tmp_path / "b.zim").write_bytes(b"x" * 200)
        mock_discover.return_value = [tmp_path / "a.zim", tmp_path / "b.zim"]

        time.sleep(0.3)
        stop.set()

    assert ks.has_zim_paths()
