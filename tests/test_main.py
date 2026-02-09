"""Tests for meshwiki.main — startup logic."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import meshwiki.main as main_module


def test_find_existing_zim_found(tmp_path):
    """When a .zim file exists in temp_dir, _find_existing_zim returns its path."""
    zim_file = tmp_path / "wikipedia_fr_all_mini_2024-01.zim"
    zim_file.write_bytes(b"fake zim")

    config = {"updater": {"temp_dir": str(tmp_path)}}
    result = main_module._find_existing_zim(config)

    assert result is not None
    assert result.name == "wikipedia_fr_all_mini_2024-01.zim"


def test_find_existing_zim_not_found(tmp_path):
    """When no .zim file exists, _find_existing_zim returns None."""
    config = {"updater": {"temp_dir": str(tmp_path)}}
    result = main_module._find_existing_zim(config)

    assert result is None


def test_find_existing_zim_missing_dir():
    """When temp_dir doesn't exist, _find_existing_zim returns None."""
    config = {"updater": {"temp_dir": "/nonexistent/path"}}
    result = main_module._find_existing_zim(config)

    assert result is None


def test_find_existing_zim_returns_most_recent(tmp_path):
    """When multiple .zim files exist, returns the most recently modified."""
    import time
    old_zim = tmp_path / "wikipedia_fr_all_mini_2023-01.zim"
    old_zim.write_bytes(b"old")
    time.sleep(0.05)
    new_zim = tmp_path / "wikipedia_fr_all_mini_2024-01.zim"
    new_zim.write_bytes(b"new")

    config = {"updater": {"temp_dir": str(tmp_path)}}
    result = main_module._find_existing_zim(config)

    assert result.name == "wikipedia_fr_all_mini_2024-01.zim"


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
