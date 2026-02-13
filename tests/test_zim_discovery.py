"""Tests for meshwiki.zim_discovery — ZIM file discovery and priority sorting."""

from pathlib import Path

from meshwiki.zim_discovery import discover_zims


def test_discover_empty_dir(tmp_path):
    """Empty directory returns empty list."""
    assert discover_zims(str(tmp_path)) == []


def test_discover_nonexistent_dir():
    """Non-existent directory returns empty list."""
    assert discover_zims("/nonexistent/path") == []


def test_discover_single_zim(tmp_path):
    """Finds a single .zim file."""
    zim = tmp_path / "test.zim"
    zim.write_bytes(b"x" * 100)
    result = discover_zims(str(tmp_path))
    assert len(result) == 1
    assert result[0].name == "test.zim"


def test_discover_sorted_by_size(tmp_path):
    """ZIM files are sorted by size ascending (smallest = highest priority)."""
    small = tmp_path / "custom.zim"
    small.write_bytes(b"x" * 100)
    large = tmp_path / "wikipedia.zim"
    large.write_bytes(b"x" * 10000)
    medium = tmp_path / "regional.zim"
    medium.write_bytes(b"x" * 1000)

    result = discover_zims(str(tmp_path))
    assert len(result) == 3
    assert result[0].name == "custom.zim"
    assert result[1].name == "regional.zim"
    assert result[2].name == "wikipedia.zim"


def test_discover_recursive(tmp_path):
    """Finds .zim files in subdirectories."""
    sub = tmp_path / "tmp"
    sub.mkdir()
    (tmp_path / "root.zim").write_bytes(b"x" * 50)
    (sub / "nested.zim").write_bytes(b"x" * 200)

    result = discover_zims(str(tmp_path))
    assert len(result) == 2
    assert result[0].name == "root.zim"
    assert result[1].name == "nested.zim"


def test_discover_excludes_downloads(tmp_path):
    """Files with .zim.download extension are excluded."""
    (tmp_path / "good.zim").write_bytes(b"x" * 100)
    (tmp_path / "partial.zim.download").write_bytes(b"x" * 100)

    result = discover_zims(str(tmp_path))
    assert len(result) == 1
    assert result[0].name == "good.zim"


def test_discover_ignores_non_zim(tmp_path):
    """Non-.zim files are ignored."""
    (tmp_path / "readme.txt").write_bytes(b"hello")
    (tmp_path / "data.json").write_bytes(b"{}")
    (tmp_path / "real.zim").write_bytes(b"x" * 100)

    result = discover_zims(str(tmp_path))
    assert len(result) == 1
    assert result[0].name == "real.zim"
