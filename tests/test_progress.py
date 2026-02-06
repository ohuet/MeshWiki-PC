"""Tests for meshwiki.progress — fixed terminal progress display."""

import logging
import io
from unittest.mock import patch, MagicMock

from meshwiki.progress import ProgressDisplay, BAR_WIDTH


def test_non_tty_start_stop():
    """start/stop on a non-TTY does not crash, _active stays False."""
    with patch("meshwiki.progress.sys") as mock_sys:
        mock_sys.stderr = io.StringIO()
        mock_sys.stderr.isatty = lambda: False
        display = ProgressDisplay()
        assert not display._active
        display.start()
        assert not display._active
        display.stop()
        assert not display._active


def test_set_progress_formats_bar():
    """Progress line contains the bar characters, percentage, and article count."""
    display = ProgressDisplay()
    display.set_progress(25.0, 1234, "2h05min", "16:30")
    line = display._progress_line
    assert "\u2588" in line  # filled block
    assert "\u2591" in line  # empty block
    assert "25.0%" in line
    assert "1234 articles" in line
    assert "2h05min" in line
    assert "16:30" in line


def test_set_progress_bar_width():
    """Bar has exactly BAR_WIDTH characters (filled + empty)."""
    display = ProgressDisplay()
    display.set_progress(50.0, 500, "1h", "17:00")
    line = display._progress_line
    # Extract bar content between [ and ]
    bar = line.split("[")[1].split("]")[0]
    assert len(bar) == BAR_WIDTH


def test_set_progress_zero_and_hundred():
    """Edge cases: 0% and 100% produce correct bars."""
    display = ProgressDisplay()

    display.set_progress(0.0, 0, "?", "?")
    bar_0 = display._progress_line.split("[")[1].split("]")[0]
    assert bar_0 == "\u2591" * BAR_WIDTH

    display.set_progress(100.0, 5000, "0s", "18:00")
    bar_100 = display._progress_line.split("[")[1].split("]")[0]
    assert bar_100 == "\u2588" * BAR_WIDTH


def test_set_info_formats_message():
    """Info line uses the ── prefix."""
    display = ProgressDisplay()
    display.set_info("Encodage et insertion de 5000 chunks...")
    assert display._info_line.startswith("\u2500\u2500 ")
    assert "5000 chunks" in display._info_line


def test_handler_integration():
    """With a mocked TTY, logging while display is active writes ANSI sequences."""
    stderr = io.StringIO()
    stderr.isatty = lambda: True  # type: ignore[attr-defined]

    with patch("meshwiki.progress.sys") as mock_sys:
        mock_sys.stderr = stderr
        display = ProgressDisplay()
        display.start()
        assert display._active

        display.set_info("Test info")
        display.set_progress(10.0, 42, "5min", "14:00")

        test_logger = logging.getLogger("test.progress")
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("A log message during progress")

        output = stderr.getvalue()
        # ANSI escape sequences should be present
        assert "\033[2A" in output
        assert "\033[K" in output
        # The log message should appear
        assert "A log message during progress" in output

        display.stop()
        assert not display._active


def test_stop_restores_handlers():
    """stop() restores original logging handlers."""
    stderr = io.StringIO()
    stderr.isatty = lambda: True  # type: ignore[attr-defined]

    root = logging.getLogger()
    original_handlers = list(root.handlers)

    with patch("meshwiki.progress.sys") as mock_sys:
        mock_sys.stderr = stderr
        display = ProgressDisplay()
        display.start()
        # Handlers should have changed
        assert root.handlers != original_handlers
        display.stop()

    # Handlers should be restored
    assert root.handlers == original_handlers


def test_stop_without_start():
    """stop() on a non-started display does not crash."""
    with patch("meshwiki.progress.sys") as mock_sys:
        mock_sys.stderr = io.StringIO()
        mock_sys.stderr.isatty = lambda: True
        display = ProgressDisplay()
        display.stop()  # Should not raise
