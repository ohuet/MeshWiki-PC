"""Fixed three-line terminal progress display with ANSI escape codes.

Displays three fixed lines at the bottom of the terminal:
- Info line: current action (e.g. "── Encodage et insertion de 5000 chunks...")
- Sub-progress line: encoding detail (e.g. "  Distant 8/20 [████░░] | Local 3/20 [██░░]")
- Progress line: bar with percentage, article count, and ETA

Log messages scroll above the fixed lines. Falls back to normal logging
when stderr is not a TTY (e.g. redirected to a file).
"""

import logging
import sys
import threading


BAR_WIDTH = 20
_NUM_FIXED = 3


def format_bar(percent: float, width: int = BAR_WIDTH) -> str:
    """Format a progress bar string: [████░░░░] without brackets."""
    filled = int(width * percent / 100)
    return "\u2588" * filled + "\u2591" * (width - filled)


class ProgressDisplay:
    """Maintain three fixed lines at the bottom of the terminal."""

    def __init__(self):
        self._progress_line = ""
        self._info_line = ""
        self._sub_progress_line = ""
        self._active = False
        self._is_tty = sys.stderr.isatty()
        self._lock = threading.Lock()
        self._original_handlers: list[logging.Handler] = []

    def start(self):
        """Activate the display and install the custom logging handler."""
        if not self._is_tty:
            return

        with self._lock:
            self._active = True
            self._sub_progress_line = ""

            root = logging.getLogger()
            self._original_handlers = list(root.handlers)

            handler = _StatusHandler(self)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))

            # Keep non-stderr handlers (e.g. RotatingFileHandler) active
            kept = [h for h in root.handlers
                    if getattr(h, 'stream', None) is not sys.stderr]
            root.handlers = [handler] + kept

            # Reserve three lines
            sys.stderr.write("\n" * _NUM_FIXED)
            sys.stderr.flush()

    def stop(self):
        """Clear the fixed lines and restore original logging handlers."""
        if not self._is_tty:
            return

        with self._lock:
            if not self._active:
                return
            self._active = False

            # Erase the three fixed lines
            erase = ""
            for i in range(_NUM_FIXED):
                if i > 0:
                    erase += "\033[A"
                erase += "\033[K"
            sys.stderr.write(f"\033[{_NUM_FIXED}A" + erase)
            sys.stderr.flush()

            root = logging.getLogger()
            # Remove _StatusHandler, keep file handlers that were active during display
            non_stderr = [h for h in root.handlers
                          if not isinstance(h, _StatusHandler)]
            # Re-add original stderr handlers that were removed in start()
            stderr_originals = [h for h in self._original_handlers
                                if getattr(h, 'stream', None) is sys.stderr]
            root.handlers = stderr_originals + non_stderr
            self._original_handlers = []

    def set_progress(self, percent: float, articles: int, eta_dur: str, eta_time: str):
        """Update the progress bar (bottom line)."""
        bar = format_bar(percent)
        self._progress_line = (
            f"[{bar}] {percent:.1f}% \u2014 {articles} articles "
            f"\u2014 reste {eta_dur} (fin ~{eta_time})"
        )
        if self._active:
            self._redraw()

    def set_info(self, message: str):
        """Update the info line (top fixed line)."""
        self._info_line = f"\u2500\u2500 {message}"
        if self._active:
            self._redraw()

    def set_sub_progress(self, message: str):
        """Update the sub-progress line (middle fixed line)."""
        self._sub_progress_line = message
        if self._active:
            self._redraw()

    def _fixed_lines(self) -> str:
        """Return the three fixed lines as a single string for writing."""
        return (
            "\033[K" + self._info_line + "\n"
            "\033[K" + self._sub_progress_line + "\n"
            "\033[K" + self._progress_line + "\n"
        )

    def _redraw(self):
        """Redraw the three fixed lines using ANSI escape codes."""
        with self._lock:
            if not self._active:
                return
            sys.stderr.write(f"\033[{_NUM_FIXED}A" + self._fixed_lines())
            sys.stderr.flush()


class _StatusHandler(logging.StreamHandler):
    """Logging handler that writes messages above the fixed progress lines."""

    def __init__(self, display: ProgressDisplay):
        super().__init__(sys.stderr)
        self._display = display

    def emit(self, record):
        with self._display._lock:
            if not self._display._active:
                super().emit(record)
                return

            try:
                msg = self.format(record)
                # Clear trailing content on each line of multi-line messages
                # (prevents old fixed-line text from bleeding through)
                msg = msg.replace("\n", "\033[K\n")
                # Move up N lines, clear, write log message, then redraw fixed lines
                sys.stderr.write(
                    f"\033[{_NUM_FIXED}A"
                    "\033[K"
                    + msg + "\033[K\n"
                    + self._display._fixed_lines()
                )
                sys.stderr.flush()
            except Exception:
                self.handleError(record)
