"""Fixed two-line terminal progress display with ANSI escape codes.

Displays two fixed lines at the bottom of the terminal:
- Info line: current action (e.g. "── Encodage et insertion de 5000 chunks...")
- Progress line: bar with percentage, article count, and ETA

Log messages scroll above the fixed lines. Falls back to normal logging
when stderr is not a TTY (e.g. redirected to a file).
"""

import logging
import sys
import threading


BAR_WIDTH = 20


class ProgressDisplay:
    """Maintain two fixed lines at the bottom of the terminal."""

    def __init__(self):
        self._progress_line = ""
        self._info_line = ""
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

            root = logging.getLogger()
            self._original_handlers = list(root.handlers)

            handler = _StatusHandler(self)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))

            root.handlers = [handler]

            # Reserve two lines
            sys.stderr.write("\n\n")
            sys.stderr.flush()

    def stop(self):
        """Clear the fixed lines and restore original logging handlers."""
        if not self._is_tty:
            return

        with self._lock:
            if not self._active:
                return
            self._active = False

            # Erase the two fixed lines
            sys.stderr.write("\033[2A\033[K\033[B\033[K\033[A")
            sys.stderr.flush()

            root = logging.getLogger()
            root.handlers = list(self._original_handlers)
            self._original_handlers = []

    def set_progress(self, percent: float, articles: int, eta_dur: str, eta_time: str):
        """Update the progress bar (bottom line)."""
        filled = int(BAR_WIDTH * percent / 100)
        bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)
        self._progress_line = (
            f"[{bar}] {percent:.1f}% \u2014 {articles} articles "
            f"\u2014 reste {eta_dur} (fin ~{eta_time})"
        )
        if self._active:
            self._redraw()

    def set_info(self, message: str):
        """Update the info line (above the progress bar)."""
        self._info_line = f"\u2500\u2500 {message}"
        if self._active:
            self._redraw()

    def _redraw(self):
        """Redraw the two fixed lines using ANSI escape codes."""
        with self._lock:
            if not self._active:
                return
            sys.stderr.write(
                "\033[2A"        # move up 2 lines
                "\033[K"         # clear info line
                + self._info_line + "\n"
                "\033[K"         # clear progress line
                + self._progress_line + "\n"
            )
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
                # Move up 2 lines, clear, write log message, then redraw fixed lines
                sys.stderr.write(
                    "\033[2A"    # move up 2 lines
                    "\033[K"     # clear line
                    + msg + "\n"
                    "\033[K"     # clear info line
                    + self._display._info_line + "\n"
                    "\033[K"     # clear progress line
                    + self._display._progress_line + "\n"
                )
                sys.stderr.flush()
            except Exception:
                self.handleError(record)
