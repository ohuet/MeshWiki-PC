"""Discover ZIM files in the data directory, sorted by size (smallest first)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_zims(data_dir: str = "data") -> list[Path]:
    """Find all .zim files in data_dir (recursive), sorted by size ascending.

    Smaller ZIM files are assumed to be more specific (e.g. custom articles)
    and get higher priority. Files ending in .zim.download (in-progress
    downloads) are excluded.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    zim_files = [
        p for p in data_path.rglob("*.zim")
        if not p.name.endswith(".zim.download") and p.is_file()
    ]
    zim_files.sort(key=lambda p: p.stat().st_size)

    if zim_files:
        logger.debug(
            "ZIM découverts (%d) : %s",
            len(zim_files),
            ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.1f} Mo)" for p in zim_files),
        )

    return zim_files
