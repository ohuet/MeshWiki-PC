"""Shared state for the active ChromaDB database.

Two databases alternate: data/chroma_a and data/chroma_b.
A JSON pointer file tracks which slot is active.
The swap is instantaneous (just a file write with fsync).
Cleanup of the old database is a simple shutil.rmtree.
"""

import json
import os
from pathlib import Path

POINTER_FILE = Path("data/active_collection.json")
_SLOTS = ("a", "b")


def _base_path() -> str:
    """Return the base vectordb path from config (without trailing slot suffix)."""
    from meshwiki import config
    return config.load_config()["vectordb"]["path"]


def get_active_slot() -> str:
    """Return the active slot letter ("a" or "b").

    Falls back to legacy detection when no pointer exists:
    if data/chroma_db exists, returns "legacy".
    """
    if not POINTER_FILE.exists():
        return "legacy"
    try:
        with open(POINTER_FILE) as f:
            data = json.load(f)
        return data["active"]
    except (json.JSONDecodeError, KeyError, OSError):
        return "legacy"


def set_active_slot(slot: str) -> None:
    """Write the active slot to the pointer file (with fsync)."""
    POINTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POINTER_FILE, "w") as f:
        json.dump({"active": slot}, f)
        f.flush()
        os.fsync(f.fileno())


def get_active_db_path() -> str:
    """Return the filesystem path to the active ChromaDB database."""
    slot = get_active_slot()
    if slot == "legacy":
        return _base_path()
    return _base_path() + "_" + slot


def get_inactive_slot() -> str:
    """Return the inactive slot letter (the one to index into next)."""
    slot = get_active_slot()
    if slot == _SLOTS[0]:
        return _SLOTS[1]
    return _SLOTS[0]


def get_inactive_db_path() -> str:
    """Return the filesystem path to the inactive ChromaDB database."""
    return _base_path() + "_" + get_inactive_slot()
