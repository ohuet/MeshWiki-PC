"""Shared state for the active ChromaDB database.

Two databases alternate: data/chroma_a and data/chroma_b.
A JSON pointer file tracks which slot is active.
The swap is instantaneous (just a file write with fsync).
Cleanup of the old database is a simple shutil.rmtree.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

POINTER_FILE = Path("data/active_collection.json")
_SLOTS = ("a", "b")
COLLECTION_NAME = "wikipedia"
_SQLITE_FILE = "chroma.sqlite3"


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


def index_exists(expected_dim: int | None) -> bool:
    """Check that the active database holds a non-empty, compatible collection.

    Reads ChromaDB's SQLite catalog directly (read-only) instead of opening a
    PersistentClient: any Chroma read (count, peek) loads the whole HNSW index
    into memory — about 13 GB for French Wikipedia — which must only happen
    when the first question arrives.
    """
    db_path = Path(get_active_db_path())
    sqlite_path = db_path / _SQLITE_FILE
    if not sqlite_path.exists():
        return False

    try:
        con = sqlite3.connect(f"file:{sqlite_path.as_posix()}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT id, dimension FROM collections WHERE name = ?",
                (COLLECTION_NAME,),
            ).fetchone()
            if row is None:
                return False
            collection_id, dimension = row
            has_records = con.execute(
                "SELECT EXISTS (SELECT 1 FROM embeddings e"
                " JOIN segments s ON e.segment_id = s.id"
                " WHERE s.collection = ? AND s.scope = 'METADATA')",
                (collection_id,),
            ).fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error as e:
        # Unknown catalog layout (Chroma upgrade?): fall back to the slow check
        logger.warning("Catalogue ChromaDB illisible (%s), vérification via ChromaDB : %s", sqlite_path, e)
        return _index_exists_via_chromadb(db_path, expected_dim)

    if not has_records:
        return False
    if expected_dim and dimension != expected_dim:
        logger.warning(
            "Index incompatible : dimension %s (attendu %d) — réindexation nécessaire",
            dimension, expected_dim,
        )
        return False
    return True


def _index_exists_via_chromadb(db_path: Path, expected_dim: int | None) -> bool:
    """Same check through the ChromaDB API — loads the vector index into memory."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            return False
        if expected_dim:
            sample = collection.peek(limit=1)
            if len(sample["embeddings"]) > 0 and len(sample["embeddings"][0]) != expected_dim:
                return False
        return True
    except Exception as e:
        logger.warning("Vérification de l'index échouée (%s): %s", db_path, e)
        return False
