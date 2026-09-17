"""Tests for meshwiki.collection_state — pointer file for active database."""

import json

from meshwiki.collection_state import (
    get_active_slot,
    set_active_slot,
    get_active_db_path,
    get_inactive_slot,
    get_inactive_db_path,
    POINTER_FILE,
)


MOCK_CONFIG = {"vectordb": {"path": "./data/chroma_db"}}


def test_get_active_slot_fallback_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", tmp_path / "missing.json")
    assert get_active_slot() == "legacy"


def test_set_and_get_active_slot(monkeypatch, tmp_path):
    pointer = tmp_path / "active.json"
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", pointer)

    set_active_slot("b")
    assert get_active_slot() == "b"

    data = json.loads(pointer.read_text())
    assert data == {"active": "b"}


def test_get_active_db_path_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", tmp_path / "missing.json")
    monkeypatch.setattr("meshwiki.collection_state._base_path", lambda: "./data/chroma_db")
    assert get_active_db_path() == "./data/chroma_db"


def test_get_active_db_path_slot_a(monkeypatch, tmp_path):
    pointer = tmp_path / "active.json"
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", pointer)
    monkeypatch.setattr("meshwiki.collection_state._base_path", lambda: "./data/chroma_db")

    set_active_slot("a")
    assert get_active_db_path() == "./data/chroma_db_a"


def test_get_inactive_slot_when_legacy(monkeypatch, tmp_path):
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", tmp_path / "missing.json")
    # No pointer → legacy → inactive is "a"
    assert get_inactive_slot() == "a"


def test_get_inactive_slot_alternates(monkeypatch, tmp_path):
    pointer = tmp_path / "active.json"
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", pointer)

    set_active_slot("a")
    assert get_inactive_slot() == "b"

    set_active_slot("b")
    assert get_inactive_slot() == "a"


def test_get_inactive_db_path(monkeypatch, tmp_path):
    pointer = tmp_path / "active.json"
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", pointer)
    monkeypatch.setattr("meshwiki.collection_state._base_path", lambda: "./data/chroma_db")

    set_active_slot("a")
    assert get_inactive_db_path() == "./data/chroma_db_b"


def test_get_active_slot_handles_corrupted_file(monkeypatch, tmp_path):
    pointer = tmp_path / "active.json"
    pointer.write_text("not valid json")
    monkeypatch.setattr("meshwiki.collection_state.POINTER_FILE", pointer)

    assert get_active_slot() == "legacy"


# --- index_exists: reads the SQLite catalog without loading the vector index ---

import sqlite3

import meshwiki.collection_state as collection_state_module


def _make_catalog(db_dir, dimension=1024, records=1, collection_name="wikipedia"):
    """Create a minimal ChromaDB-like SQLite catalog in db_dir."""
    db_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_dir / "chroma.sqlite3")
    con.executescript(
        "CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT, dimension INTEGER);"
        "CREATE TABLE segments (id TEXT PRIMARY KEY, type TEXT, scope TEXT, collection TEXT);"
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);"
    )
    con.execute("INSERT INTO collections VALUES ('c1', ?, ?)", (collection_name, dimension))
    con.execute("INSERT INTO segments VALUES ('m1', 'sqlite', 'METADATA', 'c1')")
    con.execute("INSERT INTO segments VALUES ('v1', 'hnsw', 'VECTOR', 'c1')")
    for i in range(records):
        con.execute("INSERT INTO embeddings (segment_id, embedding_id) VALUES ('m1', ?)", (str(i),))
    con.commit()
    con.close()


def _point_to(monkeypatch, db_dir):
    monkeypatch.setattr(collection_state_module, "get_active_db_path", lambda: str(db_dir))


def test_index_exists_true(monkeypatch, tmp_path):
    _make_catalog(tmp_path / "db")
    _point_to(monkeypatch, tmp_path / "db")
    assert collection_state_module.index_exists(1024) is True


def test_index_exists_false_no_db(monkeypatch, tmp_path):
    _point_to(monkeypatch, tmp_path / "missing")
    assert collection_state_module.index_exists(1024) is False


def test_index_exists_false_no_collection(monkeypatch, tmp_path):
    _make_catalog(tmp_path / "db", collection_name="other")
    _point_to(monkeypatch, tmp_path / "db")
    assert collection_state_module.index_exists(1024) is False


def test_index_exists_false_empty(monkeypatch, tmp_path):
    _make_catalog(tmp_path / "db", records=0)
    _point_to(monkeypatch, tmp_path / "db")
    assert collection_state_module.index_exists(1024) is False


def test_index_exists_false_wrong_dimension(monkeypatch, tmp_path):
    _make_catalog(tmp_path / "db", dimension=384)
    _point_to(monkeypatch, tmp_path / "db")
    assert collection_state_module.index_exists(1024) is False


def test_index_exists_never_opens_chromadb(monkeypatch, tmp_path):
    """A readable catalog must not fall back to the ChromaDB API (which loads the index)."""
    _make_catalog(tmp_path / "db")
    _point_to(monkeypatch, tmp_path / "db")

    def _fail(*args, **kwargs):
        raise AssertionError("ChromaDB API must not be used")

    monkeypatch.setattr(collection_state_module, "_index_exists_via_chromadb", _fail)
    assert collection_state_module.index_exists(1024) is True


def test_index_exists_falls_back_on_unknown_schema(monkeypatch, tmp_path):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    sqlite3.connect(db_dir / "chroma.sqlite3").close()  # empty catalog, no tables
    _point_to(monkeypatch, db_dir)
    monkeypatch.setattr(collection_state_module, "_index_exists_via_chromadb", lambda path, dim: True)
    assert collection_state_module.index_exists(1024) is True
