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
