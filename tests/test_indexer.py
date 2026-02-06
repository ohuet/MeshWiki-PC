"""Tests for meshwiki.wikipedia_indexer — ZIM indexation into ChromaDB."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from meshwiki.wikipedia_indexer import (
    _clean_html, _chunk_text, _is_content_article,
    _load_checkpoint, _save_checkpoint, CHECKPOINT_FILE,
    get_indexing_eta, _set_indexing_eta,
    _gpu_lock_clocks, _gpu_unlock_clocks,
)


def test_clean_html_strips_tags():
    html = "<p>Bonjour <b>le monde</b></p>"
    assert _clean_html(html) == "Bonjour le monde"


def test_clean_html_removes_scripts():
    html = "<p>Texte</p><script>alert('x')</script><p>suite</p>"
    result = _clean_html(html)
    assert "alert" not in result
    assert "Texte" in result
    assert "suite" in result


def test_clean_html_collapses_whitespace():
    html = "<p>Un   texte   avec   des   espaces</p>"
    assert _clean_html(html) == "Un texte avec des espaces"


def test_chunk_text_basic():
    words = " ".join(f"mot{i}" for i in range(100))
    chunks = _chunk_text(words, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    # Each chunk should have approximately 30 words
    for chunk in chunks:
        assert len(chunk.split()) <= 30


def test_chunk_text_overlap():
    words = " ".join(f"mot{i}" for i in range(60))
    chunks = _chunk_text(words, chunk_size=30, overlap=10)
    assert len(chunks) >= 2
    # Check that chunks overlap: last words of chunk 0 should appear at start of chunk 1
    words_chunk0 = chunks[0].split()
    words_chunk1 = chunks[1].split()
    # The overlap means the last 10 words of chunk 0 should be the first 10 of chunk 1
    assert words_chunk0[-10:] == words_chunk1[:10]


def test_chunk_text_empty():
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


def test_chunk_text_short():
    """Text shorter than chunk_size should produce a single chunk."""
    text = "Un petit texte"
    chunks = _chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_is_content_article_redirect():
    entry = MagicMock()
    entry.is_redirect = True
    entry.path = "A/Test"
    entry.title = "Test"
    assert _is_content_article(entry) is False


def test_is_content_article_category():
    entry = MagicMock()
    entry.is_redirect = False
    entry.path = "A/Cat"
    entry.title = "Catégorie:Sciences"
    assert _is_content_article(entry) is False


def test_is_content_article_valid():
    entry = MagicMock()
    entry.is_redirect = False
    entry.path = "A/Paris"
    entry.title = "Paris"
    assert _is_content_article(entry) is True


@patch("meshwiki.wikipedia_indexer._load_config")
@patch("meshwiki.wikipedia_indexer.chromadb")
def test_index_zim_processes_articles(mock_chromadb, mock_config):
    """Integration-style test with mocked ZIM archive."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model", "chunk_size": 50, "chunk_overlap": 10},
        "vectordb": {"path": "./test_db"},
    }

    # Mock ChromaDB
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    # Mock SentenceTransformer
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])

    # Mock Archive
    mock_entry = MagicMock()
    mock_entry.is_redirect = False
    mock_entry.path = "A/Test"
    mock_entry.title = "Test Article"
    mock_item = MagicMock()
    mock_item.content = b"<p>" + b"This is a test article with enough content. " * 20 + b"</p>"
    mock_entry.get_item.return_value = mock_item

    mock_archive = MagicMock()
    mock_archive.entry_count = 1
    mock_archive._get_entry_by_id.return_value = mock_entry

    with patch("libzim.reader.Archive", return_value=mock_archive), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        from meshwiki.wikipedia_indexer import index_zim
        stats = index_zim(Path("test.zim"), "test_collection")

    assert stats["article_count"] == 1
    assert stats["chunk_count"] >= 1
    mock_collection.add.assert_called()


@patch("meshwiki.wikipedia_indexer.CHECKPOINT_FILE")
def test_load_checkpoint_valid(mock_cp_file):
    """A matching checkpoint returns (last_entry_index, article_count, chunk_count)."""
    mock_cp_file.exists.return_value = True
    mock_cp_file.read_text.return_value = json.dumps({
        "collection_name": "wiki",
        "zim_filename": "test.zim",
        "last_entry_index": 500,
        "article_count": 100,
        "chunk_count": 300,
    })
    result = _load_checkpoint("wiki", Path("some/dir/test.zim"))
    assert result == (500, 100, 300)


@patch("meshwiki.wikipedia_indexer.CHECKPOINT_FILE")
def test_load_checkpoint_mismatch(mock_cp_file):
    """A checkpoint for a different collection returns None."""
    mock_cp_file.exists.return_value = True
    mock_cp_file.read_text.return_value = json.dumps({
        "collection_name": "other",
        "zim_filename": "test.zim",
        "last_entry_index": 500,
        "article_count": 100,
        "chunk_count": 300,
    })
    assert _load_checkpoint("wiki", Path("test.zim")) is None


@patch("meshwiki.wikipedia_indexer.CHECKPOINT_FILE")
def test_load_checkpoint_no_file(mock_cp_file):
    """No checkpoint file returns None."""
    mock_cp_file.exists.return_value = False
    assert _load_checkpoint("wiki", Path("test.zim")) is None


@patch("meshwiki.wikipedia_indexer._load_config")
@patch("meshwiki.wikipedia_indexer.chromadb")
@patch("meshwiki.wikipedia_indexer._load_checkpoint")
@patch("meshwiki.wikipedia_indexer._save_checkpoint")
@patch("meshwiki.wikipedia_indexer.CHECKPOINT_FILE")
def test_index_zim_resumes_from_checkpoint(
    mock_cp_file, mock_save_cp, mock_load_cp, mock_chromadb, mock_config
):
    """When a checkpoint exists, the collection is NOT deleted and the loop starts at the right index."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model", "chunk_size": 50, "chunk_overlap": 10},
        "vectordb": {"path": "./test_db"},
    }

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])

    # Checkpoint says we already processed entries 0-4 (5 entries, 2 articles, 6 chunks)
    mock_load_cp.return_value = (4, 2, 6)

    # Archive has 6 entries total; entry 5 is a valid article
    mock_entry = MagicMock()
    mock_entry.is_redirect = False
    mock_entry.path = "A/Resume"
    mock_entry.title = "Resume Article"
    mock_item = MagicMock()
    mock_item.content = b"<p>" + b"Content word for testing purposes here. " * 20 + b"</p>"
    mock_entry.get_item.return_value = mock_item

    mock_archive = MagicMock()
    mock_archive.entry_count = 6
    mock_archive._get_entry_by_id.return_value = mock_entry

    mock_cp_file.exists.return_value = True

    with patch("libzim.reader.Archive", return_value=mock_archive), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        from meshwiki.wikipedia_indexer import index_zim
        stats = index_zim(Path("test.zim"), "test_col")

    # Collection should NOT have been deleted (no delete_collection call)
    mock_client.delete_collection.assert_not_called()

    # Archive._get_entry_by_id should only be called for index 5 (resume at 4+1=5)
    mock_archive._get_entry_by_id.assert_called_once_with(5)

    # Stats should include the resumed counts + the new article
    assert stats["article_count"] == 3  # 2 from checkpoint + 1 new
    assert stats["chunk_count"] >= 7  # 6 from checkpoint + at least 1 new


@patch("meshwiki.wikipedia_indexer._load_config")
@patch("meshwiki.wikipedia_indexer.chromadb")
@patch("meshwiki.wikipedia_indexer._load_checkpoint")
@patch("meshwiki.wikipedia_indexer.CHECKPOINT_FILE")
def test_index_zim_deletes_checkpoint_on_completion(
    mock_cp_file, mock_load_cp, mock_chromadb, mock_config
):
    """Checkpoint file is deleted when indexation completes."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model", "chunk_size": 50, "chunk_overlap": 10},
        "vectordb": {"path": "./test_db"},
    }

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1] * 384])

    # No checkpoint — fresh indexation
    mock_load_cp.return_value = None

    mock_archive = MagicMock()
    mock_archive.entry_count = 0  # No entries to process

    mock_cp_file.exists.return_value = True

    with patch("libzim.reader.Archive", return_value=mock_archive), \
         patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        from meshwiki.wikipedia_indexer import index_zim
        index_zim(Path("test.zim"), "test_col")

    mock_cp_file.unlink.assert_called_once()


def test_get_indexing_eta_default_is_none():
    """get_indexing_eta() returns None when no indexation is in progress."""
    _set_indexing_eta(None)
    assert get_indexing_eta() is None


def test_set_indexing_eta_updates_value():
    """_set_indexing_eta() stores the ETA and get_indexing_eta() retrieves it."""
    eta = datetime.now() + timedelta(hours=1)
    _set_indexing_eta(eta)
    assert get_indexing_eta() == eta
    # Cleanup
    _set_indexing_eta(None)


def test_set_indexing_eta_clear():
    """Setting ETA to None clears it."""
    _set_indexing_eta(datetime.now() + timedelta(minutes=30))
    assert get_indexing_eta() is not None
    _set_indexing_eta(None)
    assert get_indexing_eta() is None


@patch("meshwiki.wikipedia_indexer.subprocess.run")
def test_gpu_lock_clocks_success(mock_run):
    """Lock returns True when nvidia-smi succeeds."""
    assert _gpu_lock_clocks() is True
    assert mock_run.call_count == 2
    # First call: persistence mode
    assert mock_run.call_args_list[0][0][0] == ["nvidia-smi", "-pm", "1"]
    # Second call: lock clocks
    assert "--lock-gpu-clocks=300,9999" in mock_run.call_args_list[1][0][0]


@patch("meshwiki.wikipedia_indexer.subprocess.run", side_effect=FileNotFoundError)
def test_gpu_lock_clocks_no_nvidia_smi(mock_run):
    """Lock returns False when nvidia-smi is not found."""
    assert _gpu_lock_clocks() is False


@patch("meshwiki.wikipedia_indexer.subprocess.run")
def test_gpu_unlock_clocks(mock_run):
    """Unlock calls nvidia-smi to reset clocks."""
    _gpu_unlock_clocks()
    assert mock_run.call_count == 2
    assert "--reset-gpu-clocks" in mock_run.call_args_list[0][0][0]


@patch("meshwiki.wikipedia_indexer.subprocess.run", side_effect=FileNotFoundError)
def test_gpu_unlock_clocks_no_nvidia_smi(mock_run):
    """Unlock does not raise when nvidia-smi is not found."""
    _gpu_unlock_clocks()  # Should not raise
