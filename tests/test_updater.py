"""Tests for meshwiki.wikipedia_updater — download and re-indexation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import meshwiki.wikipedia_updater as updater_module
from meshwiki.wikipedia_updater import WikipediaUpdater, _copy_collection


MOCK_CONFIG = {
    "updater": {
        "kiwix_url": "https://download.kiwix.org/zim/wikipedia/?C=M;O=D",
        "dump_pattern": "wikipedia_fr_all_mini_",
        "temp_dir": "./data/tmp",
        "max_download_retries": 3,
        "download_timeout_hours": 24,
    },
    "vectordb": {"path": "./data/chroma_db"},
    "embeddings": {"model": "test-model", "chunk_size": 500, "chunk_overlap": 50},
}

KIWIX_HTML = """
<html><body>
<a href="wikipedia_fr_all_mini_2025-01.zim">wikipedia_fr_all_mini_2025-01.zim</a>
<a href="wikipedia_fr_all_mini_2024-12.zim">wikipedia_fr_all_mini_2024-12.zim</a>
<a href="wikipedia_en_all_2025-01.zim">wikipedia_en_all_2025-01.zim</a>
</body></html>
"""


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_finds_match(mock_get, mock_config):
    mock_response = MagicMock()
    mock_response.text = KIWIX_HTML
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is not None
    url, filename = result
    assert filename == "wikipedia_fr_all_mini_2025-01.zim"
    assert "wikipedia_fr_all_mini_2025-01.zim" in url


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_no_match(mock_get, mock_config):
    mock_response = MagicMock()
    mock_response.text = "<html><a href='other_file.txt'>nope</a></html>"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is None


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.requests.get")
def test_get_latest_dump_url_network_error(mock_get, mock_config):
    import requests
    mock_get.side_effect = requests.exceptions.ConnectionError()

    updater = WikipediaUpdater()
    result = updater.get_latest_dump_url()

    assert result is None


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch.object(updater_module, "LAST_UPDATE_FILE")
def test_download_dump_skips_if_up_to_date(mock_file, mock_config):
    mock_file.exists.return_value = True
    mock_file.__fspath__ = lambda self: "data/last_update.json"

    updater = WikipediaUpdater()

    with patch.object(updater, "get_latest_dump_url", return_value=("http://example.com/file.zim", "wikipedia_fr_all_mini_2025-01.zim")), \
         patch("builtins.open", mock_open(read_data=json.dumps({"last_filename": "wikipedia_fr_all_mini_2025-01.zim"}))):
        result = updater.download_dump()

    assert result is None


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
def test_download_dump_returns_none_when_no_url(mock_config):
    updater = WikipediaUpdater()

    with patch.object(updater, "get_latest_dump_url", return_value=None):
        result = updater.download_dump()

    assert result is None


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.reset_collection")
@patch("meshwiki.wikipedia_updater.index_zim")
@patch("meshwiki.wikipedia_updater.chromadb")
def test_reindex_success(mock_chromadb, mock_index_zim, mock_reset, mock_config):
    mock_index_zim.return_value = {"article_count": 100, "chunk_count": 500}
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    updater = WikipediaUpdater()
    zim_path = MagicMock(spec=Path)
    zim_path.name = "test.zim"

    with patch("builtins.open", mock_open()):
        result = updater.reindex(zim_path)

    assert result is True
    mock_index_zim.assert_called_once_with(zim_path, collection_name="wikipedia_new")
    # Temp collection is renamed to active (not copied)
    mock_client.get_collection.assert_called_with("wikipedia_new")
    mock_collection.modify.assert_called_once_with(name="wikipedia")
    # RAG cache is invalidated after swap
    mock_reset.assert_called_once()


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.wikipedia_updater.index_zim")
@patch("meshwiki.wikipedia_updater.chromadb")
def test_reindex_failure_preserves_old_index(mock_chromadb, mock_index_zim, mock_config):
    mock_index_zim.side_effect = Exception("Indexation error")
    mock_client = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client

    updater = WikipediaUpdater()
    result = updater.reindex(MagicMock(spec=Path))

    assert result is False
    # Temp collection is NOT deleted on failure (checkpoint + collection
    # must stay in sync for safe resume after interruption)
    mock_client.delete_collection.assert_not_called()


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
def test_cleanup_deletes_zim_file(mock_config, tmp_path):
    zim_file = tmp_path / "test.zim"
    zim_file.write_bytes(b"x" * 1024)

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(tmp_path)
    updater.cleanup(zim_file)

    assert not zim_file.exists()


@patch.object(updater_module, "_load_config", return_value=MOCK_CONFIG)
def test_cleanup_removes_temp_files(mock_config, tmp_path):
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    (temp_dir / "partial.zim.part").write_bytes(b"partial")

    updater = WikipediaUpdater()
    updater.updater_config["temp_dir"] = str(temp_dir)
    updater.cleanup(MagicMock(exists=lambda: False))

    assert not list(temp_dir.iterdir())


def test_copy_collection_copies_all_data():
    """_copy_collection transfers all documents from source to dest in batches."""
    mock_client = MagicMock()

    # Source collection returns 2 batches then empty
    mock_source = MagicMock()
    mock_source.get.side_effect = [
        {
            "ids": ["id1", "id2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"title": "A"}, {"title": "B"}],
        },
        {
            "ids": ["id3"],
            "embeddings": [[0.5, 0.6]],
            "documents": ["doc3"],
            "metadatas": [{"title": "C"}],
        },
        {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
        },
    ]
    mock_dest = MagicMock()

    mock_client.get_collection.return_value = mock_source
    mock_client.get_or_create_collection.return_value = mock_dest

    _copy_collection(mock_client, "source", "dest", batch_size=2)

    assert mock_dest.add.call_count == 2
    # First batch
    first_call = mock_dest.add.call_args_list[0]
    assert first_call[1]["ids"] == ["id1", "id2"]
    # Second batch
    second_call = mock_dest.add.call_args_list[1]
    assert second_call[1]["ids"] == ["id3"]


def test_copy_collection_handles_empty_source():
    """_copy_collection does nothing when source is empty."""
    mock_client = MagicMock()

    mock_source = MagicMock()
    mock_source.get.return_value = {
        "ids": [],
        "embeddings": [],
        "documents": [],
        "metadatas": [],
    }
    mock_dest = MagicMock()

    mock_client.get_collection.return_value = mock_source
    mock_client.get_or_create_collection.return_value = mock_dest

    _copy_collection(mock_client, "source", "dest")

    mock_dest.add.assert_not_called()
