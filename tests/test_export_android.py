"""Tests for meshwiki.export_android — ChromaDB to SQLite export."""

import sqlite3
import struct
from unittest.mock import MagicMock, patch

import pytest

from meshwiki.export_android import (
    EMBEDDING_DIM,
    _pack_embedding,
    _strip_title_prefix,
    export,
)


class TestPackEmbedding:
    def test_output_length(self):
        emb = [0.0] * EMBEDDING_DIM
        blob = _pack_embedding(emb)
        assert len(blob) == EMBEDDING_DIM * 4  # 1024 × 4 = 4096

    def test_little_endian_format(self):
        emb = [1.0, 2.0] + [0.0] * (EMBEDDING_DIM - 2)
        blob = _pack_embedding(emb)
        first, second = struct.unpack_from("<2f", blob)
        assert first == pytest.approx(1.0)
        assert second == pytest.approx(2.0)

    def test_roundtrip(self):
        emb = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
        blob = _pack_embedding(emb)
        restored = list(struct.unpack(f"<{EMBEDDING_DIM}f", blob))
        for orig, rest in zip(emb, restored):
            assert orig == pytest.approx(rest, abs=1e-6)


class TestStripTitlePrefix:
    def test_removes_prefix(self):
        doc = "Paris \u2014 Paris est la capitale de la France."
        assert _strip_title_prefix(doc, "Paris") == "Paris est la capitale de la France."

    def test_no_prefix(self):
        doc = "Texte sans préfixe titre."
        assert _strip_title_prefix(doc, "Autre") == "Texte sans préfixe titre."

    def test_title_with_special_chars(self):
        doc = "L'île de la Réunion \u2014 L'île est un département français."
        assert _strip_title_prefix(doc, "L'île de la Réunion") == "L'île est un département français."


class TestExportIntegration:
    """Integration test: mock ChromaDB → real SQLite → verify schema and data."""

    @pytest.fixture
    def mock_chromadb(self):
        """Create a mock ChromaDB client with test data."""
        embeddings = [[float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)] for _ in range(3)]

        collection = MagicMock()
        collection.count.return_value = 3
        collection.get.return_value = {
            "documents": [
                "Paris \u2014 Paris est la capitale de la France.",
                "Paris \u2014 Paris compte plus de 2 millions d'habitants.",
                "Lyon \u2014 Lyon est une ville du sud-est.",
            ],
            "metadatas": [
                {"title": "Paris", "chunk_index": 0, "total_chunks": 2},
                {"title": "Paris", "chunk_index": 1, "total_chunks": 2},
                {"title": "Lyon", "chunk_index": 0, "total_chunks": 1},
            ],
            "embeddings": embeddings,
        }

        client = MagicMock()
        client.get_collection.return_value = collection
        return client

    @patch("meshwiki.export_android.get_active_db_path", return_value="data/chroma_db_a")
    @patch("meshwiki.export_android.config.load_config")
    @patch("meshwiki.export_android.chromadb.PersistentClient")
    def test_full_export(self, mock_client_cls, mock_config, mock_db_path, mock_chromadb, tmp_path):
        mock_client_cls.return_value = mock_chromadb
        mock_config.return_value = {
            "embeddings": {"model": "BAAI/bge-m3"},
            "vectordb": {"path": "data/chroma_db"},
        }

        output = str(tmp_path / "test_android.db")
        stats = export(output)

        assert stats["chunk_count"] == 3
        assert stats["file_size_mb"] > 0

        conn = sqlite3.connect(output)

        # Check chunks table
        rows = conn.execute(
            "SELECT article_title, content, embedding, position FROM chunks ORDER BY id"
        ).fetchall()
        assert len(rows) == 3

        # First chunk: Paris, position 0
        assert rows[0][0] == "Paris"
        assert rows[0][1] == "Paris est la capitale de la France."
        assert len(rows[0][2]) == EMBEDDING_DIM * 4
        assert rows[0][3] == 0

        # Second chunk: Paris, position 1
        assert rows[1][0] == "Paris"
        assert rows[1][3] == 1

        # Third chunk: Lyon
        assert rows[2][0] == "Lyon"
        assert rows[2][1] == "Lyon est une ville du sud-est."

        # Check FTS table
        fts_count = conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]
        assert fts_count == 3

        # Check FTS search works
        fts_results = conn.execute(
            "SELECT article_title FROM chunks_fts WHERE chunks_fts MATCH 'capitale'"
        ).fetchall()
        assert len(fts_results) == 1
        assert fts_results[0][0] == "Paris"

        # Check metadata table
        meta = conn.execute(
            "SELECT value FROM metadata WHERE key = 'embeddings_model'"
        ).fetchone()
        assert meta[0] == "BAAI/bge-m3"

        # Check embedding can be decoded
        blob = rows[0][2]
        decoded = list(struct.unpack(f"<{EMBEDDING_DIM}f", blob))
        assert len(decoded) == EMBEDDING_DIM
        assert decoded[0] == pytest.approx(0.0, abs=1e-6)

        conn.close()
