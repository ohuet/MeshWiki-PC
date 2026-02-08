"""Tests for meshwiki.rag — RAG pipeline (semantic search + LLM)."""

from unittest.mock import MagicMock, patch

import meshwiki.rag as rag_module


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_query_success(mock_config, mock_chromadb, mock_st, mock_llm):
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
    }

    # Mock embedding model
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    # Mock ChromaDB collection
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Paris est la capitale de la France.", "La France est en Europe."]],
        "metadatas": [[{"title": "Paris"}, {"title": "France"}]],
        "distances": [[0.15, 0.25]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    # Mock LLM
    mock_llm.generate.return_value = "Paris est la capitale de la France."

    result = rag_module.query("Quelle est la capitale de la France ?")

    assert result == "Paris est la capitale de la France."
    mock_llm.generate.assert_called_once()

    # Verify prompt structure
    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "encyclopédique" in system_prompt
    assert "Paris" in user_prompt
    assert "Question : Quelle est la capitale de la France ?" in user_prompt


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_query_no_results(mock_config, mock_chromadb, mock_st):
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
    }

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    result = rag_module.query("Une question sans réponse ?")

    assert "Aucun article pertinent" in result


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag._load_config")
def test_query_db_unavailable(mock_config):
    mock_config.side_effect = Exception("DB not found")

    result = rag_module.query("Test ?")

    assert "indisponible" in result


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_query_prompt_contains_context(mock_config, mock_chromadb, mock_st, mock_llm):
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
    }

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Antananarivo est la capitale de Madagascar."]],
        "metadatas": [[{"title": "Madagascar"}]],
        "distances": [[0.12]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_llm.generate.return_value = "Antananarivo"

    rag_module.query("Capitale de Madagascar ?")

    user_prompt = mock_llm.generate.call_args[0][1]
    assert "[Madagascar]" in user_prompt
    assert "Antananarivo" in user_prompt


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_query_filters_distant_results(mock_config, mock_chromadb, mock_st):
    """Results with cosine distance > threshold are filtered out."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
    }

    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Requin cuivre info", "Requin marteau info"]],
        "metadatas": [[{"title": "Requin cuivre"}, {"title": "Requin marteau"}]],
        "distances": [[0.55, 0.70]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    result = rag_module.query("Requin tigre ?")

    assert "Aucun article pertinent" in result


def test_reset_collection_clears_cache():
    """reset_collection() sets _collection to None so next query re-fetches."""
    rag_module._collection = MagicMock()
    assert rag_module._collection is not None

    rag_module.reset_collection()

    assert rag_module._collection is None


@patch("meshwiki.rag.llm")
def test_query_without_context_calls_llm(mock_llm):
    """query_without_context() calls LLM with ETA in system prompt and the question."""
    mock_llm.generate.return_value = "Réponse sans Wikipedia"

    result = rag_module.query_without_context("Capitale de la France ?", "14:30")

    assert result == "Réponse sans Wikipedia"
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "14:30" in system_prompt
    assert "initialisation" in system_prompt
    assert "Capitale de la France ?" in user_prompt
