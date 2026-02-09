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


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_uses_results(mock_kiwix, mock_llm):
    """query_with_kiwix_context() passes Kiwix results as context to the LLM."""
    mock_kiwix.search.return_value = [
        {"title": "Paris", "content": "Paris est la capitale de la France."},
    ]
    mock_llm.generate.return_value = "Paris est la capitale."

    result = rag_module.query_with_kiwix_context("Capitale de la France ?", "environ 1h30")

    assert result == "Paris est la capitale."
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "Kiwix" in system_prompt
    assert "environ 1h30" in system_prompt
    assert "[Paris]" in user_prompt
    assert "Paris est la capitale de la France" in user_prompt


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_fallback_no_results(mock_kiwix, mock_llm):
    """When Kiwix returns no results, falls back to query_without_context."""
    mock_kiwix.search.return_value = []
    mock_llm.generate.return_value = "Réponse sans contexte"

    result = rag_module.query_with_kiwix_context("Question obscure ?", "environ 2h")

    assert result == "Réponse sans contexte"
    mock_llm.generate.assert_called_once()

    # Should have used the NO_INDEX prompt (fallback), not the Kiwix one
    system_prompt = mock_llm.generate.call_args[0][0]
    assert "initialisation" in system_prompt
    assert "environ 2h" in system_prompt


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_is_available_true(mock_config, mock_chromadb):
    """is_available() returns True when the Wikipedia collection exists and has documents."""
    mock_config.return_value = {"vectordb": {"path": "./test_db"}}
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert rag_module.is_available() is True


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.rag._load_config")
def test_is_available_false_no_collection(mock_config, mock_chromadb):
    """is_available() returns False when the collection doesn't exist."""
    mock_config.return_value = {"vectordb": {"path": "./test_db"}}
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert rag_module.is_available() is False


@patch("meshwiki.rag._load_config")
def test_is_available_false_no_db(mock_config):
    """is_available() returns False when the database path doesn't exist."""
    mock_config.return_value = {"vectordb": {"path": "./nonexistent_db"}}

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        assert rag_module.is_available() is False


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent(mock_kiwix, mock_llm):
    """query_with_kiwix_context_permanent() uses Kiwix results without ETA."""
    mock_kiwix.search.return_value = [
        {"title": "Paris", "content": "Paris est la capitale de la France."},
    ]
    mock_llm.generate.return_value = "Paris est la capitale."

    result = rag_module.query_with_kiwix_context_permanent("Capitale de la France ?")

    assert result == "Paris est la capitale."
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "Recherche textuelle Kiwix" in system_prompt
    assert "eta" not in system_prompt.lower()
    assert "[Paris]" in user_prompt


@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent_no_results(mock_kiwix):
    """query_with_kiwix_context_permanent() returns a message when no Kiwix results."""
    mock_kiwix.search.return_value = []

    result = rag_module.query_with_kiwix_context_permanent("Question obscure ?")

    assert "Aucun résultat" in result
