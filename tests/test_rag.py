"""Tests for meshwiki.rag — RAG pipeline (semantic search + LLM)."""

from unittest.mock import MagicMock, patch

import meshwiki.rag as rag_module


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
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
@patch("meshwiki.config.load_config")
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
@patch("meshwiki.config.load_config")
def test_query_db_unavailable(mock_config):
    mock_config.side_effect = Exception("DB not found")

    result = rag_module.query("Test ?")

    assert "indisponible" in result


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
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
@patch("meshwiki.config.load_config")
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
        "distances": [[0.65, 0.80]],
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
    """query_without_context() calls LLM with system prompt and appends suffix."""
    mock_llm.generate.return_value = "Réponse sans Wikipedia"

    result = rag_module.query_without_context("Capitale de la France ?", "14:30")

    assert result == "Réponse sans Wikipedia [Sans source Wikipedia — base disponible dans 14:30]"
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "initialisation" in system_prompt
    assert "Capitale de la France ?" in user_prompt


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_uses_results(mock_kiwix, mock_llm):
    """query_with_kiwix_context() passes Kiwix results as context and appends suffix."""
    mock_kiwix.search.return_value = [
        {"title": "Paris", "content": "Paris est la capitale de la France."},
    ]
    mock_llm.generate.return_value = "Paris est la capitale."

    result = rag_module.query_with_kiwix_context("Capitale de la France ?", "environ 1h30")

    assert result == "Paris est la capitale. [Recherche Kiwix — base optimisée dans environ 1h30]"

    # First call is the cascade step 1 (long extracts)
    call_args = mock_llm.generate.call_args_list[0]
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "[Paris]" in user_prompt
    assert "Paris est la capitale de la France" in user_prompt


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_fallback_no_results(mock_kiwix, mock_llm):
    """When Kiwix returns no results, falls back to query_without_context with suffix."""
    mock_kiwix.search.return_value = []
    mock_llm.generate.return_value = "Réponse sans contexte"

    result = rag_module.query_with_kiwix_context("Question obscure ?", "environ 2h")

    assert result == "Réponse sans contexte [Sans source Wikipedia — base disponible dans environ 2h]"

    # Should have used the NO_INDEX prompt (fallback), not the Kiwix one
    system_prompt = mock_llm.generate.call_args[0][0]
    assert "initialisation" in system_prompt


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_is_available_false_force_unavailable(mock_config, mock_chromadb):
    """is_available() returns False when set_force_unavailable(True) is active."""
    mock_config.return_value = {
        "embeddings": {"embedding_dim": 1024},
        "vectordb": {"path": "./test_db"},
    }
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.peek.return_value = {"embeddings": [[0.1] * 1024]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    rag_module.set_force_unavailable(True)
    try:
        with patch("meshwiki.rag.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            assert rag_module.is_available() is False
    finally:
        rag_module.set_force_unavailable(False)


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_is_available_true(mock_config, mock_chromadb):
    """is_available() returns True when the Wikipedia collection exists with correct dimensions."""
    mock_config.return_value = {
        "embeddings": {"embedding_dim": 1024},
        "vectordb": {"path": "./test_db"},
    }
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.peek.return_value = {"embeddings": [[0.1] * 1024]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert rag_module.is_available() is True


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_is_available_false_no_collection(mock_config, mock_chromadb):
    """is_available() returns False when the collection doesn't exist."""
    mock_config.return_value = {
        "embeddings": {"embedding_dim": 1024},
        "vectordb": {"path": "./test_db"},
    }
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Collection not found")
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert rag_module.is_available() is False


@patch("meshwiki.config.load_config")
def test_is_available_false_no_db(mock_config):
    """is_available() returns False when the database path doesn't exist."""
    mock_config.return_value = {
        "embeddings": {"embedding_dim": 1024},
        "vectordb": {"path": "./nonexistent_db"},
    }

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        assert rag_module.is_available() is False


@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_is_available_false_wrong_dimension(mock_config, mock_chromadb):
    """is_available() returns False when embedding dimensions don't match."""
    mock_config.return_value = {
        "embeddings": {"embedding_dim": 1024},
        "vectordb": {"path": "./test_db"},
    }
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.peek.return_value = {"embeddings": [[0.1] * 384]}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch("meshwiki.rag.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert rag_module.is_available() is False


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent(mock_kiwix, mock_llm):
    """query_with_kiwix_context_permanent() uses Kiwix results and appends suffix."""
    mock_kiwix.search.return_value = [
        {"title": "Paris", "content": "Paris est la capitale de la France."},
    ]
    mock_llm.generate.return_value = "Paris est la capitale."

    result = rag_module.query_with_kiwix_context_permanent("Capitale de la France ?")

    assert result == "Paris est la capitale. [Recherche textuelle Kiwix]"

    call_args = mock_llm.generate.call_args_list[0]
    user_prompt = call_args[0][1]
    assert "[Paris]" in user_prompt


@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent_no_results(mock_kiwix):
    """query_with_kiwix_context_permanent() returns a message when no Kiwix results."""
    mock_kiwix.search.return_value = []

    result = rag_module.query_with_kiwix_context_permanent("Question obscure ?")

    assert "Aucun résultat" in result


@patch("meshwiki.rag.llm")
def test_query_without_data_calls_llm(mock_llm):
    """query_without_data() calls LLM and appends [Sans source Wikipedia] suffix."""
    mock_llm.generate.return_value = "Paris est la capitale."

    result = rag_module.query_without_data("Capitale de la France ?")

    assert result == "Paris est la capitale. [Sans source Wikipedia]"
    mock_llm.generate.assert_called_once()

    call_args = mock_llm.generate.call_args
    system_prompt = call_args[0][0]
    user_prompt = call_args[0][1]
    assert "connaissances générales" in system_prompt
    assert "Capitale de la France ?" in user_prompt


# --- Cascade tests ---


def test_is_no_answer_detects_patterns():
    """_is_no_answer detects 'je n'ai pas trouvé' and 'je ne sais pas'."""
    assert rag_module._is_no_answer("Je n'ai pas trouvé cette information.") is True
    assert rag_module._is_no_answer("Je ne sais pas.") is True
    assert rag_module._is_no_answer("JE N'AI PAS TROUVÉ cette info.") is True
    assert rag_module._is_no_answer("Paris est la capitale de la France.") is False
    assert rag_module._is_no_answer("") is False


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_finds_in_long_extracts(mock_kiwix, mock_llm):
    """Cascade step 1: LLM answers from long extracts, suffix appended."""
    mock_kiwix.search.return_value = [
        {"title": "Piton des Neiges", "content": "Le Piton des Neiges culmine à 3070m."},
    ]
    mock_llm.generate.return_value = "Le Piton des Neiges culmine à 3070 m."

    result = rag_module.query_with_kiwix_context_permanent("Altitude du Piton des Neiges ?")

    assert result == "Le Piton des Neiges culmine à 3070 m. [Recherche textuelle Kiwix]"
    mock_kiwix.get_article_content.assert_not_called()
    mock_kiwix.search.assert_called_once_with("Altitude du Piton des Neiges ?", max_chars_per_result=4000)


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_falls_back_to_full_article(mock_kiwix, mock_llm):
    """Cascade step 2: LLM fails on extracts, succeeds on full article with suffix."""
    mock_kiwix.search.return_value = [
        {"title": "Piton des Neiges", "content": "Le Piton des Neiges est un volcan."},
    ]
    mock_kiwix.get_article_content.return_value = (
        "Le Piton des Neiges est un volcan situé à La Réunion. "
        "Il culmine à 3 070 mètres d'altitude."
    )

    # First call (long extracts): no answer. Second call (full article): answer found.
    mock_llm.generate.side_effect = [
        "Je n'ai pas trouvé cette information.",
        "Le Piton des Neiges culmine à 3 070 m.",
    ]

    result = rag_module.query_with_kiwix_context_permanent("Altitude du Piton des Neiges ?")

    assert result == "Le Piton des Neiges culmine à 3 070 m. [Recherche textuelle Kiwix]"
    mock_kiwix.get_article_content.assert_called_once_with("Piton des Neiges")
    assert mock_llm.generate.call_count == 2


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_tries_multiple_articles(mock_kiwix, mock_llm):
    """Cascade steps 2-3: first full article fails, second succeeds with suffix."""
    mock_kiwix.search.return_value = [
        {"title": "Volcanisme", "content": "Le volcanisme est un phénomène..."},
        {"title": "Piton des Neiges", "content": "Le Piton des Neiges est un volcan..."},
        {"title": "La Réunion", "content": "La Réunion est une île..."},
    ]
    mock_kiwix.get_article_content.side_effect = [
        "Le volcanisme est un phénomène géologique...",
        "Le Piton des Neiges culmine à 3 070 m d'altitude.",
    ]

    # extracts fail, article 1 fails, article 2 succeeds
    mock_llm.generate.side_effect = [
        "Je n'ai pas trouvé cette information.",
        "Je ne sais pas.",
        "Le Piton des Neiges culmine à 3 070 m.",
    ]

    result = rag_module.query_with_kiwix_context_permanent("Altitude du Piton des Neiges ?")

    assert result == "Le Piton des Neiges culmine à 3 070 m. [Recherche textuelle Kiwix]"
    assert mock_kiwix.get_article_content.call_count == 2
    assert mock_llm.generate.call_count == 3


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_all_fail_uses_llm_opinion(mock_kiwix, mock_llm):
    """Cascade final step: all attempts fail, LLM opinion prefixed programmatically."""
    mock_kiwix.search.return_value = [
        {"title": "Article1", "content": "Contenu 1"},
    ]
    mock_kiwix.get_article_content.return_value = "Contenu complet article 1"

    # extracts fail, full article fails, then LLM opinion
    mock_llm.generate.side_effect = [
        "Je n'ai pas trouvé cette information.",
        "Je ne sais pas.",
        "La réponse est 42.",
    ]

    result = rag_module.query_with_kiwix_context_permanent("Question très obscure ?")

    assert result == "Réponse non trouvée dans la base. Mon avis : La réponse est 42."
    # Last call should use SYSTEM_PROMPT_LLM_OPINION
    last_call = mock_llm.generate.call_args_list[-1]
    system_prompt = last_call[0][0]
    assert "connaissances générales" in system_prompt


@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_no_results_falls_back(mock_kiwix, mock_llm):
    """When Kiwix returns 0 results, cascade returns None and caller falls back."""
    mock_kiwix.search.return_value = []

    result = rag_module.query_with_kiwix_context_permanent("Question obscure ?")

    assert "Aucun résultat" in result
    mock_llm.generate.assert_not_called()
