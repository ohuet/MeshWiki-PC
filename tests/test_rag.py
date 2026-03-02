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
        "rag": {"use_llm_for_kiwix_keywords": False},
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
        "rag": {"use_llm_for_kiwix_keywords": False},
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
        "rag": {"use_llm_for_kiwix_keywords": False},
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
        "rag": {"use_llm_for_kiwix_keywords": False},
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_uses_results(mock_kiwix, mock_llm, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_fallback_no_results(mock_kiwix, mock_llm, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent(mock_kiwix, mock_llm, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.kiwix_search")
def test_query_with_kiwix_context_permanent_no_results(mock_kiwix, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_finds_in_long_extracts(mock_kiwix, mock_llm, mock_config):
    """Cascade step 1: LLM answers from long extracts, suffix appended."""
    mock_kiwix.search.return_value = [
        {"title": "Piton des Neiges", "content": "Le Piton des Neiges culmine à 3070m."},
    ]
    mock_llm.generate.return_value = "Le Piton des Neiges culmine à 3070 m."

    result = rag_module.query_with_kiwix_context_permanent("Altitude du Piton des Neiges ?")

    assert result == "Le Piton des Neiges culmine à 3070 m. [Recherche textuelle Kiwix]"
    mock_kiwix.get_article_content.assert_not_called()
    mock_kiwix.search.assert_called_once_with("Altitude du Piton des Neiges ?", max_chars_per_result=4000)


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_falls_back_to_full_article(mock_kiwix, mock_llm, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_tries_multiple_articles(mock_kiwix, mock_llm, mock_config):
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


@patch("meshwiki.config.load_config", return_value={"rag": {"use_llm_for_kiwix_keywords": False}})
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_all_fail_uses_llm_opinion(mock_kiwix, mock_llm, mock_config):
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
    mock_kiwix.has_zim_paths.return_value = False

    result = rag_module.query_with_kiwix_context_permanent("Question obscure ?")

    assert "Aucun résultat" in result
    mock_llm.generate.assert_not_called()


# --- LLM search expressions tests ---


@patch("meshwiki.rag.llm")
def test_llm_search_expressions_parses_lines(mock_llm):
    """_llm_search_expressions parses LLM response into a list of expressions."""
    mock_llm.generate.return_value = "Piton de la Fournaise\nVolcanisme Réunion\nÉruption volcanique"

    result = rag_module._llm_search_expressions("Quand a eu lieu la dernière éruption ?")

    assert result == ["Piton de la Fournaise", "Volcanisme Réunion", "Éruption volcanique"]
    mock_llm.generate.assert_called_once()
    # Verify max_tokens=100 was passed
    call_kwargs = mock_llm.generate.call_args[1]
    assert call_kwargs["max_tokens"] == 100


@patch("meshwiki.rag.llm")
def test_llm_search_expressions_limits_to_4(mock_llm):
    """_llm_search_expressions limits to 4 expressions max."""
    mock_llm.generate.return_value = "Expr1\nExpr2\nExpr3\nExpr4\nExpr5\nExpr6"

    result = rag_module._llm_search_expressions("Question ?")

    assert len(result) == 4
    assert result == ["Expr1", "Expr2", "Expr3", "Expr4"]


@patch("meshwiki.rag.llm")
def test_llm_search_expressions_strips_numbering(mock_llm):
    """_llm_search_expressions removes '1. ', '- ', '• ' prefixes."""
    mock_llm.generate.return_value = "1. Premier résultat\n2. Deuxième résultat\n- Troisième\n• Quatrième"

    result = rag_module._llm_search_expressions("Question ?")

    assert result == ["Premier résultat", "Deuxième résultat", "Troisième", "Quatrième"]


@patch("meshwiki.rag.llm")
def test_llm_search_expressions_returns_empty_on_error(mock_llm):
    """_llm_search_expressions returns [] when LLM returns an error message."""
    mock_llm.generate.return_value = "Service LLM indisponible."

    result = rag_module._llm_search_expressions("Question ?")

    assert result == []


# --- LLM-assisted cascade tests ---


@patch("meshwiki.config.load_config")
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_llm_phase1_finds_answer(mock_kiwix, mock_llm, mock_config):
    """Phase 1 finds an answer from the best article per expression."""
    mock_config.return_value = {"rag": {"use_llm_for_kiwix_keywords": True}}
    mock_kiwix.has_zim_paths.return_value = True

    # LLM generates 2 expressions
    # First call: generate expressions; second call: phase 1 cascade (long extracts)
    mock_llm.generate.side_effect = [
        "Piton de la Fournaise\nVolcanisme Réunion",   # expressions
        "Le Piton de la Fournaise est entré en éruption en 2023.",  # phase 1 answer
    ]

    mock_kiwix.search.side_effect = [
        [{"title": "Piton de la Fournaise", "content": "Le Piton de la Fournaise..."}],
        [{"title": "Volcanisme", "content": "Le volcanisme à La Réunion..."}],
    ]

    result = rag_module._kiwix_cascade("Dernière éruption ?", "SYSTEM", "[suffix]")

    assert "2023" in result
    assert "[suffix]" in result


@patch("meshwiki.config.load_config")
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_llm_phase2_after_phase1_fails(mock_kiwix, mock_llm, mock_config):
    """Phase 1 fails, Phase 2 finds an answer from remaining articles."""
    mock_config.return_value = {"rag": {"use_llm_for_kiwix_keywords": True}}
    mock_kiwix.has_zim_paths.return_value = True

    mock_kiwix.search.side_effect = [
        [
            {"title": "Piton de la Fournaise", "content": "Article principal..."},
            {"title": "Éruptions historiques", "content": "Liste des éruptions..."},
        ],
        [{"title": "Volcanisme", "content": "Le volcanisme..."}],
    ]
    mock_kiwix.get_article_content.return_value = None  # No full articles → skips full-article steps

    # Call sequence: expressions, phase1 long extracts (fail), phase2 long extracts (succeed)
    # Full article steps are skipped because get_article_content returns None
    mock_llm.generate.side_effect = [
        "Piton de la Fournaise\nVolcanisme Réunion",  # expressions
        "Je ne sais pas.",                             # phase 1 long extracts fail
        "La dernière éruption a eu lieu en 2023.",     # phase 2 (Éruptions historiques) succeed
    ]

    result = rag_module._kiwix_cascade("Dernière éruption ?", "SYSTEM", "[suffix]")

    assert "2023" in result
    assert "[suffix]" in result


@patch("meshwiki.config.load_config")
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_llm_deduplicates_across_expressions(mock_kiwix, mock_llm, mock_config):
    """Articles seen in phase 1 are not repeated in phase 2."""
    mock_config.return_value = {"rag": {"use_llm_for_kiwix_keywords": True}}
    mock_kiwix.has_zim_paths.return_value = True

    # Both expressions return the same article "Piton" as first result
    mock_kiwix.search.side_effect = [
        [{"title": "Piton", "content": "Content A"}, {"title": "Unique1", "content": "Content B"}],
        [{"title": "Piton", "content": "Content A"}, {"title": "Unique2", "content": "Content C"}],
    ]
    mock_kiwix.get_article_content.return_value = None  # full-article steps skipped

    # Phase 1 picks: Piton (from expr1), Unique2 (from expr2, Piton already seen)
    # Phase 2 picks: Unique1 (remaining from expr1, not yet seen)
    mock_llm.generate.side_effect = [
        "Expr1\nExpr2",       # expressions
        "Je ne sais pas.",    # phase 1 (Piton + Unique2) long extracts fail
        "Found it!",          # phase 2 (Unique1) long extracts succeed
    ]

    result = rag_module._kiwix_cascade("Question ?", "SYSTEM", "[suffix]")

    assert "Found it!" in result
    assert "[suffix]" in result


@patch("meshwiki.config.load_config")
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
def test_kiwix_cascade_falls_back_when_disabled(mock_kiwix, mock_llm, mock_config):
    """When use_llm_for_kiwix_keywords is false, uses original behavior."""
    mock_config.return_value = {"rag": {"use_llm_for_kiwix_keywords": False}}

    mock_kiwix.search.return_value = [
        {"title": "Paris", "content": "Paris est la capitale."},
    ]
    mock_llm.generate.return_value = "Paris est la capitale de la France."

    result = rag_module._kiwix_cascade("Capitale ?", "SYSTEM", "[suffix]")

    assert "Paris est la capitale" in result
    assert "[suffix]" in result
    # Should NOT have called _llm_search_expressions (no max_tokens=100 call)
    for call in mock_llm.generate.call_args_list:
        kwargs = call[1] if call[1] else {}
        assert kwargs.get("max_tokens") is None


# --- query() with Kiwix cascade fallback tests ---


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_query_falls_back_to_kiwix_when_chromadb_fails(
    mock_config, mock_chromadb, mock_st, mock_kiwix, mock_llm
):
    """When ChromaDB has no relevant chunks, query() falls back to Kiwix cascade."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
        "rag": {"use_llm_for_kiwix_keywords": True},
    }
    mock_kiwix.has_zim_paths.return_value = True

    # ChromaDB setup — returns results but all too distant
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Some irrelevant content"]],
        "metadatas": [[{"title": "Irrelevant"}]],
        "distances": [[0.85]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    # Kiwix cascade: LLM-assisted finds answer
    mock_kiwix.search.return_value = [
        {"title": "Piton", "content": "Le Piton culmine à 3070m."},
    ]

    mock_llm.generate.side_effect = [
        "Piton des Neiges",               # LLM expressions
        "Le Piton culmine à 3070m.",       # phase 1 answer
    ]

    result = rag_module.query("Altitude du Piton ?")

    assert "3070" in result
    assert "[Recherche textuelle Kiwix]" in result
    # ChromaDB was queried first
    mock_collection.query.assert_called_once()


@patch.object(rag_module, "_collection", None)
@patch.object(rag_module, "_model", None)
@patch("meshwiki.rag.llm")
@patch("meshwiki.rag.kiwix_search")
@patch("meshwiki.rag.SentenceTransformer")
@patch("meshwiki.rag.chromadb")
@patch("meshwiki.config.load_config")
def test_query_chromadb_first_then_kiwix_cascade(
    mock_config, mock_chromadb, mock_st, mock_kiwix, mock_llm
):
    """ChromaDB is tried first; when all chunks fail, Kiwix cascade is used."""
    mock_config.return_value = {
        "embeddings": {"model": "test-model"},
        "vectordb": {"path": "./test_db"},
        "rag": {"use_llm_for_kiwix_keywords": True, "max_distance": 0.60},
    }
    mock_kiwix.has_zim_paths.return_value = True

    # ChromaDB setup — relevant chunks but LLM can't answer from them
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
    mock_st.return_value = mock_model

    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Some partial info about volcanoes"]],
        "metadatas": [[{"title": "Volcanisme"}]],
        "distances": [[0.30]],
    }
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    # Kiwix cascade: LLM-assisted finds answer
    mock_kiwix.search.return_value = [
        {"title": "Piton des Neiges", "content": "Le Piton culmine à 3070m."},
    ]

    mock_llm.generate.side_effect = [
        "Je ne sais pas.",                  # ChromaDB chunk fails
        "Piton des Neiges",                 # LLM expressions
        "Le Piton culmine à 3070m.",        # Kiwix phase 1 answer
    ]

    result = rag_module.query("Altitude du Piton des Neiges ?")

    assert "3070" in result
    assert "[Recherche textuelle Kiwix]" in result
    # ChromaDB was queried first, then Kiwix cascade
    mock_collection.query.assert_called_once()
