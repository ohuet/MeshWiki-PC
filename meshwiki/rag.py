"""RAG pipeline: semantic search in ChromaDB + LLM answer generation."""

import logging
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from meshwiki import kiwix_search, llm
from meshwiki import config, collection_state

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un assistant encyclopédique offline.
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Si les extraits ne contiennent PAS d'éléments permettant de répondre, réponds exactement "Je n'ai pas trouvé cette information." Ne complète JAMAIS avec tes propres connaissances."""

SYSTEM_PROMPT_NO_INDEX = """Tu es un assistant encyclopédique sur l'île de La Réunion.
La base Wikipedia est en cours d'initialisation (temps restant : {eta}).
Tu ne disposes PAS d'extraits Wikipedia pour le moment.
Réponds du mieux possible avec tes connaissances générales.
Termine ta réponse par : "[Sans source Wikipedia — base disponible dans {eta}]"
Ta réponse doit faire une ou deux phrases, max 400 caractères (transmission radio)."""

SYSTEM_PROMPT_KIWIX = """Tu es un assistant encyclopédique offline.
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Si les extraits ne contiennent PAS d'éléments permettant de répondre, réponds exactement "Je n'ai pas trouvé cette information." Ne complète JAMAIS avec tes propres connaissances.
Termine ta réponse par : "[Recherche Kiwix — base optimisée dans {eta}]" """

SYSTEM_PROMPT_NO_DATA = """Tu es un assistant encyclopédique sur l'île de La Réunion.
Tu ne disposes PAS d'extraits Wikipedia pour le moment.
Réponds du mieux possible avec tes connaissances générales.
Termine ta réponse par : "[Sans source Wikipedia]"
Ta réponse doit faire une ou deux phrases, max 400 caractères (transmission radio)."""

SYSTEM_PROMPT_KIWIX_PERMANENT = """Tu es un assistant encyclopédique offline.
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Si les extraits ne contiennent PAS d'éléments permettant de répondre, réponds exactement "Je n'ai pas trouvé cette information." Ne complète JAMAIS avec tes propres connaissances.
Termine ta réponse par : "[Recherche textuelle Kiwix]" """

_model = None
_collection = None
_force_unavailable = False


def _get_model():
    global _model
    if _model is None:
        cfg = config.load_config()
        embeddings_config = cfg["embeddings"]
        model_name = embeddings_config["model"]
        truncate_dim = embeddings_config.get("truncate_dim")
        model_kwargs = {"torch_dtype": "float16"}
        try:
            _model = SentenceTransformer(
                model_name, truncate_dim=truncate_dim, local_files_only=True,
                model_kwargs=model_kwargs,
            )
        except OSError:
            logger.info("Downloading embedding model: %s (first time)", model_name)
            _model = SentenceTransformer(
                model_name, truncate_dim=truncate_dim,
                model_kwargs=model_kwargs,
            )
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        cfg = config.load_config()
        client = chromadb.PersistentClient(path=collection_state.get_active_db_path())
        _collection = client.get_collection("wikipedia")
    return _collection


def reset_collection() -> None:
    """Invalidate the cached collection so the next query re-fetches it.

    Must be called after a collection rename/swap in the updater.
    """
    global _collection
    _collection = None


def query(question: str) -> str:
    """Answer a question using semantic search over Wikipedia + LLM.

    Returns the LLM-generated answer or an error message in French.
    """
    try:
        model = _get_model()
        collection = _get_collection()
    except Exception as e:
        logger.error("Failed to load RAG components: %s", e)
        return "Erreur : base Wikipedia indisponible."

    # Encode the question
    question_embedding = model.encode(question).tolist()

    # Semantic search
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Log all retrieved chunks for debugging
    for doc, meta, dist in zip(documents, metadatas, distances):
        logger.debug("  [%.3f] %s: %.80s...", dist, meta.get("title", "?"), doc)

    # Filter out low-relevance results (cosine distance > threshold)
    max_distance = 0.45
    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
        if dist <= max_distance
    ]

    if not filtered:
        logger.info("No relevant chunks found (best distance: %.3f)", distances[0] if distances else -1)
        return "Aucun article pertinent trouvé pour cette question."

    logger.info("Search: %d/%d chunks kept (distance <= %.2f)", len(filtered), len(documents), max_distance)

    # Build context from retrieved chunks
    context_parts = []
    for doc, meta, dist in filtered:
        title = meta.get("title", "")
        context_parts.append(f"[{title}] {doc}")

    context = "\n\n".join(context_parts)

    user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si les extraits ne contiennent pas la réponse, dis "Je ne sais pas"."""

    return llm.generate(SYSTEM_PROMPT, user_prompt)


def query_without_context(question: str, eta_str: str) -> str:
    """Answer a question without Wikipedia context (during indexation)."""
    system = SYSTEM_PROMPT_NO_INDEX.format(eta=eta_str)
    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    return llm.generate(system, user_prompt)


def query_with_kiwix_context(question: str, eta_str: str) -> str:
    """Answer a question using Kiwix full-text search as context (during indexation).

    Falls back to query_without_context() if Kiwix returns no results.
    """
    results = kiwix_search.search(question)
    if not results:
        return query_without_context(question, eta_str)

    context_parts = []
    for r in results:
        context_parts.append(f"[{r['title']}] {r['content']}")
    context = "\n\n".join(context_parts)

    system = SYSTEM_PROMPT_KIWIX.format(eta=eta_str)
    user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si les extraits ne contiennent pas la réponse, dis "Je ne sais pas"."""

    return llm.generate(system, user_prompt)


def set_force_unavailable(value: bool) -> None:
    """Force is_available() to return False (used by --noindex flag)."""
    global _force_unavailable
    _force_unavailable = value


def is_available() -> bool:
    """Check if the ChromaDB Wikipedia index is available and compatible.

    Returns False if the collection doesn't exist, is empty, uses
    embeddings with a different dimension, or --noindex is active.
    """
    if _force_unavailable:
        return False
    try:
        cfg = config.load_config()
        db_path = Path(collection_state.get_active_db_path())
        if not db_path.exists():
            return False
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("wikipedia")
        if collection.count() == 0:
            return False
        expected_dim = cfg["embeddings"].get("truncate_dim") or cfg["embeddings"].get("embedding_dim")
        if expected_dim:
            sample = collection.peek(limit=1)
            if len(sample["embeddings"]) > 0:
                actual_dim = len(sample["embeddings"][0])
                if actual_dim != expected_dim:
                    return False
        return True
    except Exception as e:
        logger.warning("Index check failed: %s", e)
        return False


def query_with_kiwix_context_permanent(question: str) -> str:
    """Answer a question using Kiwix full-text search as permanent fallback (no ETA).

    Used when no ChromaDB index exists and no indexation is in progress.
    Returns a simple message if Kiwix returns no results.
    """
    results = kiwix_search.search(question)
    if not results:
        return "Aucun résultat trouvé pour cette question."

    context_parts = []
    for r in results:
        context_parts.append(f"[{r['title']}] {r['content']}")
    context = "\n\n".join(context_parts)

    user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si les extraits ne contiennent pas la réponse, dis "Je ne sais pas"."""

    return llm.generate(SYSTEM_PROMPT_KIWIX_PERMANENT, user_prompt)


def query_without_data(question: str) -> str:
    """Answer a question using only the LLM's general knowledge.

    Used when neither the ChromaDB index nor a ZIM file are available.
    """
    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    return llm.generate(SYSTEM_PROMPT_NO_DATA, user_prompt)
