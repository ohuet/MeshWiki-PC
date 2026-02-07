"""RAG pipeline: semantic search in ChromaDB + LLM answer generation."""

import logging

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from meshwiki import llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un assistant encyclopédique offline.
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
Si les extraits ne contiennent pas la réponse, indique que tu n'as pas trouvé la réponse.
Ne fabrique JAMAIS d'information."""

SYSTEM_PROMPT_NO_INDEX = """Tu es un assistant encyclopédique sur l'île de La Réunion.
La base Wikipedia est en cours d'initialisation (temps restant : {eta}).
Tu ne disposes PAS d'extraits Wikipedia pour le moment.
Réponds du mieux possible avec tes connaissances générales.
Termine ta réponse par : "[Sans source Wikipedia — base disponible dans {eta}]"
Ta réponse doit faire une ou deux phrases, max 400 caractères (transmission radio)."""

_model = None
_collection = None


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _get_model():
    global _model
    if _model is None:
        config = _load_config()
        _model = SentenceTransformer(config["embeddings"]["model"])
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        config = _load_config()
        client = chromadb.PersistentClient(path=config["vectordb"]["path"])
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
        n_results=5,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "Aucun article pertinent trouvé pour cette question."

    # Build context from retrieved chunks
    context_parts = []
    for doc, meta in zip(documents, metadatas):
        title = meta.get("title", "")
        context_parts.append(f"[{title}] {doc}")

    context = "\n\n".join(context_parts)

    user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise."""

    return llm.generate(SYSTEM_PROMPT, user_prompt)


def query_without_context(question: str, eta_str: str) -> str:
    """Answer a question without Wikipedia context (during indexation)."""
    system = SYSTEM_PROMPT_NO_INDEX.format(eta=eta_str)
    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    return llm.generate(system, user_prompt)
