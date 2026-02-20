"""RAG pipeline: semantic search in ChromaDB + LLM answer generation."""

import logging
import re
from collections import OrderedDict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from meshwiki import kiwix_search, llm
from meshwiki import config, collection_state

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Tu es un assistant encyclopédique offline sur l'île de La Réunion (France).
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Privilégie les informations relatives à la France et La Réunion.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Lis TOUS les extraits avant de répondre. Choisis celui qui répond le MIEUX à la question.
Si la question porte sur un traitement ou remède, choisis l'extrait qui propose une solution, pas celui qui mentionne le symptôme comme effet secondaire.
Si l'extrait contient des éléments de réponse, même partiels ou nuancés, utilise-les.
INTERDIT : Ne réponds JAMAIS "Non" ou "Ce n'est pas le cas" si aucun extrait ne le dit explicitement. Si les extraits ne parlent pas du sujet, réponds "Je ne sais pas."
Ne complète JAMAIS avec tes propres connaissances."""

SYSTEM_PROMPT_NO_INDEX = """Tu es un assistant encyclopédique sur l'île de La Réunion.
La base Wikipedia est en cours d'initialisation.
Tu ne disposes PAS d'extraits Wikipedia pour le moment.
Réponds du mieux possible avec tes connaissances générales.
Ta réponse doit faire une ou deux phrases, max 400 caractères (transmission radio)."""

SYSTEM_PROMPT_KIWIX = """Tu es un assistant encyclopédique offline sur l'île de La Réunion (France).
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Privilégie les informations relatives à la France et La Réunion.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Lis TOUS les extraits avant de répondre. Choisis celui qui répond le MIEUX à la question.
Si la question porte sur un traitement ou remède, choisis l'extrait qui propose une solution, pas celui qui mentionne le symptôme comme effet secondaire.
Si l'extrait contient des éléments de réponse, même partiels ou nuancés, utilise-les.
INTERDIT : Ne réponds JAMAIS "Non" ou "Ce n'est pas le cas" si aucun extrait ne le dit explicitement. Si les extraits ne parlent pas du sujet, réponds "Je ne sais pas."
Ne complète JAMAIS avec tes propres connaissances."""

SYSTEM_PROMPT_NO_DATA = """Tu es un assistant encyclopédique sur l'île de La Réunion.
Tu ne disposes PAS d'extraits Wikipedia pour le moment.
Réponds du mieux possible avec tes connaissances générales.
Ta réponse doit faire une ou deux phrases, max 400 caractères (transmission radio)."""

SYSTEM_PROMPT_KIWIX_PERMANENT = """Tu es un assistant encyclopédique offline sur l'île de La Réunion (France).
Tu réponds UNIQUEMENT à partir des extraits Wikipedia fournis ci-dessous.
Privilégie les informations relatives à la France et La Réunion.
Tes réponses doivent être :
- Concises (max 400 caractères si possible, car transmises par radio)
- Factuelles et précises
- En français
IMPORTANT : Lis TOUS les extraits avant de répondre. Choisis celui qui répond le MIEUX à la question.
Si la question porte sur un traitement ou remède, choisis l'extrait qui propose une solution, pas celui qui mentionne le symptôme comme effet secondaire.
Si l'extrait contient des éléments de réponse, même partiels ou nuancés, utilise-les.
INTERDIT : Ne réponds JAMAIS "Non" ou "Ce n'est pas le cas" si aucun extrait ne le dit explicitement. Si les extraits ne parlent pas du sujet, réponds "Je ne sais pas."
Ne complète JAMAIS avec tes propres connaissances."""

SYSTEM_PROMPT_LLM_OPINION = """Tu es un assistant encyclopédique. L'information n'a PAS été trouvée dans la base Wikipedia locale.
Réponds avec tes connaissances générales.
Ta réponse doit être concise (max 400 caractères, transmission radio)."""

SYSTEM_PROMPT_KEYWORDS = """Tu es un moteur de recherche. Donne UNIQUEMENT une liste d'expressions de recherche Wikipedia (1 à 4, une par ligne). Pas d'explication, pas de numérotation."""

# Known French error messages returned by llm.generate() on failure
_LLM_ERROR_MESSAGES = (
    "Erreur : le modèle LLM n'a pas répondu à temps.",
    "Service LLM indisponible.",
    "Erreur lors de la génération de la réponse.",
)

_RE_WIKI_REFS = re.compile(r"\[(?:\d+|Notes?\s*\d+)\]")

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
        model_kwargs = {"dtype": "float16"}
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

    When use_llm_for_kiwix_keywords is enabled, tries LLM-assisted Kiwix
    search first, then falls back to ChromaDB semantic search.

    Returns the LLM-generated answer or an error message in French.
    """
    # If LLM-assisted Kiwix search is enabled, try it first
    try:
        cfg = config.load_config()
        use_llm_kiwix = cfg.get("rag", {}).get("use_llm_for_kiwix_keywords", True)
    except Exception:
        use_llm_kiwix = False
    tried_llm_kiwix = False

    if use_llm_kiwix and kiwix_search.has_zim_paths():
        tried_llm_kiwix = True
        llm_result = _llm_assisted_kiwix_search(
            question, SYSTEM_PROMPT_KIWIX_PERMANENT, "[Recherche textuelle Kiwix]"
        )
        if llm_result is not None:
            return llm_result
        logger.info("Recherche Kiwix assistée LLM non concluante, tentative ChromaDB...")

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
    cfg = config.load_config()
    max_distance = cfg.get("rag", {}).get("max_distance", 0.60)
    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
        if dist <= max_distance
    ]

    if not filtered:
        logger.info("No relevant chunks found (best distance: %.3f)", distances[0] if distances else -1)
        return "Aucun article pertinent trouvé pour cette question."

    # Deduplicate: max 2 chunks per article to ensure diversity
    max_per_article = 2
    title_counts: dict[str, int] = {}
    deduplicated = []
    for item in filtered:
        title = item[1].get("title", "")
        count = title_counts.get(title, 0)
        if count < max_per_article:
            deduplicated.append(item)
            title_counts[title] = count + 1
    filtered = deduplicated

    logger.info("Search: %d/%d chunks kept (distance <= %.2f)", len(filtered), len(documents), max_distance)

    # Try chunks in groups of 4 until the LLM finds an answer
    group_size = 4
    for start in range(0, len(filtered), group_size):
        group = filtered[start:start + group_size]
        context_parts = [
            f"[{meta.get('title', '')}] {_RE_WIKI_REFS.sub('', doc)}"
            for doc, meta, dist in group
        ]
        context = "\n\n".join(context_parts)
        titles = [meta.get("title", "?") for _, meta, _ in group]

        user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si aucun extrait ne parle du sujet précis de la question, réponds "Je ne sais pas." Ne réponds jamais "Non" à partir d'extraits qui ne mentionnent pas le sujet."""

        response = llm.generate(SYSTEM_PROMPT, user_prompt)

        group_label = f"{start + 1}-{start + len(group)}/{len(filtered)}"
        if not _is_no_answer(response):
            logger.info("Réponse trouvée aux chunks %s %s", group_label, titles)
            return response

        logger.info("Chunks %s %s : pas de réponse", group_label, titles)

    # All chunks exhausted — try Kiwix cascade (skip LLM path if already tried)
    if kiwix_search.has_zim_paths():
        logger.info("Aucun chunk ChromaDB concluant, tentative Kiwix...")
        kiwix_result = _kiwix_cascade(
            question, SYSTEM_PROMPT_KIWIX_PERMANENT, "[Recherche textuelle Kiwix]",
            skip_llm=tried_llm_kiwix,
        )
        if kiwix_result is not None:
            return kiwix_result

    return response


def query_without_context(question: str, eta_str: str) -> str:
    """Answer a question without Wikipedia context (during indexation)."""
    system = SYSTEM_PROMPT_NO_INDEX
    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    response = llm.generate(system, user_prompt)
    return f"{response} [Sans source Wikipedia — base disponible dans {eta_str}]"


def _is_no_answer(response: str) -> bool:
    """Detect if the LLM response indicates the information was not found."""
    lower = response.lower()
    return "je n'ai pas trouvé" in lower or "je ne sais pas" in lower


def _try_kiwix_articles(results: list[dict], question: str, system_prompt: str, suffix: str) -> str | None:
    """Try to answer a question from a list of Kiwix search results.

    Step 1: All articles together (long extracts) → LLM.
    Steps 2-N: Each article individually (full content) → LLM.

    Returns the answer with suffix, or None if all attempts say "je ne sais pas".
    """
    # Step 1: Try with long extracts from all articles
    context_parts = [f"[{r['title']}] {r['content']}" for r in results]
    context = "\n\n".join(context_parts)
    user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si aucun extrait ne parle du sujet précis de la question, réponds "Je ne sais pas." Ne réponds jamais "Non" à partir d'extraits qui ne mentionnent pas le sujet."""

    response = llm.generate(system_prompt, user_prompt)
    if not _is_no_answer(response):
        return f"{response} {suffix}"

    # Steps 2-N: Try each full article individually
    for r in results:
        full_content = kiwix_search.get_article_content(r["title"])
        if not full_content:
            continue
        context = f"[{r['title']}] {full_content}"
        user_prompt = f"""Extraits Wikipedia pertinents :
---
{context}
---

Question : {question}

Réponds de façon concise. Si aucun extrait ne parle du sujet précis de la question, réponds "Je ne sais pas." Ne réponds jamais "Non" à partir d'extraits qui ne mentionnent pas le sujet."""

        response = llm.generate(system_prompt, user_prompt)
        if not _is_no_answer(response):
            return f"{response} {suffix}"

    return None


def _llm_search_expressions(question: str) -> list[str]:
    """Ask the LLM to generate 1-4 search expressions for a question.

    Returns an empty list if the LLM fails or returns an error.
    """
    user_prompt = f"Expressions de recherche pour : {question}"
    logger.info("Génération d'expressions de recherche LLM pour : %r", question)
    response = llm.generate(SYSTEM_PROMPT_KEYWORDS, user_prompt, max_tokens=100)

    # Check for known error messages
    if response in _LLM_ERROR_MESSAGES:
        logger.warning("LLM error generating search expressions: %s", response)
        return []

    logger.info("Réponse brute du LLM : %r", response)

    # Parse: split lines, strip, clean numbering prefixes, filter empty
    expressions = []
    for line in response.split("\n"):
        line = line.strip()
        # Remove numbering like "1. ", "2) ", "- ", "• "
        line = re.sub(r"^(?:\d+[.\)]\s*|[-•]\s*)", "", line).strip()
        if line:
            expressions.append(line)

    # Limit to 4 expressions
    expressions = expressions[:4]

    logger.info("Expressions retenues : %s", expressions)
    return expressions


def _llm_assisted_kiwix_search(question: str, system_prompt: str, suffix: str) -> str | None:
    """LLM-assisted Kiwix search: generate expressions, Phase 1/2.

    Asks the LLM to generate search expressions, searches Kiwix for each,
    and uses a two-phase strategy with deduplication.

    Returns the answer with suffix, or None if all phases fail or no
    expressions were generated. Does NOT fall back to LLM opinion.
    """
    expressions = _llm_search_expressions(question)
    if not expressions:
        return None

    # Search each expression
    all_results: OrderedDict[str, list[dict]] = OrderedDict()
    for expr in expressions:
        results = kiwix_search.search(expr, num_results=3, max_chars_per_result=4000)
        all_results[expr] = results
        titles = [r["title"] for r in results]
        logger.info("Recherche Kiwix %r → %d résultats : %s", expr, len(results), titles)

    seen_titles: set[str] = set()

    # Phase 1: best unique article from each expression
    phase1: list[dict] = []
    for expr, results in all_results.items():
        for r in results:
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                phase1.append(r)
                break

    if phase1:
        logger.info("Phase 1: %d articles from %d expressions", len(phase1), len(expressions))
        result = _try_kiwix_articles(phase1, question, system_prompt, suffix)
        if result is not None:
            return result

    # Phase 2: remaining articles from each expression
    for expr, results in all_results.items():
        remaining = [r for r in results if r["title"] not in seen_titles]
        if remaining:
            for r in remaining:
                seen_titles.add(r["title"])
            logger.info("Phase 2 (%s): %d remaining articles", expr, len(remaining))
            result = _try_kiwix_articles(remaining, question, system_prompt, suffix)
            if result is not None:
                return result

    return None


def _kiwix_cascade(question: str, system_prompt: str, suffix: str, skip_llm: bool = False) -> str | None:
    """Cascade Kiwix search: LLM expressions → Phase 1/2 → plain search → LLM opinion.

    When use_llm_for_kiwix_keywords is enabled and skip_llm is False, uses
    LLM-assisted search with a two-phase strategy and deduplication.
    Falls back to plain Kiwix search if disabled, skipped, or expressions fail.

    Returns the LLM answer with appropriate suffix/prefix, or None if
    Kiwix returned no results at all.
    """
    if not skip_llm:
        cfg = config.load_config()
        use_llm = cfg.get("rag", {}).get("use_llm_for_kiwix_keywords", True)

        if use_llm and kiwix_search.has_zim_paths():
            result = _llm_assisted_kiwix_search(question, system_prompt, suffix)
            if result is not None:
                return result

            # All LLM phases exhausted → LLM opinion
            user_prompt = f"Question : {question}\n\nRéponds de façon concise."
            response = llm.generate(SYSTEM_PROMPT_LLM_OPINION, user_prompt)
            return f"Réponse non trouvée dans la base. Mon avis : {response}"

    # Fallback: plain Kiwix search (option disabled, skipped, or no expressions)
    results = kiwix_search.search(question, max_chars_per_result=4000)
    if not results:
        return None

    result = _try_kiwix_articles(results, question, system_prompt, suffix)
    if result is not None:
        return result

    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    response = llm.generate(SYSTEM_PROMPT_LLM_OPINION, user_prompt)
    return f"Réponse non trouvée dans la base. Mon avis : {response}"


def query_with_kiwix_context(question: str, eta_str: str) -> str:
    """Answer a question using Kiwix full-text search as context (during indexation).

    Uses cascade strategy: long extracts → full articles → LLM opinion.
    Falls back to query_without_context() if Kiwix returns no results.
    """
    suffix = f"[Recherche Kiwix — base optimisée dans {eta_str}]"
    result = _kiwix_cascade(question, SYSTEM_PROMPT_KIWIX, suffix)
    if result is None:
        return query_without_context(question, eta_str)
    return result


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

    Uses cascade strategy: long extracts → full articles → LLM opinion.
    Used when no ChromaDB index exists and no indexation is in progress.
    Returns a simple message if Kiwix returns no results.
    """
    result = _kiwix_cascade(question, SYSTEM_PROMPT_KIWIX_PERMANENT, "[Recherche textuelle Kiwix]")
    if result is None:
        return "Aucun résultat trouvé pour cette question."
    return result


def query_without_data(question: str) -> str:
    """Answer a question using only the LLM's general knowledge.

    Used when neither the ChromaDB index nor a ZIM file are available.
    """
    user_prompt = f"Question : {question}\n\nRéponds de façon concise."
    response = llm.generate(SYSTEM_PROMPT_NO_DATA, user_prompt)
    return f"{response} [Sans source Wikipedia]"
