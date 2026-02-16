"""Remote embedding via OpenAI-compatible or Ollama /api/embed API."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger(__name__)

# Validated embedding dimension (set on first successful call, reset between runs)
_validated_dim: int | None = None


def reset() -> None:
    """Reset validated dimension state between indexing runs."""
    global _validated_dim
    _validated_dim = None


def is_configured(config: dict) -> bool:
    """Check whether remote embedding is configured in config."""
    remote = config.get("embeddings", {}).get("remote")
    return remote is not None and "base_url" in remote


def _parse_openai(data: dict, count: int):
    """Parse OpenAI /v1/embeddings response format.

    Returns list of embedding vectors in input order, or None on error.
    """
    results = data.get("data", [])
    if len(results) != count:
        logger.error(
            "Remote embedding returned %d results for %d texts",
            len(results), count,
        )
        return None
    # Sort by index to guarantee input order
    results.sort(key=lambda x: x["index"])
    return [item["embedding"] for item in results]


def _parse_ollama(data: dict, count: int):
    """Parse Ollama /api/embed response format.

    Returns list of embedding vectors in input order, or None on error.
    """
    embeddings = data.get("embeddings", [])
    if len(embeddings) != count:
        logger.error(
            "Remote embedding returned %d results for %d texts",
            len(embeddings), count,
        )
        return None
    return embeddings


def prepare(config: dict) -> dict | None:
    """Prepare remote embedding request parameters from config.

    Returns a dict with url, model, timeout, parse_fn, session (with
    keep-alive and auth headers), or None if remote is not configured.
    """
    if not is_configured(config):
        return None

    remote = config["embeddings"]["remote"]
    base_url = remote["base_url"].rstrip("/")
    api_type = remote.get("api_type", "openai")
    api_key = remote.get("api_key")

    if api_type == "ollama":
        url = f"{base_url}/api/embed"
        parse_fn = _parse_ollama
    else:
        url = f"{base_url}/v1/embeddings"
        parse_fn = _parse_openai

    session = requests.Session()
    if api_key:
        session.headers["Authorization"] = f"Bearer {api_key}"

    read_timeout = remote.get("timeout", 300)
    connect_timeout = min(30, read_timeout)

    return {
        "url": url,
        "model": remote["model"],
        "timeout": (connect_timeout, read_timeout),
        "parse_fn": parse_fn,
        "session": session,
        "expected_dim": config["embeddings"].get("embedding_dim"),
    }


def encode_single(texts: list[str], params: dict) -> list[list[float]] | None:
    """Encode texts via a single HTTP request.

    Uses the params dict from ``prepare()``.
    Returns a list of embedding vectors in input order, or None on failure.
    """
    global _validated_dim

    url = params["url"]
    t0 = time.monotonic()

    try:
        resp = params["session"].post(
            url,
            json={"input": texts, "model": params["model"]},
            timeout=params["timeout"],
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(
            "Remote embedding timeout after %ss for %d texts",
            params["timeout"], len(texts),
        )
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error("Remote embedding connection error: %s — %s", url, e)
        return None
    except requests.exceptions.HTTPError as e:
        logger.error("Remote embedding HTTP error: %s", e)
        return None

    data = resp.json()
    embeddings = params["parse_fn"](data, len(texts))
    if embeddings is None:
        return None

    # Validate dimension on first successful result
    if embeddings and _validated_dim is None:
        dim = len(embeddings[0])
        expected_dim = params["expected_dim"]
        if expected_dim is not None and dim != expected_dim:
            logger.error(
                "Remote embedding dimension mismatch: got %d, expected %d",
                dim, expected_dim,
            )
            return None
        _validated_dim = dim
        logger.info("Remote embedding dimension validated: %d", dim)

    elapsed = time.monotonic() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0
    logger.debug(
        "Remote embedding: %d texts in %.1fs (%.0f texts/s)",
        len(texts), elapsed, rate,
    )

    return embeddings


def encode_batch(
    texts: list[str],
    config: dict,
    on_sub_batch: Callable[[int, int], None] | None = None,
) -> list[list[float]] | None:
    """Encode texts via a remote embedding API.

    Splits texts into sub-batches sent concurrently using ``max_concurrent``
    threads (default 4). Uses ``requests.Session`` for connection reuse.

    If *on_sub_batch* is provided, it is called after each sub-batch completes
    with ``(completed_count, total_count)``.

    Returns a list of embedding vectors, or None on any failure.
    """
    params = prepare(config)
    if params is None:
        return None

    remote = config["embeddings"]["remote"]
    batch_size = remote.get("batch_size", 256)
    max_concurrent = remote.get("max_concurrent", 4)

    # Build sub-batches: list of (start_index, texts)
    sub_batches = []
    for start in range(0, len(texts), batch_size):
        sub_batches.append((start, texts[start : start + batch_size]))

    results: dict[int, list[list[float]]] = {}
    t0_total = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(encode_single, sb_texts, params): sb_start
            for sb_start, sb_texts in sub_batches
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                return None
            results[futures[future]] = result
            completed += 1
            if on_sub_batch is not None:
                on_sub_batch(completed, len(sub_batches))

    # Concatenate in sub-batch order
    all_embeddings = []
    for sb_start, _ in sub_batches:
        all_embeddings.extend(results[sb_start])

    elapsed_total = time.monotonic() - t0_total
    rate_total = len(texts) / elapsed_total if elapsed_total > 0 else 0
    logger.info(
        "Remote embedding total: %d texts in %.1fs (%.0f texts/s, %d concurrent)",
        len(texts), elapsed_total, rate_total, max_concurrent,
    )

    return all_embeddings
