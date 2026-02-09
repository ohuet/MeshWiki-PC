"""Remote embedding via OpenAI-compatible or Ollama /api/embed API."""

import logging
import time

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


def _parse_openai(data: dict, sub_batch_len: int, start: int):
    """Parse OpenAI /v1/embeddings response format.

    Returns list of (global_index, embedding) tuples, or None on error.
    """
    results = data.get("data", [])
    if len(results) != sub_batch_len:
        logger.error(
            "Remote embedding returned %d results for %d texts",
            len(results), sub_batch_len,
        )
        return None
    return [(start + item["index"], item["embedding"]) for item in results]


def _parse_ollama(data: dict, sub_batch_len: int, start: int):
    """Parse Ollama /api/embed response format.

    Returns list of (global_index, embedding) tuples, or None on error.
    """
    embeddings = data.get("embeddings", [])
    if len(embeddings) != sub_batch_len:
        logger.error(
            "Remote embedding returned %d results for %d texts",
            len(embeddings), sub_batch_len,
        )
        return None
    return [(start + i, emb) for i, emb in enumerate(embeddings)]


def encode_batch(texts: list[str], config: dict) -> list[list[float]] | None:
    """Encode texts via a remote embedding API.

    Supports two API types (config key ``api_type``):
    - ``"openai"`` (default): POST to ``<base_url>/v1/embeddings``
    - ``"ollama"``: POST to ``<base_url>/api/embed``

    Sends texts in sub-batches of configurable size to avoid oversized payloads.

    Returns a list of embedding vectors, or None on any failure (timeout,
    connection error, HTTP error, dimension mismatch).
    """
    global _validated_dim

    if not is_configured(config):
        return None

    remote = config["embeddings"]["remote"]
    base_url = remote["base_url"].rstrip("/")
    model = remote["model"]
    batch_size = remote.get("batch_size", 256)
    timeout = remote.get("timeout", 300)
    api_key = remote.get("api_key")
    api_type = remote.get("api_type", "openai")
    expected_dim = config["embeddings"].get("embedding_dim")

    if api_type == "ollama":
        url = f"{base_url}/api/embed"
        parse_fn = _parse_ollama
    else:
        url = f"{base_url}/v1/embeddings"
        parse_fn = _parse_openai

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    all_embeddings: list[tuple[int, list[float]]] = []

    for start in range(0, len(texts), batch_size):
        sub_batch = texts[start : start + batch_size]
        t0 = time.monotonic()

        try:
            resp = requests.post(
                url,
                json={"input": sub_batch, "model": model},
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("Remote embedding timeout after %ds for %d texts", timeout, len(sub_batch))
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error("Remote embedding connection error: %s — %s", url, e)
            return None
        except requests.exceptions.HTTPError as e:
            logger.error("Remote embedding HTTP error: %s", e)
            return None

        data = resp.json()
        parsed = parse_fn(data, len(sub_batch), start)
        if parsed is None:
            return None

        # Validate dimension on first successful result
        if parsed:
            dim = len(parsed[0][1])
            if _validated_dim is None:
                if expected_dim is not None and dim != expected_dim:
                    logger.error(
                        "Remote embedding dimension mismatch: got %d, expected %d",
                        dim, expected_dim,
                    )
                    return None
                _validated_dim = dim
                logger.info("Remote embedding dimension validated: %d", dim)

        all_embeddings.extend(parsed)

        elapsed = time.monotonic() - t0
        rate = len(sub_batch) / elapsed if elapsed > 0 else 0
        logger.info(
            "Remote embedding: %d texts in %.1fs (%.0f texts/s)",
            len(sub_batch), elapsed, rate,
        )

    # Sort by original index to guarantee order
    all_embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in all_embeddings]
