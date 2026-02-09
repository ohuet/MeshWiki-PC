"""Remote embedding via OpenAI-compatible /v1/embeddings API."""

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


def encode_batch(texts: list[str], config: dict) -> list[list[float]] | None:
    """Encode texts via a remote OpenAI-compatible /v1/embeddings API.

    Sends texts in sub-batches of configurable size to avoid oversized payloads.
    Results are sorted by the ``index`` field returned by the API.

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
    expected_dim = config["embeddings"].get("embedding_dim")
    url = f"{base_url}/v1/embeddings"

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
        results = data.get("data", [])

        if len(results) != len(sub_batch):
            logger.error(
                "Remote embedding returned %d results for %d texts",
                len(results), len(sub_batch),
            )
            return None

        # Validate dimension on first successful result
        if results:
            dim = len(results[0]["embedding"])
            if _validated_dim is None:
                if expected_dim is not None and dim != expected_dim:
                    logger.error(
                        "Remote embedding dimension mismatch: got %d, expected %d",
                        dim, expected_dim,
                    )
                    return None
                _validated_dim = dim
                logger.info("Remote embedding dimension validated: %d", dim)

        for item in results:
            all_embeddings.append((start + item["index"], item["embedding"]))

        elapsed = time.monotonic() - t0
        rate = len(sub_batch) / elapsed if elapsed > 0 else 0
        logger.info(
            "Remote embedding: %d texts in %.1fs (%.0f texts/s)",
            len(sub_batch), elapsed, rate,
        )

    # Sort by original index to guarantee order
    all_embeddings.sort(key=lambda x: x[0])
    return [emb for _, emb in all_embeddings]
