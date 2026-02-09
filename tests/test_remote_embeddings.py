"""Tests for meshwiki.remote_embeddings — Remote embedding API client."""

from unittest.mock import MagicMock, patch

import requests

import meshwiki.remote_embeddings as remote_mod
from meshwiki.remote_embeddings import encode_batch, is_configured, reset


MOCK_CONFIG = {
    "embeddings": {
        "model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "remote": {
            "base_url": "http://gpu-server:3000",
            "model": "bge-m3",
            "batch_size": 256,
            "timeout": 300,
        },
    }
}

MOCK_CONFIG_WITH_KEY = {
    "embeddings": {
        "model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "remote": {
            "base_url": "http://gpu-server:3000",
            "model": "bge-m3",
            "api_key": "sk-test-key-123",
            "batch_size": 256,
            "timeout": 300,
        },
    }
}

MOCK_CONFIG_OLLAMA = {
    "embeddings": {
        "model": "BAAI/bge-m3",
        "embedding_dim": 1024,
        "remote": {
            "base_url": "http://gpu-server:11434",
            "model": "bge-m3",
            "api_type": "ollama",
            "batch_size": 256,
            "timeout": 300,
        },
    }
}

MOCK_CONFIG_NO_REMOTE = {
    "embeddings": {
        "model": "BAAI/bge-m3",
        "embedding_dim": 1024,
    }
}


def _make_response(texts, dim=1024, shuffle=False):
    """Build a mock OpenAI API response with embeddings of the given dimension."""
    data = []
    indices = list(range(len(texts)))
    if shuffle:
        indices = list(reversed(indices))
    for i, idx in enumerate(indices):
        data.append({"index": idx, "embedding": [0.1] * dim})
    return {"data": data}


def _make_ollama_response(count, dim=1024):
    """Build a mock Ollama /api/embed response."""
    return {"embeddings": [[0.1] * dim for _ in range(count)]}


def setup_function():
    reset()


def test_not_configured_returns_none():
    result = encode_batch(["hello"], MOCK_CONFIG_NO_REMOTE)
    assert result is None


def test_is_configured_true():
    assert is_configured(MOCK_CONFIG) is True


def test_is_configured_false():
    assert is_configured(MOCK_CONFIG_NO_REMOTE) is False


@patch("meshwiki.remote_embeddings.requests.post")
def test_successful_encoding(mock_post):
    texts = ["hello world", "bonjour le monde"]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(texts)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = encode_batch(texts, MOCK_CONFIG)

    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 1024
    mock_post.assert_called_once()


@patch("meshwiki.remote_embeddings.requests.post")
def test_dimension_mismatch_returns_none(mock_post):
    texts = ["hello"]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(texts, dim=768)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = encode_batch(texts, MOCK_CONFIG)

    assert result is None


@patch("meshwiki.remote_embeddings.requests.post")
def test_timeout_returns_none(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()

    result = encode_batch(["hello"], MOCK_CONFIG)

    assert result is None


@patch("meshwiki.remote_embeddings.requests.post")
def test_connection_error_returns_none(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()

    result = encode_batch(["hello"], MOCK_CONFIG)

    assert result is None


@patch("meshwiki.remote_embeddings.requests.post")
def test_results_sorted_by_index(mock_post):
    texts = ["a", "b", "c"]
    # Return results in reversed order
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(texts, shuffle=True)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = encode_batch(texts, MOCK_CONFIG)

    assert result is not None
    assert len(result) == 3


@patch("meshwiki.remote_embeddings.requests.post")
def test_sub_batching(mock_post):
    """600 texts with batch_size=256 should produce 3 HTTP calls."""
    texts = [f"text {i}" for i in range(600)]

    def side_effect(*args, **kwargs):
        sub_batch = kwargs.get("json", args[1] if len(args) > 1 else {}).get("input", [])
        resp = MagicMock()
        resp.json.return_value = _make_response(sub_batch)
        resp.raise_for_status = MagicMock()
        return resp

    mock_post.side_effect = side_effect

    result = encode_batch(texts, MOCK_CONFIG)

    assert result is not None
    assert len(result) == 600
    assert mock_post.call_count == 3  # 256 + 256 + 88


@patch("meshwiki.remote_embeddings.requests.post")
def test_url_construction(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(["test"])
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    encode_batch(["test"], MOCK_CONFIG)

    call_args = mock_post.call_args
    assert call_args[0][0] == "http://gpu-server:3000/v1/embeddings"


@patch("meshwiki.remote_embeddings.requests.post")
def test_http_error_returns_none(mock_post):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    mock_post.return_value = mock_resp

    result = encode_batch(["hello"], MOCK_CONFIG)

    assert result is None


@patch("meshwiki.remote_embeddings.requests.post")
def test_api_key_sent_as_bearer_token(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(["test"])
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    encode_batch(["test"], MOCK_CONFIG_WITH_KEY)

    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer sk-test-key-123"


@patch("meshwiki.remote_embeddings.requests.post")
def test_no_api_key_sends_no_auth_header(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(["test"])
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    encode_batch(["test"], MOCK_CONFIG)

    headers = mock_post.call_args[1]["headers"]
    assert "Authorization" not in headers


# --- Ollama API type tests ---


@patch("meshwiki.remote_embeddings.requests.post")
def test_ollama_url_construction(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(1)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    encode_batch(["test"], MOCK_CONFIG_OLLAMA)

    assert mock_post.call_args[0][0] == "http://gpu-server:11434/api/embed"


@patch("meshwiki.remote_embeddings.requests.post")
def test_ollama_successful_encoding(mock_post):
    texts = ["hello", "world", "test"]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(3)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = encode_batch(texts, MOCK_CONFIG_OLLAMA)

    assert result is not None
    assert len(result) == 3
    assert len(result[0]) == 1024


@patch("meshwiki.remote_embeddings.requests.post")
def test_ollama_dimension_mismatch_returns_none(mock_post):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(1, dim=768)
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    result = encode_batch(["hello"], MOCK_CONFIG_OLLAMA)

    assert result is None


@patch("meshwiki.remote_embeddings.requests.post")
def test_ollama_sub_batching(mock_post):
    """600 texts with batch_size=256 should produce 3 HTTP calls (Ollama)."""
    texts = [f"text {i}" for i in range(600)]

    def side_effect(*args, **kwargs):
        sub_batch = kwargs.get("json", {}).get("input", [])
        resp = MagicMock()
        resp.json.return_value = _make_ollama_response(len(sub_batch))
        resp.raise_for_status = MagicMock()
        return resp

    mock_post.side_effect = side_effect

    result = encode_batch(texts, MOCK_CONFIG_OLLAMA)

    assert result is not None
    assert len(result) == 600
    assert mock_post.call_count == 3


@patch("meshwiki.remote_embeddings.requests.post")
def test_default_api_type_is_openai(mock_post):
    """Config without api_type defaults to OpenAI endpoint."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(["test"])
    mock_resp.raise_for_status = MagicMock()
    mock_post.return_value = mock_resp

    encode_batch(["test"], MOCK_CONFIG)

    assert "/v1/embeddings" in mock_post.call_args[0][0]
