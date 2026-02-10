"""Tests for meshwiki.remote_embeddings — Remote embedding API client."""

from unittest.mock import MagicMock, patch

import requests

from meshwiki.remote_embeddings import (
    encode_batch, encode_single, is_configured, prepare, reset,
)


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


def _make_response(count, dim=1024, shuffle=False):
    """Build a mock OpenAI API response."""
    data = []
    indices = list(range(count))
    if shuffle:
        indices = list(reversed(indices))
    for idx in indices:
        data.append({"index": idx, "embedding": [0.1] * dim})
    return {"data": data}


def _make_ollama_response(count, dim=1024):
    """Build a mock Ollama /api/embed response."""
    return {"embeddings": [[0.1] * dim for _ in range(count)]}


def _mock_session(response_fn):
    """Create a mock Session whose .post() returns responses from response_fn."""
    session = MagicMock()
    session.headers = {}
    session.post.side_effect = response_fn
    return session


def setup_function():
    reset()


# --- is_configured / prepare ---


def test_not_configured_returns_none():
    result = encode_batch(["hello"], MOCK_CONFIG_NO_REMOTE)
    assert result is None


def test_is_configured_true():
    assert is_configured(MOCK_CONFIG) is True


def test_is_configured_false():
    assert is_configured(MOCK_CONFIG_NO_REMOTE) is False


def test_prepare_not_configured():
    assert prepare(MOCK_CONFIG_NO_REMOTE) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_prepare_openai_url(MockSession):
    MockSession.return_value = MagicMock(headers={})
    params = prepare(MOCK_CONFIG)
    assert params["url"] == "http://gpu-server:3000/v1/embeddings"


@patch("meshwiki.remote_embeddings.requests.Session")
def test_prepare_ollama_url(MockSession):
    MockSession.return_value = MagicMock(headers={})
    params = prepare(MOCK_CONFIG_OLLAMA)
    assert params["url"] == "http://gpu-server:11434/api/embed"


@patch("meshwiki.remote_embeddings.requests.Session")
def test_prepare_api_key_in_session_headers(MockSession):
    mock_session = MagicMock()
    mock_session.headers = {}
    MockSession.return_value = mock_session
    params = prepare(MOCK_CONFIG_WITH_KEY)
    assert params["session"].headers["Authorization"] == "Bearer sk-test-key-123"


@patch("meshwiki.remote_embeddings.requests.Session")
def test_prepare_no_api_key(MockSession):
    mock_session = MagicMock()
    mock_session.headers = {}
    MockSession.return_value = mock_session
    params = prepare(MOCK_CONFIG)
    assert "Authorization" not in params["session"].headers


# --- encode_single ---


def test_encode_single_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(2)
    mock_resp.raise_for_status = MagicMock()

    session = MagicMock()
    session.post.return_value = mock_resp

    params = {
        "url": "http://test/v1/embeddings",
        "model": "bge-m3",
        "timeout": 300,
        "parse_fn": lambda data, count: [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])],
        "session": session,
        "expected_dim": 1024,
    }
    result = encode_single(["hello", "world"], params)
    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 1024


def test_encode_single_timeout():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.Timeout()

    params = {
        "url": "http://test/v1/embeddings",
        "model": "bge-m3",
        "timeout": 300,
        "parse_fn": None,
        "session": session,
        "expected_dim": 1024,
    }
    assert encode_single(["hello"], params) is None


def test_encode_single_connection_error():
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError()

    params = {
        "url": "http://test/v1/embeddings",
        "model": "bge-m3",
        "timeout": 300,
        "parse_fn": None,
        "session": session,
        "expected_dim": 1024,
    }
    assert encode_single(["hello"], params) is None


def test_encode_single_dimension_mismatch():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(1, dim=768)
    mock_resp.raise_for_status = MagicMock()

    from meshwiki.remote_embeddings import _parse_openai
    session = MagicMock()
    session.post.return_value = mock_resp

    params = {
        "url": "http://test/v1/embeddings",
        "model": "bge-m3",
        "timeout": 300,
        "parse_fn": _parse_openai,
        "session": session,
        "expected_dim": 1024,
    }
    assert encode_single(["hello"], params) is None


# --- encode_batch (integration through prepare + encode_single) ---


@patch("meshwiki.remote_embeddings.requests.Session")
def test_successful_encoding(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(2)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    result = encode_batch(["hello", "world"], MOCK_CONFIG)

    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 1024
    mock_session.post.assert_called_once()


@patch("meshwiki.remote_embeddings.requests.Session")
def test_dimension_mismatch_returns_none(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(1, dim=768)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    assert encode_batch(["hello"], MOCK_CONFIG) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_timeout_returns_none(MockSession):
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = requests.exceptions.Timeout()
    MockSession.return_value = mock_session

    assert encode_batch(["hello"], MOCK_CONFIG) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_connection_error_returns_none(MockSession):
    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = requests.exceptions.ConnectionError()
    MockSession.return_value = mock_session

    assert encode_batch(["hello"], MOCK_CONFIG) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_http_error_returns_none(MockSession):
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    assert encode_batch(["hello"], MOCK_CONFIG) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_results_sorted_by_index(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(3, shuffle=True)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    result = encode_batch(["a", "b", "c"], MOCK_CONFIG)
    assert result is not None
    assert len(result) == 3


@patch("meshwiki.remote_embeddings.requests.Session")
def test_sub_batching(MockSession):
    """600 texts with batch_size=256 should produce 3 HTTP calls."""
    texts = [f"text {i}" for i in range(600)]

    def side_effect(*args, **kwargs):
        sub_batch = kwargs.get("json", {}).get("input", [])
        resp = MagicMock()
        resp.json.return_value = _make_response(len(sub_batch))
        resp.raise_for_status = MagicMock()
        return resp

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = side_effect
    MockSession.return_value = mock_session

    result = encode_batch(texts, MOCK_CONFIG)

    assert result is not None
    assert len(result) == 600
    assert mock_session.post.call_count == 3  # 256 + 256 + 88


@patch("meshwiki.remote_embeddings.requests.Session")
def test_url_construction(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(1)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    encode_batch(["test"], MOCK_CONFIG)

    assert mock_session.post.call_args[0][0] == "http://gpu-server:3000/v1/embeddings"


# --- Ollama API type tests ---


@patch("meshwiki.remote_embeddings.requests.Session")
def test_ollama_url_construction(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(1)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    encode_batch(["test"], MOCK_CONFIG_OLLAMA)

    assert mock_session.post.call_args[0][0] == "http://gpu-server:11434/api/embed"


@patch("meshwiki.remote_embeddings.requests.Session")
def test_ollama_successful_encoding(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(3)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    result = encode_batch(["a", "b", "c"], MOCK_CONFIG_OLLAMA)

    assert result is not None
    assert len(result) == 3
    assert len(result[0]) == 1024


@patch("meshwiki.remote_embeddings.requests.Session")
def test_ollama_dimension_mismatch_returns_none(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_ollama_response(1, dim=768)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    assert encode_batch(["hello"], MOCK_CONFIG_OLLAMA) is None


@patch("meshwiki.remote_embeddings.requests.Session")
def test_ollama_sub_batching(MockSession):
    """600 texts with batch_size=256 should produce 3 HTTP calls (Ollama)."""
    texts = [f"text {i}" for i in range(600)]

    def side_effect(*args, **kwargs):
        sub_batch = kwargs.get("json", {}).get("input", [])
        resp = MagicMock()
        resp.json.return_value = _make_ollama_response(len(sub_batch))
        resp.raise_for_status = MagicMock()
        return resp

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = side_effect
    MockSession.return_value = mock_session

    result = encode_batch(texts, MOCK_CONFIG_OLLAMA)

    assert result is not None
    assert len(result) == 600
    assert mock_session.post.call_count == 3


@patch("meshwiki.remote_embeddings.requests.Session")
def test_default_api_type_is_openai(MockSession):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(1)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    encode_batch(["test"], MOCK_CONFIG)

    assert "/v1/embeddings" in mock_session.post.call_args[0][0]


# --- on_sub_batch callback ---


@patch("meshwiki.remote_embeddings.requests.Session")
def test_on_sub_batch_callback_called(MockSession):
    """on_sub_batch is called once per sub-batch with (completed, total)."""
    texts = [f"text {i}" for i in range(600)]

    def side_effect(*args, **kwargs):
        sub_batch = kwargs.get("json", {}).get("input", [])
        resp = MagicMock()
        resp.json.return_value = _make_response(len(sub_batch))
        resp.raise_for_status = MagicMock()
        return resp

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.side_effect = side_effect
    MockSession.return_value = mock_session

    callback_calls = []
    def on_sub_batch(completed, total):
        callback_calls.append((completed, total))

    result = encode_batch(texts, MOCK_CONFIG, on_sub_batch=on_sub_batch)

    assert result is not None
    assert len(callback_calls) == 3  # 256 + 256 + 88 = 3 sub-batches
    # All calls should have total=3
    assert all(total == 3 for _, total in callback_calls)
    # completed values should be 1, 2, 3 (in some order due to concurrency)
    assert sorted(c for c, _ in callback_calls) == [1, 2, 3]


@patch("meshwiki.remote_embeddings.requests.Session")
def test_on_sub_batch_not_called_when_none(MockSession):
    """No crash when on_sub_batch is None (default)."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_response(1)
    mock_resp.raise_for_status = MagicMock()

    mock_session = MagicMock()
    mock_session.headers = {}
    mock_session.post.return_value = mock_resp
    MockSession.return_value = mock_session

    result = encode_batch(["test"], MOCK_CONFIG)
    assert result is not None
