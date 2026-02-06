"""Tests for meshwiki.llm — Ollama LLM interface."""

from unittest.mock import MagicMock, patch

import requests

import meshwiki.llm as llm_module
from meshwiki.llm import generate


MOCK_CONFIG = {
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "mistral",
        "temperature": 0.1,
        "max_tokens": 300,
    }
}


@patch.object(llm_module, "_config", MOCK_CONFIG)
@patch("meshwiki.llm.requests.post")
def test_successful_generation(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Paris est la capitale de la France."}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    result = generate("Tu es un assistant.", "Quelle est la capitale de la France ?")

    assert result == "Paris est la capitale de la France."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
    assert payload["model"] == "mistral"
    assert payload["stream"] is False


@patch.object(llm_module, "_config", MOCK_CONFIG)
@patch("meshwiki.llm.requests.post")
def test_timeout_returns_french_error(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout()

    result = generate("system", "user")

    assert "pas répondu à temps" in result


@patch.object(llm_module, "_config", MOCK_CONFIG)
@patch("meshwiki.llm.requests.post")
def test_connection_error_returns_unavailable(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError()

    result = generate("system", "user")

    assert "indisponible" in result


@patch.object(llm_module, "_config", MOCK_CONFIG)
@patch("meshwiki.llm.requests.post")
def test_http_error_returns_generic_error(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    mock_post.return_value = mock_response

    result = generate("system", "user")

    assert "Erreur" in result


@patch.object(llm_module, "_config", MOCK_CONFIG)
@patch("meshwiki.llm.requests.post")
def test_payload_structure(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    generate("sys prompt", "user prompt")

    payload = mock_post.call_args[1]["json"]
    assert payload["system"] == "sys prompt"
    assert payload["prompt"] == "user prompt"
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 300
