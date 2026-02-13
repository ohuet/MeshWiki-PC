"""Interface with Ollama local LLM."""

import logging

import requests

from meshwiki import config

logger = logging.getLogger(__name__)


def generate(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Ollama and return the generated text.

    Returns a French error message if Ollama is unavailable.
    """
    cfg = config.load_config()
    ollama = cfg["ollama"]

    url = f"{ollama['base_url']}/api/generate"
    payload = {
        "model": ollama["model"],
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": ollama["temperature"],
            "num_predict": ollama["max_tokens"],
        },
    }

    logger.info("LLM system: %s", system_prompt)
    logger.info("LLM prompt: %s", user_prompt)

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        logger.info("LLM response: %s", answer)
        return answer
    except requests.exceptions.Timeout:
        logger.error("Ollama timeout after 300s")
        return "Erreur : le modèle LLM n'a pas répondu à temps."
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama at %s", ollama["base_url"])
        return "Service LLM indisponible."
    except requests.exceptions.RequestException as e:
        logger.error("Ollama request error: %s", e)
        return "Erreur lors de la génération de la réponse."
