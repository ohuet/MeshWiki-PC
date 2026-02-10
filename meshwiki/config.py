"""Centralized configuration loader with local override support.

Loads ``config.yaml`` as the base configuration, then deep-merges
``config.local.yaml`` on top if the file exists.  This allows keeping
secrets (API keys, URLs) out of version control.
"""

from pathlib import Path

import yaml

CONFIG_FILE = Path("config.yaml")
LOCAL_CONFIG_FILE = Path("config.local.yaml")

_config: dict | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config() -> dict:
    """Return the merged configuration (cached after first call)."""
    global _config
    if _config is not None:
        return _config

    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if LOCAL_CONFIG_FILE.exists():
        with open(LOCAL_CONFIG_FILE, encoding="utf-8") as f:
            local = yaml.safe_load(f)
        if local:
            _deep_merge(config, local)

    _config = config
    return _config


def reset() -> None:
    """Clear the cached config (useful for tests)."""
    global _config
    _config = None
