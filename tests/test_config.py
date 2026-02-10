"""Tests for meshwiki.config — centralized config loader with local override."""

from unittest.mock import patch

import meshwiki.config as config_module
from meshwiki.config import load_config, reset, _deep_merge


def setup_function():
    reset()


def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = _deep_merge(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_deep_merge_nested():
    base = {"embeddings": {"model": "bge-m3", "remote": {"base_url": "http://default"}}}
    override = {"embeddings": {"remote": {"base_url": "http://custom", "api_key": "sk-123"}}}
    result = _deep_merge(base, override)
    assert result["embeddings"]["model"] == "bge-m3"
    assert result["embeddings"]["remote"]["base_url"] == "http://custom"
    assert result["embeddings"]["remote"]["api_key"] == "sk-123"


def test_deep_merge_override_replaces_non_dict():
    base = {"a": {"b": 1}}
    override = {"a": "replaced"}
    result = _deep_merge(base, override)
    assert result["a"] == "replaced"


@patch.object(config_module, "LOCAL_CONFIG_FILE")
@patch.object(config_module, "CONFIG_FILE")
def test_load_without_local(mock_cfg, mock_local):
    """When config.local.yaml doesn't exist, only config.yaml is loaded."""
    import yaml
    mock_cfg.__fspath__ = lambda self: "config.yaml"
    mock_local.exists.return_value = False

    base_config = {"embeddings": {"model": "bge-m3"}}

    from unittest.mock import mock_open
    with patch("builtins.open", mock_open(read_data=yaml.dump(base_config))):
        result = load_config()

    assert result["embeddings"]["model"] == "bge-m3"


@patch.object(config_module, "LOCAL_CONFIG_FILE")
@patch.object(config_module, "CONFIG_FILE")
def test_load_with_local_override(mock_cfg, mock_local):
    """config.local.yaml values override config.yaml values."""
    import yaml
    mock_cfg.__fspath__ = lambda self: "config.yaml"
    mock_local.exists.return_value = True
    mock_local.__fspath__ = lambda self: "config.local.yaml"

    base_config = {"embeddings": {"model": "bge-m3", "remote": {"base_url": "http://default"}}}
    local_config = {"embeddings": {"remote": {"base_url": "http://custom", "api_key": "sk-secret"}}}

    call_count = 0

    def mock_open_fn(path, **kwargs):
        nonlocal call_count
        from unittest.mock import mock_open
        call_count += 1
        if call_count == 1:
            return mock_open(read_data=yaml.dump(base_config))()
        return mock_open(read_data=yaml.dump(local_config))()

    with patch("builtins.open", side_effect=mock_open_fn):
        result = load_config()

    assert result["embeddings"]["model"] == "bge-m3"
    assert result["embeddings"]["remote"]["base_url"] == "http://custom"
    assert result["embeddings"]["remote"]["api_key"] == "sk-secret"


def test_load_config_is_cached():
    """Second call returns cached result without re-reading files."""
    with patch.object(config_module, "LOCAL_CONFIG_FILE") as mock_local, \
         patch.object(config_module, "CONFIG_FILE") as mock_cfg:
        import yaml
        from unittest.mock import mock_open
        mock_cfg.__fspath__ = lambda self: "config.yaml"
        mock_local.exists.return_value = False

        with patch("builtins.open", mock_open(read_data=yaml.dump({"a": 1}))):
            first = load_config()
            second = load_config()

    assert first is second


def test_reset_clears_cache():
    """After reset(), load_config re-reads files."""
    with patch.object(config_module, "LOCAL_CONFIG_FILE") as mock_local, \
         patch.object(config_module, "CONFIG_FILE") as mock_cfg:
        import yaml
        from unittest.mock import mock_open
        mock_cfg.__fspath__ = lambda self: "config.yaml"
        mock_local.exists.return_value = False

        with patch("builtins.open", mock_open(read_data=yaml.dump({"a": 1}))):
            first = load_config()

    reset()

    with patch.object(config_module, "LOCAL_CONFIG_FILE") as mock_local, \
         patch.object(config_module, "CONFIG_FILE") as mock_cfg:
        import yaml
        from unittest.mock import mock_open
        mock_cfg.__fspath__ = lambda self: "config.yaml"
        mock_local.exists.return_value = False

        with patch("builtins.open", mock_open(read_data=yaml.dump({"a": 2}))):
            second = load_config()

    assert first["a"] == 1
    assert second["a"] == 2
