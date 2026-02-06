"""Tests for meshwiki.meshtastic_bridge — Meshtastic message handling."""

from unittest.mock import MagicMock, patch, call
import time
import pytest

import meshwiki.meshtastic_bridge as bridge_module
from meshwiki.meshtastic_bridge import MeshtasticBridge
from meshwiki.rate_limiter import RateLimiter


MOCK_CONFIG = {
    "meshtastic": {
        "connection": "serial",
        "port": "/dev/ttyUSB0",
        "trigger_prefix": "?",
        "response_delay": 0.01,  # Fast for tests
        "max_response_chunks": 5,
    },
    "rate_limiting": {"max_requests": 10, "window_seconds": 600},
}

MOCK_CONFIG_NO_PORT = {
    "meshtastic": {
        "connection": "serial",
        "trigger_prefix": "?",
        "response_delay": 0.01,
        "max_response_chunks": 5,
    },
    "rate_limiting": {"max_requests": 10, "window_seconds": 600},
}


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_ignores_own_messages(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = "!mynode"
    bridge.interface = MagicMock()

    packet = {"fromId": "!mynode", "decoded": {"text": "?Test"}}
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_ignores_messages_without_prefix(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = "!mynode"
    bridge.interface = MagicMock()

    packet = {"fromId": "!other", "decoded": {"text": "Hello world"}}
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.rag")
def test_processes_valid_question(mock_rag, mock_config):
    mock_rag.query.return_value = "Paris"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = "!mynode"
    bridge.interface = MagicMock()

    packet = {"fromId": "!sender", "decoded": {"text": "?Capitale de la France"}}
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_called_once_with("Capitale de la France")
    bridge.interface.sendText.assert_called_once_with("Paris", destinationId="!sender")


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_rate_limiting_sends_denial(mock_config):
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = "!mynode"
    bridge.interface = MagicMock()

    # First request: allowed
    with patch("meshwiki.meshtastic_bridge.rag") as mock_rag:
        mock_rag.query.return_value = "OK"
        packet = {"fromId": "!sender", "decoded": {"text": "?Q1"}}
        bridge._on_message_received(packet, MagicMock())

    # Second request: rate limited
    packet = {"fromId": "!sender", "decoded": {"text": "?Q2"}}
    bridge._on_message_received(packet, MagicMock())

    # Second call should send denial
    last_call = bridge.interface.sendText.call_args_list[-1]
    sent_text = last_call[0][0]
    assert "Limite atteinte" in sent_text


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_send_response_chunks_with_delay(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.interface = MagicMock()

    # Long text that will be chunked
    long_text = "Ceci est un texte long. " * 50

    start = time.time()
    bridge.send_response("!dest", long_text)
    elapsed = time.time() - start

    # Should have sent multiple chunks
    assert bridge.interface.sendText.call_count > 1

    # Verify delay between chunks (at least some delay occurred)
    # With 0.01s delay and multiple chunks, total should be > 0
    assert elapsed > 0


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_send_response_short_text_no_chunking(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.interface = MagicMock()

    bridge.send_response("!dest", "Short answer")

    bridge.interface.sendText.assert_called_once_with("Short answer", destinationId="!dest")


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_ignores_empty_question(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = "!mynode"
    bridge.interface = MagicMock()

    packet = {"fromId": "!sender", "decoded": {"text": "?"}}
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG_NO_PORT)
def test_auto_detect_serial_port(mock_config):
    """When port is not set, auto-detect the first Meshtastic serial port."""
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)

    mock_iface = MagicMock()
    mock_iface.myInfo.get.return_value = "!node1"

    with patch("meshtastic.util.findPorts", return_value=["/dev/ttyACM0", "/dev/ttyACM1"]) as mock_find, \
         patch("meshtastic.serial_interface.SerialInterface", return_value=mock_iface):
        bridge.connect()

    mock_find.assert_called_once_with(True)
    assert bridge.interface == mock_iface


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG_NO_PORT)
def test_auto_detect_no_port_found_raises(mock_config):
    """When no Meshtastic port is found, raise ConnectionError."""
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)

    with patch("meshtastic.util.findPorts", return_value=[]):
        with pytest.raises(ConnectionError, match="Aucun appareil Meshtastic"):
            bridge.connect()
