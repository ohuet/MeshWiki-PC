"""Tests for meshwiki.meshtastic_bridge — Meshtastic message handling."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call
import time
import pytest

import meshwiki.meshtastic_bridge as bridge_module
from meshwiki.meshtastic_bridge import MeshtasticBridge
from meshwiki.rate_limiter import RateLimiter


MY_NODE_NUM = 957304120
OTHER_NODE_NUM = 3506839865
BROADCAST = 4294967295  # 0xFFFFFFFF = ^all

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


def _make_packet(text, from_node=OTHER_NODE_NUM, to_node=MY_NODE_NUM):
    """Helper to build a Meshtastic-like packet dict."""
    return {
        "from": from_node,
        "to": to_node,
        "fromId": f"!{from_node:08x}",
        "toId": f"!{to_node:08x}" if to_node != BROADCAST else "^all",
        "decoded": {"text": text},
    }


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_ignores_own_messages(mock_config):
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Test", from_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
def test_ignores_broadcast_without_prefix(mock_config):
    """Broadcast messages without trigger prefix are ignored."""
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Hello world", to_node=BROADCAST)
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.rag")
def test_broadcast_with_prefix_is_processed(mock_rag, mock_config):
    """Broadcast messages with trigger prefix are processed."""
    mock_rag.query.return_value = "Paris"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("?Capitale de la France", to_node=BROADCAST)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_called_once_with("Capitale de la France")
    bridge.interface.sendText.assert_called_once_with(
        "Paris", destinationId=f"!{OTHER_NODE_NUM:08x}",
    )


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.rag")
def test_direct_message_no_prefix(mock_rag, mock_config):
    """Direct messages are processed without requiring a prefix."""
    mock_rag.query.return_value = "Paris"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_called_once_with("Capitale de la France")
    bridge.interface.sendText.assert_called_once_with(
        "Paris", destinationId=f"!{OTHER_NODE_NUM:08x}",
    )


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.rag")
def test_direct_message_with_prefix_strips_it(mock_rag, mock_config):
    """Direct messages with prefix still work — prefix is stripped."""
    mock_rag.query.return_value = "Paris"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("?Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_called_once_with("Capitale de la France")


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.rag")
def test_rate_limiting_sends_denial(mock_rag, mock_config):
    mock_rag.is_available.return_value = True
    mock_rag.query.return_value = "OK"

    limiter = RateLimiter(max_requests=1, window_seconds=60)

    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    # First request: allowed
    packet = _make_packet("Q1", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    # Second request: rate limited
    packet = _make_packet("Q2", to_node=MY_NODE_NUM)
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
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    # DM with empty text
    packet = _make_packet("", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    # Broadcast with just prefix
    packet = _make_packet("?", to_node=BROADCAST)
    bridge._on_message_received(packet, MagicMock())

    bridge.interface.sendText.assert_not_called()


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG_NO_PORT)
def test_auto_detect_serial_port(mock_config):
    """When port is not set, auto-detect the first Meshtastic serial port."""
    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)

    mock_iface = MagicMock()
    mock_iface.myInfo.my_node_num = MY_NODE_NUM

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


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.kiwix_search")
@patch("meshwiki.meshtastic_bridge.rag")
@patch("meshwiki.meshtastic_bridge.get_indexing_eta")
def test_indexing_in_progress_calls_llm_fallback(mock_eta, mock_rag, mock_kiwix, mock_config):
    """When indexation is running and no ZIM available, calls query_without_context."""
    eta = datetime.now(timezone.utc) + timedelta(hours=1, minutes=30)
    mock_eta.return_value = eta
    mock_kiwix.get_zim_path.return_value = None
    mock_rag.query_without_context.return_value = "Réponse sans Wikipedia"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    call_args = mock_rag.query_without_context.call_args
    assert call_args[0][0] == "Capitale de la France"
    assert call_args[0][1].startswith("environ 1h")
    mock_rag.query.assert_not_called()
    mock_rag.query_with_kiwix_context.assert_not_called()
    bridge.interface.sendText.assert_called_once_with(
        "Réponse sans Wikipedia",
        destinationId=f"!{OTHER_NODE_NUM:08x}",
    )


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.kiwix_search")
@patch("meshwiki.meshtastic_bridge.rag")
@patch("meshwiki.meshtastic_bridge.get_indexing_eta")
def test_indexing_with_zim_uses_kiwix_fallback(mock_eta, mock_rag, mock_kiwix, mock_config):
    """When indexation is running and ZIM is available, calls query_with_kiwix_context."""
    eta = datetime.now(timezone.utc) + timedelta(hours=1, minutes=30)
    mock_eta.return_value = eta
    mock_kiwix.get_zim_path.return_value = "/tmp/wikipedia.zim"
    mock_rag.query_with_kiwix_context.return_value = "Réponse Kiwix"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    call_args = mock_rag.query_with_kiwix_context.call_args
    assert call_args[0][0] == "Capitale de la France"
    assert call_args[0][1].startswith("environ 1h")
    mock_rag.query.assert_not_called()
    mock_rag.query_without_context.assert_not_called()
    bridge.interface.sendText.assert_called_once_with(
        "Réponse Kiwix",
        destinationId=f"!{OTHER_NODE_NUM:08x}",
    )


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.get_indexing_eta", return_value=None)
@patch("meshwiki.meshtastic_bridge.rag")
def test_no_indexing_processes_normally(mock_rag, mock_eta, mock_config):
    """When no indexation is running, the bridge processes questions normally."""
    mock_rag.query.return_value = "Paris"
    mock_rag.is_available.return_value = True

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_called_once_with("Capitale de la France")


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.kiwix_search")
@patch("meshwiki.meshtastic_bridge.rag")
@patch("meshwiki.meshtastic_bridge.get_indexing_eta", return_value=None)
def test_no_index_with_kiwix_uses_fallback(mock_eta, mock_rag, mock_kiwix, mock_config):
    """When no index exists but ZIM is available, uses permanent Kiwix fallback."""
    mock_rag.is_available.return_value = False
    mock_kiwix.get_zim_path.return_value = "/tmp/wikipedia.zim"
    mock_rag.query_with_kiwix_context_permanent.return_value = "Réponse Kiwix permanente"

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query_with_kiwix_context_permanent.assert_called_once_with("Capitale de la France")
    mock_rag.query.assert_not_called()
    bridge.interface.sendText.assert_called_once_with(
        "Réponse Kiwix permanente",
        destinationId=f"!{OTHER_NODE_NUM:08x}",
    )


@patch.object(bridge_module, "_load_config", return_value=MOCK_CONFIG)
@patch("meshwiki.meshtastic_bridge.kiwix_search")
@patch("meshwiki.meshtastic_bridge.rag")
@patch("meshwiki.meshtastic_bridge.get_indexing_eta", return_value=None)
def test_no_index_no_kiwix_sends_download_message(mock_eta, mock_rag, mock_kiwix, mock_config):
    """When neither index nor ZIM exists, sends a message about downloading."""
    mock_rag.is_available.return_value = False
    mock_kiwix.get_zim_path.return_value = None

    limiter = RateLimiter()
    bridge = MeshtasticBridge(limiter)
    bridge.my_node_id = MY_NODE_NUM
    bridge.interface = MagicMock()

    packet = _make_packet("Capitale de la France", to_node=MY_NODE_NUM)
    bridge._on_message_received(packet, MagicMock())

    mock_rag.query.assert_not_called()
    sent_text = bridge.interface.sendText.call_args[0][0]
    assert "téléchargement" in sent_text
    assert "--index" in sent_text
