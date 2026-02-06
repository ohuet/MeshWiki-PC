"""Meshtastic communication bridge — listen for questions, send chunked answers."""

import logging
import time
import threading

import yaml
from pubsub import pub

from meshwiki.chunker import split_message
from meshwiki.rate_limiter import RateLimiter
from meshwiki.wikipedia_indexer import get_indexing_eta
from meshwiki import rag

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


class MeshtasticBridge:
    """Connects to a Meshtastic radio and handles question/answer flow."""

    def __init__(self, rate_limiter: RateLimiter):
        self.config = _load_config()
        self.mesh_config = self.config["meshtastic"]
        self.rate_limiter = rate_limiter
        self.interface = None
        self.my_node_id = None
        self._running = True

    def connect(self) -> None:
        """Connect to the Meshtastic module via serial or BLE."""
        connection_type = self.mesh_config["connection"]
        port = self.mesh_config.get("port")

        if port is None and connection_type == "serial":
            from meshtastic.util import findPorts
            ports = findPorts(True)
            if not ports:
                raise ConnectionError(
                    "Aucun appareil Meshtastic détecté sur les ports série"
                )
            port = ports[0]
            logger.info("Port Meshtastic auto-détecté : %s", port)

        logger.info("Connecting to Meshtastic (%s: %s)...", connection_type, port)

        if connection_type == "serial":
            from meshtastic.serial_interface import SerialInterface
            self.interface = SerialInterface(port)
        elif connection_type == "ble":
            from meshtastic.ble_interface import BLEInterface
            self.interface = BLEInterface(port)
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")

        self.my_node_id = self.interface.myInfo.my_node_num
        pub.subscribe(self._on_message_received, "meshtastic.receive.text")
        logger.info(
            "Connected to Meshtastic: %s (%s) — node %s",
            self.interface.getLongName(),
            self.interface.getShortName(),
            self.my_node_id,
        )
        if self.interface.metadata:
            md = self.interface.metadata
            logger.info(
                "Module: %s — firmware %s",
                md.hw_model, md.firmware_version,
            )

    def _on_message_received(self, packet, interface) -> None:
        """Handle incoming text messages."""
        sender = packet.get("fromId")
        text = packet.get("decoded", {}).get("text", "")

        # Ignore our own messages
        if packet.get("from") == self.my_node_id:
            return

        # Direct messages: treat full text as question
        # Broadcasts: require trigger prefix
        is_direct = packet.get("to") == self.my_node_id
        prefix = self.mesh_config["trigger_prefix"]

        if is_direct:
            question = text.removeprefix(prefix).strip()
        elif text.startswith(prefix):
            question = text[len(prefix):].strip()
        else:
            return

        if not question:
            return

        logger.info("Question from %s: %s", sender, question)

        # Check if indexation is in progress
        eta = get_indexing_eta()
        if eta is not None:
            eta_str = eta.strftime("%H:%M")
            self.send_response(sender, f"Données en cours d'initialisation. Fin prévue à {eta_str}.")
            return

        # Check rate limit
        allowed, denial_msg = self.rate_limiter.check(str(sender))
        if not allowed:
            logger.info("Rate limited %s: %s", sender, denial_msg)
            self.send_response(sender, denial_msg)
            return

        # Process through RAG pipeline
        try:
            answer = rag.query(question)
        except Exception as e:
            logger.error("RAG error: %s", e)
            answer = "Erreur lors du traitement de la question."

        logger.info("Answer to %s: %s", sender, answer)
        self.send_response(sender, answer)

    def send_response(self, destination_id: str, text: str) -> None:
        """Send a response, chunking if necessary with delays between chunks."""
        if not self.interface:
            logger.error("Cannot send: not connected")
            return

        max_bytes = 220
        delay = self.mesh_config["response_delay"]

        if len(text.encode("utf-8")) <= max_bytes:
            chunks = [text]
        else:
            chunks = split_message(text, max_bytes)

        for i, chunk in enumerate(chunks):
            try:
                self.interface.sendText(chunk, destinationId=destination_id)
                logger.info("Sent chunk %d/%d to %s", i + 1, len(chunks), destination_id)
            except Exception as e:
                logger.error("Failed to send chunk %d to %s: %s", i + 1, destination_id, e)
                break

            if i < len(chunks) - 1:
                time.sleep(delay)

    def reconnect_loop(self) -> None:
        """Auto-reconnect loop: retry every 30 seconds on disconnect."""
        while self._running:
            try:
                if self.interface is None:
                    self.connect()
                time.sleep(1)
            except Exception as e:
                logger.exception("Connection lost. Retrying in 30s...")
                if self.interface:
                    try:
                        self.interface.close()
                    except Exception:
                        pass
                self.interface = None
                time.sleep(30)

    def close(self) -> None:
        """Disconnect from Meshtastic."""
        self._running = False
        if self.interface:
            try:
                self.interface.close()
            except Exception:
                pass
            self.interface = None
