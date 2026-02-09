"""MeshWiki entry point — orchestrates all components."""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

from meshwiki import kiwix_search
from meshwiki.meshtastic_bridge import MeshtasticBridge
from meshwiki.rate_limiter import RateLimiter
from meshwiki.wikipedia_updater import WikipediaUpdater, LAST_UPDATE_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOCK_FILE = Path("data/meshwiki.lock")
_lock_fd = None


def _check_already_running() -> None:
    """Exit if another MeshWiki instance is already running (OS-level file lock)."""
    global _lock_fd
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _lock_fd = open(LOCK_FILE, "w")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.error("MeshWiki est déjà en cours d'exécution")
        sys.exit(1)


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _check_ollama(config: dict) -> bool:
    """Verify that Ollama is reachable."""
    url = config["ollama"]["base_url"]
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _index_exists(config: dict) -> bool:
    """Check if a ChromaDB index exists."""
    db_path = Path(config["vectordb"]["path"])
    if not db_path.exists():
        return False

    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("wikipedia")
        return collection.count() > 0
    except Exception:
        return False


def _is_update_due(config: dict) -> bool:
    """Check if a scheduled update is due based on last_update.json."""
    if not LAST_UPDATE_FILE.exists():
        return True

    try:
        with open(LAST_UPDATE_FILE) as f:
            data = json.load(f)
        last_update = datetime.fromisoformat(data["last_update"])
        interval = timedelta(days=config["updater"]["interval_days"])
        return datetime.now() - last_update > interval
    except (KeyError, ValueError):
        return True


def _wants_indexation() -> bool:
    """Check if the user requested indexation via command-line argument."""
    return any(arg in ("/index", "-index", "--index") for arg in sys.argv[1:])


def _find_existing_zim(config: dict) -> Path | None:
    """Find an existing .zim file in the temp directory."""
    temp_dir = Path(config["updater"]["temp_dir"])
    if not temp_dir.exists():
        return None
    zim_files = sorted(temp_dir.glob("*.zim"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zim_files[0] if zim_files else None


def _run_background_download(config: dict) -> None:
    """Download a ZIM file in background (without indexation).

    When the download completes, activates the Kiwix fallback.
    """
    def _download():
        try:
            import os
            if hasattr(os, "nice"):
                os.nice(10)
        except OSError:
            pass

        updater = WikipediaUpdater()
        zim_path = updater.download_dump()
        if zim_path is not None:
            kiwix_search.set_zim_path(zim_path)
            logger.info("ZIM téléchargé, fallback Kiwix activé : %s", zim_path.name)

    thread = threading.Thread(target=_download, name="zim-downloader", daemon=True)
    thread.start()


def _run_background_update(config: dict) -> None:
    """Run Wikipedia update in a background thread with lowered priority."""
    def _update():
        try:
            import os
            # Lower thread priority on Unix
            if hasattr(os, "nice"):
                os.nice(10)
        except OSError:
            pass

        updater = WikipediaUpdater()
        updater.run_update()

    thread = threading.Thread(target=_update, name="wikipedia-updater", daemon=True)
    thread.start()


def _start_update_scheduler(config: dict) -> threading.Event:
    """Start a periodic download scheduler in a background thread.

    Only downloads new ZIM files — never triggers indexation automatically.
    Returns a stop event to cancel the scheduler.
    """
    stop_event = threading.Event()
    allowed_hours = config["updater"]["allowed_hours"]

    def _scheduler():
        while not stop_event.is_set():
            now = datetime.now()
            start_hour, end_hour = allowed_hours

            if start_hour <= now.hour < end_hour and _is_update_due(config):
                logger.info("Téléchargement planifié en cours...")
                updater = WikipediaUpdater()
                zim_path = updater.download_dump()
                if zim_path is not None:
                    kiwix_search.set_zim_path(zim_path)
                    logger.info(
                        "Nouveau ZIM téléchargé : %s — lancez avec --index pour réindexer",
                        zim_path.name,
                    )

            # Check every hour
            stop_event.wait(3600)

    thread = threading.Thread(target=_scheduler, name="update-scheduler", daemon=True)
    thread.start()
    return stop_event


def main() -> None:
    """Main entry point for MeshWiki."""
    logger.info("Démarrage de MeshWiki...")
    _check_already_running()

    # Load configuration
    try:
        config = _load_config()
    except FileNotFoundError:
        logger.error("config.yaml introuvable")
        sys.exit(1)

    # Check Ollama
    if not _check_ollama(config):
        logger.warning("Ollama n'est pas accessible à %s", config["ollama"]["base_url"])
        logger.warning("Le service démarrera mais les réponses LLM ne fonctionneront pas.")

    # Initialize rate limiter
    rl_config = config["rate_limiting"]
    rate_limiter = RateLimiter(
        max_requests=rl_config["max_requests"],
        window_seconds=rl_config["window_seconds"],
    )

    # Connect to Meshtastic
    bridge = MeshtasticBridge(rate_limiter)

    # Check/create index
    if _index_exists(config):
        # Index ready — check for scheduled update (download only, never auto-reindex)
        if config["updater"]["enabled"] and config["updater"]["check_on_startup"]:
            if _is_update_due(config):
                if _wants_indexation():
                    logger.info("Mise à jour + indexation demandée, lancement en arrière-plan...")
                    _run_background_update(config)
                else:
                    logger.info("Mise à jour planifiée, téléchargement du ZIM en arrière-plan...")
                    _run_background_download(config)
    else:
        zim_path = _find_existing_zim(config)
        if zim_path:
            kiwix_search.set_zim_path(zim_path)
            logger.info("Pas d'index — recherche Kiwix activée (%s)", zim_path.name)
        else:
            logger.info("Aucun ZIM trouvé, téléchargement en arrière-plan...")
            _run_background_download(config)

        if _wants_indexation():
            logger.info("Indexation demandée, lancement en arrière-plan...")
            _run_background_update(config)
        else:
            logger.info("Lancez avec --index pour démarrer l'indexation ChromaDB")

    # Graceful shutdown
    stop_event = None

    def _signal_handler(sig, frame):
        logger.info("Arrêt de MeshWiki...")
        bridge.close()
        if stop_event:
            stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start update scheduler
    if config["updater"]["enabled"]:
        stop_event = _start_update_scheduler(config)

    # Connect and run
    logger.info("MeshWiki opérationnel. En attente de messages...")
    bridge.reconnect_loop()
