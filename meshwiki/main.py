"""MeshWiki entry point — orchestrates all components."""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
from logging.handlers import RotatingFileHandler
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from meshwiki import kiwix_search
from meshwiki import config
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


def _check_ollama(config: dict) -> bool:
    """Verify that Ollama is reachable."""
    url = config["ollama"]["base_url"]
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def _index_exists(config: dict) -> bool:
    """Check if a usable ChromaDB index exists.

    Verifies that the collection exists, has documents, and uses
    embeddings compatible with the current model configuration.
    """
    db_path = Path(config["vectordb"]["path"])
    if not db_path.exists():
        return False

    import chromadb
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("wikipedia")
        if collection.count() == 0:
            return False
        # Verify embedding dimensions match current model
        expected_dim = config["embeddings"].get("truncate_dim") or config["embeddings"].get("embedding_dim")
        if expected_dim:
            sample = collection.peek(limit=1)
            if sample["embeddings"]:
                actual_dim = len(sample["embeddings"][0])
                if actual_dim != expected_dim:
                    logger.warning(
                        "Index incompatible : dimension %d (attendu %d) — réindexation nécessaire",
                        actual_dim, expected_dim,
                    )
                    return False
        return True
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


def _wants_update() -> bool:
    """Check if the user requested an update check via command-line argument."""
    return any(arg in ("/update", "-update", "--update") for arg in sys.argv[1:])


def _wants_offline() -> bool:
    """Check if the user requested offline mode (no downloads)."""
    return any(arg in ("/offline", "-offline", "--offline") for arg in sys.argv[1:])


def _wants_noindex() -> bool:
    """Check if the user requested to ignore the ChromaDB index."""
    return any(arg in ("/noindex", "-noindex", "--noindex") for arg in sys.argv[1:])


def _wants_nowiki() -> bool:
    """Check if the user requested to ignore the ZIM file."""
    return any(arg in ("/nowiki", "-nowiki", "--nowiki") for arg in sys.argv[1:])


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
            logger.info("ZIM téléchargé : %s — lancez avec --index pour indexer", zim_path.name)

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
        cfg = config.load_config()
    except FileNotFoundError:
        logger.error("config.yaml introuvable")
        sys.exit(1)

    # File logging with rotation
    log_cfg = cfg.get("logging", {})
    log_file = log_cfg.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_cfg.get("max_size_mb", 5) * 1024 * 1024,
            backupCount=log_cfg.get("backup_count", 3),
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
        logger.info("Logs enregistrés dans %s", log_path)

    # Check Ollama
    if not _check_ollama(cfg):
        logger.warning("Ollama n'est pas accessible à %s", cfg["ollama"]["base_url"])
        logger.warning("Le service démarrera mais les réponses LLM ne fonctionneront pas.")

    # Initialize rate limiter
    rl_config = cfg["rate_limiting"]
    rate_limiter = RateLimiter(
        max_requests=rl_config["max_requests"],
        window_seconds=rl_config["window_seconds"],
    )

    # Connect to Meshtastic
    bridge = MeshtasticBridge(rate_limiter)

    # Apply --noindex / --nowiki flags
    noindex = _wants_noindex()
    nowiki = _wants_nowiki()
    offline = _wants_offline()

    if noindex:
        from meshwiki import rag
        rag.set_force_unavailable(True)
        logger.info("--noindex : index ChromaDB désactivé")

    if nowiki:
        kiwix_search.set_disabled(True)
        logger.info("--nowiki : fichier ZIM désactivé")

    # Check/create index
    if not noindex and _index_exists(cfg):
        logger.info("Index ChromaDB disponible")
        # Update check only if --update is passed
        if _wants_update() or _wants_indexation():
            if _wants_indexation():
                logger.info("Mise à jour + indexation demandée, lancement en arrière-plan...")
                _run_background_update(cfg)
            else:
                logger.info("Mise à jour demandée, téléchargement du ZIM en arrière-plan...")
                _run_background_download(cfg)
    else:
        if not nowiki:
            zim_path = _find_existing_zim(cfg)
            if zim_path:
                kiwix_search.set_zim_path(zim_path)
                logger.info("Recherche Kiwix activée comme fallback (%s)", zim_path.name)
            elif not offline:
                logger.info("Aucun ZIM trouvé — téléchargement en arrière-plan...")
                _run_background_download(cfg)
            else:
                logger.info("Aucun index ni ZIM trouvé — mode offline, le LLM répondra seul.")
        else:
            if not offline and not noindex:
                logger.info("Le LLM répondra seul (ZIM désactivé, pas d'index)")

        if not noindex and _wants_indexation():
            logger.info("Indexation demandée, lancement en arrière-plan...")
            _run_background_update(cfg)
        elif not noindex:
            logger.info("Lancez avec --index pour créer l'index ChromaDB")

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

    # Start update scheduler (disabled in offline mode)
    if cfg["updater"]["enabled"] and not _wants_offline():
        stop_event = _start_update_scheduler(cfg)

    # Connect and run
    logger.info("MeshWiki opérationnel. En attente de messages...")
    bridge.reconnect_loop()
