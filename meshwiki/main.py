"""MeshWiki entry point — orchestrates all components."""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import logging
from logging.handlers import RotatingFileHandler
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from meshwiki import kiwix_search
from meshwiki import config, collection_state
from meshwiki.meshtastic_bridge import MeshtasticBridge
from meshwiki.rate_limiter import RateLimiter
from meshwiki.wikipedia_updater import WikipediaUpdater, LAST_UPDATE_FILE
from meshwiki.zim_discovery import discover_zims

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


def _cleanup_inactive_db() -> None:
    """Remove the inactive ChromaDB database left over from a previous swap.

    Called at startup before any ChromaDB connection, so no file handles
    are open and shutil.rmtree succeeds even on Windows.
    Skips cleanup when a checkpoint exists (partial indexation to resume).
    """
    from meshwiki.wikipedia_indexer import has_checkpoint
    if has_checkpoint():
        return

    inactive_path = Path(collection_state.get_inactive_db_path())
    if inactive_path.exists():
        try:
            shutil.rmtree(inactive_path)
            logger.info("Ancien index nettoyé : %s", inactive_path)
        except OSError as e:
            logger.warning("Impossible de supprimer %s : %s", inactive_path, e)


def _restart_app() -> None:
    """Restart the application after a successful reindexation."""
    logger.info("Redémarrage automatique après réindexation...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


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
    embeddings compatible with the current model configuration —
    without loading the index, which is deferred to the first question.
    """
    expected_dim = config["embeddings"].get("truncate_dim") or config["embeddings"].get("embedding_dim")
    return collection_state.index_exists(expected_dim)


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


def _start_zim_watcher(interval: int = 30) -> threading.Event:
    """Watch data/ for added/removed ZIM files and update kiwix_search paths.

    Returns a stop event for clean shutdown.
    """
    stop_event = threading.Event()
    previous_paths: list[str] = [str(p) for p in discover_zims()]

    def _watcher():
        nonlocal previous_paths
        while not stop_event.is_set():
            stop_event.wait(interval)
            if stop_event.is_set():
                break
            current_paths = [str(p) for p in discover_zims()]
            if current_paths != previous_paths:
                added = set(current_paths) - set(previous_paths)
                removed = set(previous_paths) - set(current_paths)
                for p in added:
                    logger.info("Nouveau ZIM détecté : %s", Path(p).name)
                for p in removed:
                    logger.info("ZIM retiré : %s", Path(p).name)
                kiwix_search.set_zim_paths(current_paths)
                previous_paths = current_paths

    thread = threading.Thread(target=_watcher, name="zim-watcher", daemon=True)
    thread.start()
    return stop_event


def _run_background_download(config: dict) -> None:
    """Download a ZIM file in background (without indexation).

    When the download completes, refreshes ZIM paths for Kiwix search.
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
            kiwix_search.set_zim_paths(discover_zims())
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
        success = updater.run_update()
        if success:
            _restart_app()

    thread = threading.Thread(target=_update, name="wikipedia-updater", daemon=True)
    thread.start()


def _run_background_reindex(config: dict) -> None:
    """Force a full reindex in a background thread (skips download check)."""
    from meshwiki.zim_discovery import discover_zims

    def _reindex():
        try:
            import os
            if hasattr(os, "nice"):
                os.nice(10)
        except OSError:
            pass

        zim_paths = discover_zims()
        if not zim_paths:
            logger.error("Aucun fichier ZIM trouvé pour la réindexation")
            return

        updater = WikipediaUpdater()
        success = updater.reindex(zim_paths[0])
        if success:
            logger.info("Réindexation complète terminée avec succès")
            _restart_app()
        else:
            logger.error("Échec de la réindexation")

    thread = threading.Thread(target=_reindex, name="wikipedia-reindex", daemon=True)
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
                    kiwix_search.set_zim_paths(discover_zims())
                    logger.info(
                        "Nouveau ZIM téléchargé : %s — lancez avec --index pour réindexer",
                        zim_path.name,
                    )

            # Check once a day (interval_days controls actual update frequency)
            stop_event.wait(86400)

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

    # Cleanup inactive database left over from previous swap (before any ChromaDB connection)
    _cleanup_inactive_db()

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
    logger.info("Vérification de la connexion Ollama...")
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

    # Discover ZIM files and set up Kiwix search
    if not nowiki:
        zim_paths = discover_zims()
        if zim_paths:
            kiwix_search.set_zim_paths(zim_paths)
            logger.info(
                "Recherche Kiwix activée (%d ZIM) : %s",
                len(zim_paths),
                ", ".join(p.name for p in zim_paths),
            )

    # Check/create index
    if not noindex:
        logger.info("Vérification de l'index ChromaDB...")
    if not noindex and _index_exists(cfg):
        logger.info("Index ChromaDB disponible (chargé au premier message)")
        # Update check only if --update is passed
        if _wants_update() or _wants_indexation():
            if _wants_indexation():
                from meshwiki import wikipedia_indexer
                if wikipedia_indexer.has_checkpoint():
                    # Partial index in progress — resume automatically
                    logger.info("Checkpoint détecté, reprise de l'indexation en arrière-plan...")
                    _run_background_reindex(cfg)
                else:
                    # No partial index — ask before starting from scratch
                    answer = input("Un index existe déjà. Réindexer ? (o/N) ").strip().lower()
                    if answer not in ("o", "oui", "y", "yes"):
                        logger.info("Réindexation annulée par l'utilisateur")
                    else:
                        logger.info("Réindexation forcée, lancement en arrière-plan...")
                        _run_background_reindex(cfg)
            else:
                logger.info("Mise à jour demandée, téléchargement du ZIM en arrière-plan...")
                _run_background_download(cfg)
    else:
        if not nowiki and not kiwix_search.has_zim_paths():
            if not offline:
                logger.info("Aucun ZIM trouvé — téléchargement en arrière-plan...")
                _run_background_download(cfg)
            else:
                logger.info("Aucun index ni ZIM trouvé — mode offline, le LLM répondra seul.")
        elif nowiki:
            if not offline and not noindex:
                logger.info("Le LLM répondra seul (ZIM désactivé, pas d'index)")

        if not noindex and _wants_indexation():
            logger.info("Indexation demandée, lancement en arrière-plan...")
            _run_background_update(cfg)
        elif not noindex:
            logger.info("Lancez avec --index pour créer l'index ChromaDB")

    # Graceful shutdown
    stop_events: list[threading.Event] = []

    def _signal_handler(sig, frame):
        logger.info("Arrêt de MeshWiki...")
        bridge.close()
        for ev in stop_events:
            ev.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start ZIM watcher (hot-reload)
    if not nowiki:
        stop_events.append(_start_zim_watcher())

    # Start update scheduler (disabled in offline mode)
    if cfg["updater"]["enabled"] and not _wants_offline():
        stop_events.append(_start_update_scheduler(cfg))

    # Connect and run
    bridge.reconnect_loop()
