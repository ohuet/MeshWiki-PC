"""Automatic Wikipedia dump download and safe re-indexation."""

import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from meshwiki import config, collection_state
from meshwiki.wikipedia_indexer import index_zim
from meshwiki.rag import reset_collection
from meshwiki import kiwix_search

logger = logging.getLogger(__name__)

LAST_UPDATE_FILE = Path("data/last_update.json")


def _rmtree_safe(path: str) -> None:
    """Delete a directory tree, ignoring errors if it doesn't exist."""
    try:
        shutil.rmtree(path)
        logger.info("Ancien index supprimé : %s", path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Impossible de supprimer %s : %s", path, e)


class WikipediaUpdater:
    """Handles Wikipedia dump download, indexation, and safe collection swap."""

    def __init__(self):
        self.config = config.load_config()
        self.updater_config = self.config["updater"]

    def get_latest_dump_url(self) -> tuple[str, str] | None:
        """Scrape Kiwix download page to find the latest matching ZIM file.

        Returns (full_url, filename) or None if not found.
        """
        kiwix_url = self.updater_config["kiwix_url"]
        pattern = self.updater_config["dump_pattern"]

        try:
            response = requests.get(kiwix_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch Kiwix page: %s", e)
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]
            filename = href.split("/")[-1]
            if filename.startswith(pattern) and filename.endswith(".zim"):
                base_url = kiwix_url.rsplit("/", 1)[0] + "/"
                full_url = href if href.startswith("http") else base_url + href
                return (full_url, filename)

        logger.warning("No matching ZIM file found with pattern '%s'", pattern)
        return None

    def download_dump(self) -> Path | None:
        """Download the latest ZIM dump if a newer version is available.

        Returns the path to the downloaded file, or None.
        """
        result = self.get_latest_dump_url()
        if result is None:
            return None

        url, filename = result

        # Check if we already have this version
        if LAST_UPDATE_FILE.exists():
            with open(LAST_UPDATE_FILE) as f:
                last_update = json.load(f)
            if last_update.get("last_filename") == filename:
                logger.info("Already up to date: %s", filename)
                return None

        temp_dir = Path(self.updater_config["temp_dir"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        dest = temp_dir / filename

        # ZIM already present on disk (e.g. LAST_UPDATE_FILE missing but file kept)
        if dest.exists():
            logger.info("ZIM file already present: %s", dest.name)
            # Update tracking so _is_update_due won't re-trigger every cycle
            LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_UPDATE_FILE, "w") as f:
                json.dump({
                    "last_filename": filename,
                    "last_update": datetime.now().isoformat(),
                }, f, indent=2)
            return dest

        download_path = dest.with_suffix(".zim.download")

        logger.info("Downloading %s ...", filename)

        try:
            headers = {}
            existing_size = 0
            if download_path.exists():
                existing_size = download_path.stat().st_size
                headers["Range"] = f"bytes={existing_size}-"
                logger.info("Resuming download from byte %d", existing_size)

            response = requests.get(url, headers=headers, stream=True, timeout=30)

            if response.status_code == 416:
                logger.info("File already fully downloaded")
                download_path.replace(dest)
                return dest

            response.raise_for_status()

            mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"
            total = int(response.headers.get("content-length", 0)) + existing_size
            downloaded = existing_size

            with open(download_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (10 * 1024 * 1024) < 8192:
                        progress = (downloaded / total) * 100
                        logger.info("Download progress: %.1f%%", progress)

            # Download complete — atomically replace the old ZIM
            download_path.replace(dest)
            logger.info("Download complete: %s", dest)
            return dest

        except requests.RequestException as e:
            logger.error("Download failed: %s", e)
            return None

    def reindex(self, zim_path: Path) -> bool:
        """Build a new index and safely swap it with the active one.

        Strategy:
        1. Index into a fresh ChromaDB database (alternating chroma_a / chroma_b)
        2. Validate article_count > 0
        3. Write pointer file to switch active database (instant)
        4. Delete old database directory in background (shutil.rmtree)
        On failure: pointer untouched, old index stays intact.

        Returns True on success, False on failure.
        """
        target_db_path = collection_state.get_inactive_db_path()
        target_slot = collection_state.get_inactive_slot()

        kiwix_search.set_zim_path(zim_path)
        try:
            # Step 1: Index into a fresh database
            logger.info("Indexing into database '%s'...", target_db_path)
            stats = index_zim(zim_path, db_path=target_db_path)

            if stats["article_count"] == 0:
                logger.error("Indexation produced 0 articles, aborting")
                _rmtree_safe(target_db_path)
                return False

            # Step 2: Instant swap — write pointer file
            old_db_path = collection_state.get_active_db_path()
            logger.info(
                "Index OK (%d articles). Swapping: %s -> %s",
                stats["article_count"], old_db_path, target_db_path,
            )
            collection_state.set_active_slot(target_slot)

            # Invalidate RAG cache so it picks up the new database
            reset_collection()

            # Delete old database in background (instant rmtree)
            threading.Thread(
                target=_rmtree_safe,
                args=(old_db_path,),
                daemon=True,
            ).start()

            # Update tracking file
            LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LAST_UPDATE_FILE, "w") as f:
                json.dump({
                    "last_filename": zim_path.name,
                    "last_update": datetime.now().isoformat(),
                    "index_article_count": stats["article_count"],
                    "index_chunk_count": stats["chunk_count"],
                }, f, indent=2)

            logger.info("Index swap complete")
            return True

        except Exception as e:
            # Don't delete the target database here: on Windows, Python
            # interpreter shutdown can raise exceptions in daemon threads
            # (module globals set to None). Deleting the database while
            # the checkpoint file survives causes data loss on resume.
            # The database and checkpoint stay in sync for safe resume.
            logger.error("Reindexation failed: %s", e)
            return False
        finally:
            kiwix_search.set_zim_path(None)

    def cleanup(self) -> None:
        """Remove temporary download files (but keep the .zim as fallback)."""
        temp_dir = Path(self.updater_config["temp_dir"])
        if temp_dir.exists():
            for f in temp_dir.iterdir():
                if f.suffix == ".zim":
                    continue
                try:
                    f.unlink()
                    logger.info("Cleaned up temp file: %s", f.name)
                except OSError:
                    pass

    def run_update(self) -> None:
        """Full update cycle: check, download, reindex, cleanup."""
        logger.info("Vérification des mises à jour Wikipedia...")

        zim_path = self.download_dump()
        if zim_path is None:
            return

        logger.info("Nouveau dump disponible, ré-indexation en cours...")
        success = self.reindex(zim_path)

        self.cleanup()
        if success:
            logger.info("Base Wikipedia mise à jour avec succès")
        else:
            logger.info("Échec de la mise à jour, ancien index conservé")
