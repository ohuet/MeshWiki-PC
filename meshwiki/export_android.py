"""Export ChromaDB index to SQLite for the Android MeshWiki app.

Reads the active ChromaDB collection and produces a SQLite database
with the schema expected by the Android Room-based IndexImporter:
- chunks (id, article_title, content, embedding BLOB, position)
- chunks_fts (FTS4 virtual table on article_title + content)
- metadata (key/value pairs)

Usage:
    python -m meshwiki.export_android [output_path]
"""

import argparse
import logging
import os
import sqlite3
import struct
import sys
import time

import chromadb

from meshwiki import config
from meshwiki.collection_state import get_active_db_path
from meshwiki.progress import ProgressDisplay, format_bar

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024
BATCH_SIZE = 5000
DEFAULT_OUTPUT = "data/meshwiki_android.db"


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a list of floats into a little-endian binary blob."""
    return struct.pack(f"<{EMBEDDING_DIM}f", *embedding)


def _strip_title_prefix(document: str, title: str) -> str:
    """Remove the 'Title — ' prefix added during indexation."""
    prefix = f"{title} \u2014 "
    if document.startswith(prefix):
        return document[len(prefix):]
    return document


def export(output_path: str = DEFAULT_OUTPUT) -> dict:
    """Export the active ChromaDB collection to a SQLite database.

    Returns stats: {"chunk_count": int, "file_size_mb": float}
    """
    cfg = config.load_config()
    db_path = get_active_db_path()
    embeddings_model = cfg["embeddings"]["model"]

    logger.info("Ouverture de ChromaDB : %s", db_path)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("wikipedia")
    total = collection.count()
    logger.info("Collection 'wikipedia' : %d chunks", total)

    if total == 0:
        logger.error("La collection est vide, rien à exporter")
        sys.exit(1)

    # Prepare output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            position INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts4(
            article_title, content, content="chunks"
        )
    """)
    conn.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()

    progress = ProgressDisplay()
    progress.start()
    start_time = time.monotonic()
    exported = 0

    try:
        offset = 0
        while offset < total:
            limit = min(BATCH_SIZE, total - offset)
            progress.set_info(
                f"Lecture ChromaDB {offset}–{offset + limit} / {total}..."
            )

            result = collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas", "embeddings"],
            )

            rows = []
            for doc, meta, emb in zip(
                result["documents"], result["metadatas"], result["embeddings"]
            ):
                title = meta["title"]
                content = _strip_title_prefix(doc, title)
                embedding_blob = _pack_embedding(emb)
                position = meta["chunk_index"]
                rows.append((title, content, embedding_blob, position))

            conn.executemany(
                "INSERT INTO chunks (article_title, content, embedding, position) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

            exported += len(rows)
            offset += limit

            elapsed = time.monotonic() - start_time
            pct = exported / total * 100
            if exported > 0:
                remaining = elapsed / exported * (total - exported)
                mins, secs = divmod(int(remaining), 60)
                eta_dur = f"{mins}min{secs:02d}s" if mins else f"{secs}s"
                eta_time = time.strftime(
                    "%H:%M", time.localtime(time.time() + remaining)
                )
            else:
                eta_dur = "?"
                eta_time = "?"
            progress.set_progress(pct, exported, eta_dur, eta_time)

        # Build FTS index
        progress.set_info("Construction de l'index FTS4...")
        conn.execute(
            "INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"
        )
        conn.commit()

        # Insert metadata
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("embeddings_model", embeddings_model),
        )
        conn.commit()

    finally:
        progress.stop()

    # VACUUM to compact the file
    logger.info("VACUUM du fichier SQLite...")
    conn.execute("VACUUM")
    conn.close()

    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    elapsed = time.monotonic() - start_time
    mins, secs = divmod(int(elapsed), 60)

    logger.info(
        "Export terminé : %d chunks → %s (%.1f Mo) en %dmin%02ds",
        exported, output_path, file_size_mb, mins, secs,
    )

    return {"chunk_count": exported, "file_size_mb": file_size_mb}


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Exporte l'index ChromaDB vers SQLite pour Android"
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Chemin du fichier SQLite de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    stats = export(args.output)
    print(
        f"\n{stats['chunk_count']} chunks exportés — "
        f"{stats['file_size_mb']:.1f} Mo"
    )


if __name__ == "__main__":
    main()
