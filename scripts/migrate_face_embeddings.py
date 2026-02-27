#!/usr/bin/env python3
"""One-time migration: re-embed labeled faces using InsightFace after DeepFace swap.

Run once after installing InsightFace. Deletes all existing embeddings and
re-processes every verified+labeled image through the new backend.

Usage:
    cd ~/Documents/projects/ha-aria
    .venv/bin/python scripts/migrate_face_embeddings.py
"""

import logging
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    db_path = Path.home() / "ha-logs/intelligence/cache/faces.db"
    if not db_path.exists():
        logger.error("faces.db not found at %s", db_path)
        return 1

    # Check InsightFace is available before touching the DB
    try:
        from aria.faces.extractor import FaceExtractor  # noqa: PLC0415
    except ImportError as e:
        logger.error("Cannot import FaceExtractor: %s", e)
        return 1

    # Initialize extractor — triggers buffalo_l download on first run (~300MB, may take 60s)
    logger.info("Initializing InsightFace extractor...")
    from aria.faces.extractor import _get_app  # noqa: PLC0415

    if _get_app() is None:
        logger.error("InsightFace failed to initialize — check installation")
        return 1
    extractor = FaceExtractor()
    logger.info("InsightFace ready")

    # Load all verified labeled records BEFORE clearing
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute("""
            SELECT DISTINCT person_name, image_path
            FROM face_embeddings
            WHERE verified = 1
              AND person_name IS NOT NULL
              AND image_path IS NOT NULL
        """).fetchall()

    logger.info("Found %d verified labeled images to re-embed", len(rows))
    if not rows:
        logger.warning("No labeled images found — nothing to migrate")
        return 0

    # Clear ALL existing embeddings
    with closing(sqlite3.connect(str(db_path))) as conn:
        deleted = conn.execute("DELETE FROM face_embeddings").rowcount
        conn.commit()
    logger.info("Cleared %d existing embeddings", deleted)

    # Re-embed each labeled image
    inserted = 0
    skipped_missing = 0
    skipped_no_face = 0

    with closing(sqlite3.connect(str(db_path))) as conn:
        for person_name, image_path in rows:
            path = Path(image_path)
            if not path.exists():
                logger.warning("  SKIP (file missing): %s -> %s", person_name, image_path)
                skipped_missing += 1
                continue

            embedding = extractor.extract_embedding(str(path))
            if embedding is None:
                logger.warning("  SKIP (no face detected): %s -> %s", person_name, image_path)
                skipped_no_face += 1
                continue

            blob = embedding.astype(np.float32).tobytes()
            conn.execute(
                """
                INSERT INTO face_embeddings
                    (person_name, embedding, event_id, image_path, confidence, source, verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_name,
                    blob,
                    f"migration_{path.stem}",
                    str(path),
                    1.0,
                    "migration",
                    1,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
            logger.info("  OK: %s -> %s", person_name, path.name)
            inserted += 1

    logger.info(
        "\nMigration complete: %d inserted, %d skipped (missing), %d skipped (no face)",
        inserted,
        skipped_missing,
        skipped_no_face,
    )
    if skipped_no_face:
        logger.warning(
            "%d images had no detectable face — these people will need to be re-labeled from new Frigate events.",
            skipped_no_face,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
