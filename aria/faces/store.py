"""Face embedding store — SQLite CRUD for embeddings and review queue."""

import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Input record for add_embedding — groups related fields to stay under PLR0913."""

    person_name: str | None
    embedding: np.ndarray
    event_id: str
    image_path: str
    confidence: float
    source: str
    verified: bool = False


class FaceEmbeddingStore:
    """SQLite-backed store for face embeddings and review queue."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT,
                    embedding   BLOB NOT NULL,
                    event_id    TEXT,
                    image_path  TEXT,
                    confidence  REAL,
                    source      TEXT NOT NULL,
                    verified    INTEGER DEFAULT 0,
                    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS face_review_queue (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id        TEXT NOT NULL,
                    image_path      TEXT NOT NULL,
                    embedding       BLOB NOT NULL,
                    top_candidates  TEXT,
                    priority        REAL DEFAULT 0.5,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at     DATETIME,
                    person_name     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_embeddings_person
                    ON face_embeddings(person_name);
                CREATE INDEX IF NOT EXISTS idx_queue_priority
                    ON face_review_queue(priority DESC, reviewed_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_event_live
                    ON face_embeddings(event_id) WHERE source = 'live';
                CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_event_bootstrap
                    ON face_embeddings(event_id) WHERE source = 'bootstrap';
            """)
            conn.commit()

        # Migration: add camera column if it doesn't exist
        with closing(sqlite3.connect(self.db_path)) as conn:
            try:
                conn.execute("ALTER TABLE face_review_queue ADD COLUMN camera TEXT")
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

    # --- Embeddings ---

    def add_embedding(  # noqa: PLR0913
        self,
        person_name: str | None,
        embedding: np.ndarray,
        event_id: str,
        image_path: str,
        confidence: float,
        source: str,
        verified: bool = False,
    ) -> int:
        blob = embedding.astype(np.float32).tobytes()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                """INSERT INTO face_embeddings
                   (person_name, embedding, event_id, image_path, confidence, source, verified)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (person_name, blob, event_id, image_path, confidence, source, int(verified)),
            )
            conn.commit()
            return cur.lastrowid

    def get_embeddings_for_person(self, person_name: str) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM face_embeddings WHERE person_name = ?",
                (person_name,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_all_named_embeddings(self) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM face_embeddings WHERE person_name IS NOT NULL").fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_known_people(self) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute(
                """SELECT person_name, COUNT(*) as count
                   FROM face_embeddings
                   WHERE person_name IS NOT NULL
                   GROUP BY person_name
                   ORDER BY count DESC"""
            ).fetchall()
        return [{"person_name": r[0], "count": r[1]} for r in rows]

    # --- Adaptive threshold ---

    @staticmethod
    def get_threshold_for_person(_person_name: str, labeled_count: int) -> float:
        """Per-person adaptive threshold — tightens as sample count grows.

        Formula from arxiv 1810.11160 (+22% accuracy vs fixed threshold).
        """
        return max(0.50, 0.85 - (0.005 * labeled_count))

    # --- Review queue ---

    def add_to_review_queue(  # noqa: PLR0913
        self,
        event_id: str,
        image_path: str,
        embedding: np.ndarray,
        top_candidates: list[dict],
        priority: float,
        camera: str = "",
    ) -> int:
        blob = embedding.astype(np.float32).tobytes()
        candidates_json = json.dumps(top_candidates)
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                """INSERT INTO face_review_queue
                   (event_id, image_path, embedding, top_candidates, priority, camera)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event_id, image_path, blob, candidates_json, priority, camera),
            )
            conn.commit()
            return cur.lastrowid

    def get_review_queue(self, limit: int = 20) -> list[dict]:
        """Return pending queue items, highest priority first."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM face_review_queue
                   WHERE reviewed_at IS NULL
                   ORDER BY priority DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["top_candidates"] = json.loads(d["top_candidates"] or "[]")
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
            _fix_naive_utc(d, "created_at")
            result.append(d)
        return result

    def mark_reviewed(self, queue_id: int, person_name: str | None = None) -> bool:
        """Mark a queue item as reviewed. Returns True if this call did the marking.

        Adding `AND reviewed_at IS NULL` makes the update idempotent: only the first
        concurrent request wins. Callers should skip side-effects (add_embedding) when
        this returns False — the item was already reviewed by a concurrent request.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                """UPDATE face_review_queue
                   SET reviewed_at = ?, person_name = ?
                   WHERE id = ? AND reviewed_at IS NULL""",
                (datetime.now(UTC).isoformat(), person_name, queue_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_queue_depth(self) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            return conn.execute("SELECT COUNT(*) FROM face_review_queue WHERE reviewed_at IS NULL").fetchone()[0]

    def get_pending_queue_item(self, queue_id: int) -> dict | None:
        """Return a single pending (unreviewed) queue item by id, with embedding.

        Returns None if not found or already reviewed. This avoids scanning the
        full queue with a LIMIT when labeling a specific item (critical for queues
        deeper than 1000 items after a large bootstrap run).
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM face_review_queue WHERE id = ? AND reviewed_at IS NULL",
                (queue_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["top_candidates"] = json.loads(d["top_candidates"] or "[]")
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
        _fix_naive_utc(d, "created_at")
        return d

    def get_queue_image_path(self, queue_id: int) -> str | None:
        """Return image_path for a queue item (any review status)."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT image_path FROM face_review_queue WHERE id = ?", (queue_id,)).fetchone()
        return row[0] if row else None

    def dismiss_queue(self) -> int:
        """Mark all pending (unreviewed) queue items as dismissed. Returns count cleared."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                """UPDATE face_review_queue
                   SET reviewed_at = ?, person_name = 'dismissed'
                   WHERE reviewed_at IS NULL""",
                (datetime.now(UTC).isoformat(),),
            )
            conn.commit()
            return cur.rowcount

    def dismiss_single(self, queue_id: int) -> bool:
        """Mark a single queue item as dismissed. Returns True if found."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "UPDATE face_review_queue SET reviewed_at = ?, person_name = 'dismissed'"
                " WHERE id = ? AND reviewed_at IS NULL",
                (datetime.now(UTC).isoformat(), queue_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # --- Embedding management ---

    def get_embeddings_by_id(self, embedding_id: int) -> dict | None:
        """Fetch a single embedding by id. Returns dict or None if not found."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM face_embeddings WHERE id = ?", (embedding_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def delete_embedding(self, embedding_id: int) -> bool:
        """Delete a single embedding by id. Returns True if a row was deleted."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute("DELETE FROM face_embeddings WHERE id = ?", (embedding_id,))
            conn.commit()
            return cur.rowcount > 0

    def delete_embedding_for_person(self, embedding_id: int, person_name: str) -> bool:
        """Delete a single embedding by id, constrained to a specific person.

        Returns True only if a row was deleted AND it belonged to person_name.
        Returns False if the id doesn't exist or belongs to a different person.
        This prevents one person's samples from being deleted via another person's URL.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM face_embeddings WHERE id = ? AND person_name = ?",
                (embedding_id, person_name),
            )
            conn.commit()
            return cur.rowcount > 0

    def rename_person(self, old_name: str, new_name: str) -> int:
        """Rename all embeddings from old_name to new_name. Returns count updated."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "UPDATE face_embeddings SET person_name = ? WHERE person_name = ?",
                (new_name, old_name),
            )
            conn.commit()
            return cur.rowcount

    def delete_person(self, person_name: str) -> int:
        """Delete all embeddings for a person. Returns count deleted."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute("DELETE FROM face_embeddings WHERE person_name = ?", (person_name,))
            conn.commit()
            return cur.rowcount

    def purge_reviewed_items(self, older_than_days: int = 7) -> int:
        """Delete reviewed/dismissed queue items older than N days. Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                "DELETE FROM face_review_queue WHERE reviewed_at IS NOT NULL AND reviewed_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount

    def get_auto_label_rate(self) -> float:
        """Return fraction of face events that were auto-labeled vs queued for review.

        auto_labels = live-source embeddings (auto-labeled by pipeline)
        queued      = all review queue items (reviewed + unreviewed)
        rate        = auto_labels / (auto_labels + queued)
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            auto_count = conn.execute("SELECT COUNT(*) FROM face_embeddings WHERE source = 'live'").fetchone()[0]
            queue_count = conn.execute("SELECT COUNT(*) FROM face_review_queue").fetchone()[0]
        total = auto_count + queue_count
        return round(auto_count / total, 3) if total > 0 else 0.0


def _fix_naive_utc(d: dict, field: str) -> None:
    """Normalize a naive UTC timestamp to ISO-8601 with Z suffix.

    SQLite CURRENT_TIMESTAMP returns "YYYY-MM-DD HH:MM:SS" (no timezone).
    JavaScript's Date constructor parses bare timestamps as local time, causing
    display offsets for non-UTC users. Appending Z marks them explicitly UTC.
    """
    val = d.get(field)
    if val and isinstance(val, str) and not val.endswith("Z") and "+" not in val:
        d[field] = val.replace(" ", "T") + "Z"


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("embedding"):
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32).copy()
    _fix_naive_utc(d, "created_at")
    return d
