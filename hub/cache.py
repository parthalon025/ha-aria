"""SQLite cache manager with JSON columns and versioning."""

import json
import aiosqlite
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
from pathlib import Path


class CacheManager:
    """Manages SQLite cache for hub data storage."""

    def __init__(self, db_path: str):
        """Initialize cache manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Initialize database schema."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Connect to database
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # Create tables
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                category TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                last_updated TEXT NOT NULL,
                metadata TEXT
            )
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                category TEXT,
                data TEXT,
                metadata TEXT
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
            ON events(timestamp DESC)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_type
            ON events(event_type)
        """)

        # Shadow engine tables
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                context TEXT NOT NULL,
                predictions TEXT NOT NULL,
                outcome TEXT,
                actual TEXT,
                confidence REAL NOT NULL,
                is_exploration BOOLEAN DEFAULT FALSE,
                propagated_count INTEGER DEFAULT 0,
                window_seconds INTEGER NOT NULL,
                resolved_at TEXT
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
            ON predictions(timestamp DESC)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_outcome
            ON predictions(outcome)
        """)

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_state (
                id INTEGER PRIMARY KEY DEFAULT 1,
                current_stage TEXT NOT NULL DEFAULT 'backtest',
                stage_entered_at TEXT NOT NULL,
                backtest_accuracy REAL,
                shadow_accuracy_7d REAL,
                suggest_approval_rate_14d REAL,
                autonomous_contexts TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        await self._conn.commit()

    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get(self, category: str) -> Optional[Dict[str, Any]]:
        """Get data from cache by category.

        Args:
            category: Cache category (e.g., "entities", "areas")

        Returns:
            Cache entry with data, version, last_updated, metadata or None if not found
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM cache WHERE category = ?",
            (category,)
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "category": row["category"],
            "data": json.loads(row["data"]),
            "version": row["version"],
            "last_updated": row["last_updated"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None
        }

    async def set(
        self,
        category: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Set data in cache, incrementing version.

        Args:
            category: Cache category
            data: Data to store (will be JSON-serialized)
            metadata: Optional metadata

        Returns:
            New version number
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        # Get current version
        current = await self.get(category)
        new_version = (current["version"] + 1) if current else 1

        # Store data
        await self._conn.execute(
            """
            INSERT INTO cache (category, data, version, last_updated, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(category) DO UPDATE SET
                data = excluded.data,
                version = excluded.version,
                last_updated = excluded.last_updated,
                metadata = excluded.metadata
            """,
            (
                category,
                json.dumps(data),
                new_version,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            )
        )

        await self._conn.commit()

        # Log event
        await self.log_event(
            event_type="cache_update",
            category=category,
            metadata={"version": new_version}
        )

        return new_version

    async def delete(self, category: str) -> bool:
        """Delete category from cache.

        Args:
            category: Cache category to delete

        Returns:
            True if deleted, False if not found
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "DELETE FROM cache WHERE category = ?",
            (category,)
        )
        await self._conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            await self.log_event(
                event_type="cache_delete",
                category=category
            )

        return deleted

    async def list_categories(self) -> List[str]:
        """List all cache categories.

        Returns:
            List of category names
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT category FROM cache ORDER BY category"
        )
        rows = await cursor.fetchall()
        return [row["category"] for row in rows]

    async def log_event(
        self,
        event_type: str,
        category: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an event to the events table.

        Args:
            event_type: Type of event (e.g., "cache_update", "module_registered")
            category: Related cache category (optional)
            data: Event data (optional)
            metadata: Event metadata (optional)
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        await self._conn.execute(
            """
            INSERT INTO events (timestamp, event_type, category, data, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                event_type,
                category,
                json.dumps(data) if data else None,
                json.dumps(metadata) if metadata else None
            )
        )
        await self._conn.commit()

    async def get_events(
        self,
        event_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent events from the events table.

        Args:
            event_type: Filter by event type (optional)
            category: Filter by category (optional)
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        query = "SELECT * FROM events WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "category": row["category"],
                "data": json.loads(row["data"]) if row["data"] else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None
            }
            for row in rows
        ]

    # ========================================================================
    # Shadow engine: predictions
    # ========================================================================

    async def insert_prediction(
        self,
        prediction_id: str,
        timestamp: str,
        context: Dict[str, Any],
        predictions: List[Any],
        confidence: float,
        window_seconds: int,
        is_exploration: bool = False,
    ) -> None:
        """Insert a new prediction record.

        Args:
            prediction_id: Unique ID for this prediction
            timestamp: ISO timestamp when prediction was made
            context: Full context snapshot (will be JSON-serialized)
            predictions: Array of predictions (will be JSON-serialized)
            confidence: Confidence score (0.0-1.0)
            window_seconds: Evaluation window in seconds
            is_exploration: Whether this is an exploration prediction
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        await self._conn.execute(
            """
            INSERT INTO predictions
                (id, timestamp, context, predictions, confidence, window_seconds, is_exploration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                timestamp,
                json.dumps(context),
                json.dumps(predictions),
                confidence,
                window_seconds,
                is_exploration,
            ),
        )
        await self._conn.commit()

    async def update_prediction_outcome(
        self,
        prediction_id: str,
        outcome: str,
        actual: Optional[Dict[str, Any]] = None,
        propagated_count: int = 0,
    ) -> None:
        """Update a prediction with its outcome.

        Args:
            prediction_id: ID of the prediction to update
            outcome: Result — 'correct', 'disagreement', or 'nothing'
            actual: What actually happened (will be JSON-serialized)
            propagated_count: Number of times this prediction was propagated
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        await self._conn.execute(
            """
            UPDATE predictions
            SET outcome = ?, actual = ?, propagated_count = ?, resolved_at = ?
            WHERE id = ?
            """,
            (
                outcome,
                json.dumps(actual) if actual else None,
                propagated_count,
                datetime.now().isoformat(),
                prediction_id,
            ),
        )
        await self._conn.commit()

    async def get_recent_predictions(
        self,
        limit: int = 50,
        outcome_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent predictions, optionally filtered by outcome type.

        Args:
            limit: Maximum number of predictions to return
            outcome_filter: Filter by outcome ('correct', 'disagreement', 'nothing', or None for all)

        Returns:
            List of prediction dicts
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        query = "SELECT * FROM predictions WHERE 1=1"
        params: list = []

        if outcome_filter is not None:
            query += " AND outcome = ?"
            params.append(outcome_filter)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [self._prediction_from_row(row) for row in rows]

    async def get_pending_predictions(
        self,
        before_timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get predictions with NULL outcome whose window has expired.

        Args:
            before_timestamp: Only return predictions made before this ISO timestamp.
                If None, uses current time minus window_seconds for each row.

        Returns:
            List of pending prediction dicts
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        if before_timestamp is None:
            before_timestamp = datetime.now().isoformat()

        query = """
            SELECT * FROM predictions
            WHERE outcome IS NULL
              AND datetime(timestamp, '+' || window_seconds || ' seconds') <= datetime(?)
            ORDER BY timestamp ASC
        """
        params: list = [before_timestamp]

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [self._prediction_from_row(row) for row in rows]

    async def get_accuracy_stats(self, days: int = 7) -> Dict[str, Any]:
        """Calculate accuracy metrics over a time window.

        Args:
            days: Number of days to look back

        Returns:
            Dict with overall_accuracy, per_outcome breakdown, and daily_trend
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        # Overall counts by outcome
        cursor = await self._conn.execute(
            """
            SELECT outcome, COUNT(*) as cnt
            FROM predictions
            WHERE resolved_at IS NOT NULL AND timestamp >= ?
            GROUP BY outcome
            """,
            (cutoff,),
        )
        rows = await cursor.fetchall()

        per_outcome: Dict[str, int] = {}
        total_resolved = 0
        correct_count = 0
        for row in rows:
            per_outcome[row["outcome"]] = row["cnt"]
            total_resolved += row["cnt"]
            if row["outcome"] == "correct":
                correct_count = row["cnt"]

        overall_accuracy = (correct_count / total_resolved) if total_resolved > 0 else 0.0

        # Daily trend
        cursor = await self._conn.execute(
            """
            SELECT date(resolved_at) as day,
                   SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) as correct,
                   COUNT(*) as total
            FROM predictions
            WHERE resolved_at IS NOT NULL AND timestamp >= ?
            GROUP BY date(resolved_at)
            ORDER BY day ASC
            """,
            (cutoff,),
        )
        trend_rows = await cursor.fetchall()

        daily_trend = [
            {
                "date": row["day"],
                "correct": row["correct"],
                "total": row["total"],
                "accuracy": row["correct"] / row["total"] if row["total"] > 0 else 0.0,
            }
            for row in trend_rows
        ]

        return {
            "overall_accuracy": overall_accuracy,
            "total_resolved": total_resolved,
            "per_outcome": per_outcome,
            "daily_trend": daily_trend,
        }

    # ========================================================================
    # Shadow engine: pipeline state
    # ========================================================================

    async def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state, creating default row if not exists.

        Returns:
            Pipeline state dict
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM pipeline_state WHERE id = 1"
        )
        row = await cursor.fetchone()

        if not row:
            now = datetime.now().isoformat()
            await self._conn.execute(
                """
                INSERT INTO pipeline_state
                    (id, current_stage, stage_entered_at, updated_at)
                VALUES (1, 'backtest', ?, ?)
                """,
                (now, now),
            )
            await self._conn.commit()

            cursor = await self._conn.execute(
                "SELECT * FROM pipeline_state WHERE id = 1"
            )
            row = await cursor.fetchone()

        return {
            "id": row["id"],
            "current_stage": row["current_stage"],
            "stage_entered_at": row["stage_entered_at"],
            "backtest_accuracy": row["backtest_accuracy"],
            "shadow_accuracy_7d": row["shadow_accuracy_7d"],
            "suggest_approval_rate_14d": row["suggest_approval_rate_14d"],
            "autonomous_contexts": (
                json.loads(row["autonomous_contexts"])
                if row["autonomous_contexts"]
                else None
            ),
            "updated_at": row["updated_at"],
        }

    async def update_pipeline_state(self, **kwargs) -> None:
        """Update pipeline state fields.

        Args:
            **kwargs: Fields to update. Supported fields:
                current_stage, stage_entered_at, backtest_accuracy,
                shadow_accuracy_7d, suggest_approval_rate_14d,
                autonomous_contexts
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        # Ensure default row exists
        await self.get_pipeline_state()

        allowed_fields = {
            "current_stage",
            "stage_entered_at",
            "backtest_accuracy",
            "shadow_accuracy_7d",
            "suggest_approval_rate_14d",
            "autonomous_contexts",
        }

        updates = {}
        for key, value in kwargs.items():
            if key not in allowed_fields:
                raise ValueError(f"Unknown pipeline_state field: {key}")
            if key == "autonomous_contexts" and value is not None:
                updates[key] = json.dumps(value)
            else:
                updates[key] = value

        if not updates:
            return

        updates["updated_at"] = datetime.now().isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(1)  # WHERE id = 1

        await self._conn.execute(
            f"UPDATE pipeline_state SET {set_clause} WHERE id = ?",
            values,
        )
        await self._conn.commit()

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _prediction_from_row(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert a predictions table row to a dict."""
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "context": json.loads(row["context"]),
            "predictions": json.loads(row["predictions"]),
            "outcome": row["outcome"],
            "actual": json.loads(row["actual"]) if row["actual"] else None,
            "confidence": row["confidence"],
            "is_exploration": bool(row["is_exploration"]),
            "propagated_count": row["propagated_count"],
            "window_seconds": row["window_seconds"],
            "resolved_at": row["resolved_at"],
        }
