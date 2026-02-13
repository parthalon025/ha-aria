"""SQLite cache manager with JSON columns and versioning."""

import json
import aiosqlite
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List


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

        # Enable WAL mode for concurrent reads + busy timeout for lock contention
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

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

        # Phase 2: Config store
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT,
                default_value TEXT,
                value_type TEXT NOT NULL,
                label TEXT,
                description TEXT,
                category TEXT,
                min_value REAL,
                max_value REAL,
                options TEXT,
                step REAL,
                updated_at TEXT NOT NULL
            )
        """)

        # Phase 2: Entity curation
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_curation (
                entity_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                tier INTEGER NOT NULL,
                reason TEXT,
                auto_classification TEXT,
                human_override BOOLEAN DEFAULT FALSE,
                metrics TEXT,
                group_id TEXT,
                decided_at TEXT NOT NULL,
                decided_by TEXT
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_curation_tier
            ON entity_curation(tier)
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_curation_status
            ON entity_curation(status)
        """)

        # Phase 2: Config change history
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_at TEXT NOT NULL,
                changed_by TEXT
            )
        """)

        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_config_history_key
            ON config_history(key)
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
    # Retention / pruning
    # ========================================================================

    async def prune_events(self, retention_days: int = 7) -> int:
        """Delete events older than retention_days. Returns count deleted."""
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = await self._conn.execute(
            "DELETE FROM events WHERE timestamp < ?", (cutoff,)
        )
        await self._conn.commit()
        return cursor.rowcount

    async def prune_predictions(self, retention_days: int = 30) -> int:
        """Delete resolved predictions older than retention_days."""
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        cursor = await self._conn.execute(
            "DELETE FROM predictions WHERE resolved_at IS NOT NULL AND resolved_at < ?",
            (cutoff,)
        )
        await self._conn.commit()
        return cursor.rowcount

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
        offset: int = 0,
        outcome_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent predictions, optionally filtered by outcome type.

        Args:
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip (for pagination)
            outcome_filter: Filter by outcome ('correct', 'disagreement', 'nothing', or None for all)

        Returns:
            List of prediction dicts
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        query = "SELECT * FROM predictions WHERE 1=1"
        params: list = []

        if outcome_filter == "pending":
            query += " AND outcome IS NULL"
        elif outcome_filter is not None:
            query += " AND outcome = ?"
            params.append(outcome_filter)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)

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
            days: Number of days to look back (filters by prediction time,
                not resolution time — a prediction made 8 days ago but
                resolved today is excluded from a 7-day window)

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
    # Phase 2: Config store
    # ========================================================================

    async def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a single config parameter by key.

        Args:
            key: Config parameter key (e.g., 'shadow.min_confidence')

        Returns:
            Config row as dict, or None if not found.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM config WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._config_from_row(row)

    async def get_all_config(self) -> List[Dict[str, Any]]:
        """Get all config parameters.

        Returns:
            List of config dicts, ordered by category then key.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM config ORDER BY category, key"
        )
        rows = await cursor.fetchall()
        return [self._config_from_row(row) for row in rows]

    async def set_config(
        self, key: str, value: str, changed_by: str = "user"
    ) -> Dict[str, Any]:
        """Update a config parameter value with validation and history.

        Args:
            key: Config parameter key.
            value: New value (as string).
            changed_by: Who made the change.

        Returns:
            Updated config dict.

        Raises:
            ValueError: If key not found or value fails validation.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        current = await self.get_config(key)
        if current is None:
            raise ValueError(f"Config key not found: {key}")

        # Validate against constraints
        self._validate_config_value(value, current)

        old_value = current["value"]
        now = datetime.now().isoformat()

        # Update config
        await self._conn.execute(
            "UPDATE config SET value = ?, updated_at = ? WHERE key = ?",
            (value, now, key),
        )

        # Write history
        await self._conn.execute(
            """INSERT INTO config_history (key, old_value, new_value, changed_at, changed_by)
               VALUES (?, ?, ?, ?, ?)""",
            (key, old_value, value, now, changed_by),
        )

        await self._conn.commit()
        return await self.get_config(key)

    async def upsert_config_default(
        self,
        key: str,
        default_value: str,
        value_type: str,
        label: str = "",
        description: str = "",
        category: str = "",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        options: Optional[str] = None,
        step: Optional[float] = None,
    ) -> bool:
        """Insert a config default if the key doesn't already exist.

        Uses INSERT OR IGNORE so user overrides are preserved.

        Args:
            key: Config parameter key.
            default_value: Default value as string.
            value_type: One of 'number', 'string', 'boolean', 'select'.
            label: Human-readable label.
            description: Help text.
            category: Grouping category.
            min_value: Minimum for number types.
            max_value: Maximum for number types.
            options: Comma-separated options for select type.
            step: Step increment for number sliders.

        Returns:
            True if inserted, False if key already existed.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        now = datetime.now().isoformat()
        cursor = await self._conn.execute(
            """INSERT OR IGNORE INTO config
               (key, value, default_value, value_type, label, description,
                category, min_value, max_value, options, step, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key, default_value, default_value, value_type, label,
                description, category, min_value, max_value, options,
                step, now,
            ),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def reset_config(self, key: str, changed_by: str = "user") -> Dict[str, Any]:
        """Reset a config parameter to its default value.

        Args:
            key: Config parameter key.
            changed_by: Who reset it.

        Returns:
            Updated config dict.

        Raises:
            ValueError: If key not found.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        current = await self.get_config(key)
        if current is None:
            raise ValueError(f"Config key not found: {key}")

        return await self.set_config(key, current["default_value"], changed_by)

    async def get_config_value(self, key: str, fallback: Any = None) -> Any:
        """Convenience method: get decoded config value only.

        Returns the value decoded to its native Python type
        (float for number, bool for boolean, str otherwise).

        Args:
            key: Config parameter key.
            fallback: Value to return if key not found.

        Returns:
            Decoded value or fallback.
        """
        config = await self.get_config(key)
        if config is None:
            return fallback
        return self._decode_config_value(config["value"], config["value_type"])

    # ========================================================================
    # Phase 2: Entity curation
    # ========================================================================

    async def get_all_curation(self) -> List[Dict[str, Any]]:
        """Get all entity curation records.

        Returns:
            List of curation dicts, ordered by tier then entity_id.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM entity_curation ORDER BY tier, entity_id"
        )
        rows = await cursor.fetchall()
        return [self._curation_from_row(row) for row in rows]

    async def get_curation(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get curation record for a single entity.

        Args:
            entity_id: HA entity ID.

        Returns:
            Curation dict or None.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT * FROM entity_curation WHERE entity_id = ?", (entity_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._curation_from_row(row)

    async def get_curation_summary(self) -> Dict[str, Any]:
        """Get aggregated curation counts by tier and status.

        Returns:
            Dict with total, per_tier, and per_status breakdowns.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT tier, status, COUNT(*) as cnt FROM entity_curation GROUP BY tier, status"
        )
        rows = await cursor.fetchall()

        per_tier: Dict[int, int] = {}
        per_status: Dict[str, int] = {}
        total = 0
        for row in rows:
            tier = row["tier"]
            status = row["status"]
            cnt = row["cnt"]
            per_tier[tier] = per_tier.get(tier, 0) + cnt
            per_status[status] = per_status.get(status, 0) + cnt
            total += cnt

        return {
            "total": total,
            "per_tier": per_tier,
            "per_status": per_status,
        }

    async def upsert_curation(
        self,
        entity_id: str,
        status: str,
        tier: int,
        reason: str = "",
        auto_classification: str = "",
        human_override: bool = False,
        metrics: Optional[Dict[str, Any]] = None,
        group_id: str = "",
        decided_by: str = "system",
    ) -> None:
        """Insert or update an entity curation record.

        Args:
            entity_id: HA entity ID.
            status: Classification status (included, excluded, auto_excluded, promoted).
            tier: Tier level (1=auto-exclude, 2=edge cases, 3=default include).
            reason: Human-readable reason for classification.
            auto_classification: Machine-generated classification label.
            human_override: Whether a human overrode the auto classification.
            metrics: JSON metrics dict.
            group_id: Device group identifier.
            decided_by: Who made the decision.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        now = datetime.now().isoformat()
        await self._conn.execute(
            """INSERT INTO entity_curation
               (entity_id, status, tier, reason, auto_classification,
                human_override, metrics, group_id, decided_at, decided_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(entity_id) DO UPDATE SET
                   status = excluded.status,
                   tier = excluded.tier,
                   reason = excluded.reason,
                   auto_classification = excluded.auto_classification,
                   human_override = excluded.human_override,
                   metrics = excluded.metrics,
                   group_id = excluded.group_id,
                   decided_at = excluded.decided_at,
                   decided_by = excluded.decided_by
            """,
            (
                entity_id, status, tier, reason, auto_classification,
                human_override,
                json.dumps(metrics) if metrics else None,
                group_id, now, decided_by,
            ),
        )
        await self._conn.commit()

    async def bulk_update_curation(
        self,
        entity_ids: List[str],
        status: str,
        decided_by: str = "user",
    ) -> int:
        """Bulk update status for multiple entities.

        Args:
            entity_ids: List of entity IDs to update.
            status: New status to set.
            decided_by: Who made the change.

        Returns:
            Number of rows updated.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        if not entity_ids:
            return 0

        now = datetime.now().isoformat()
        placeholders = ",".join("?" for _ in entity_ids)
        cursor = await self._conn.execute(
            f"""UPDATE entity_curation
                SET status = ?, human_override = TRUE,
                    decided_at = ?, decided_by = ?
                WHERE entity_id IN ({placeholders})""",
            [status, now, decided_by] + entity_ids,
        )
        await self._conn.commit()
        return cursor.rowcount

    async def get_included_entity_ids(self) -> set:
        """Get the set of entity IDs that are included or promoted.

        Returns:
            Set of entity_id strings where status in ('included', 'promoted').
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        cursor = await self._conn.execute(
            "SELECT entity_id FROM entity_curation WHERE status IN ('included', 'promoted')"
        )
        rows = await cursor.fetchall()
        return {row["entity_id"] for row in rows}

    # ========================================================================
    # Phase 2: Config history
    # ========================================================================

    async def get_config_history(
        self, key: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get config change history.

        Args:
            key: Filter by config key (optional).
            limit: Maximum records to return.

        Returns:
            List of history dicts, most recent first.
        """
        if not self._conn:
            raise RuntimeError("Cache not initialized. Call initialize() first.")

        query = "SELECT * FROM config_history WHERE 1=1"
        params: list = []

        if key is not None:
            query += " AND key = ?"
            params.append(key)

        query += " ORDER BY changed_at DESC LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "key": row["key"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "changed_at": row["changed_at"],
                "changed_by": row["changed_by"],
            }
            for row in rows
        ]

    # ========================================================================
    # Internal helpers
    # ========================================================================

    def _config_from_row(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert a config table row to a dict."""
        return {
            "key": row["key"],
            "value": row["value"],
            "default_value": row["default_value"],
            "value_type": row["value_type"],
            "label": row["label"],
            "description": row["description"],
            "category": row["category"],
            "min_value": row["min_value"],
            "max_value": row["max_value"],
            "options": row["options"],
            "step": row["step"],
            "updated_at": row["updated_at"],
        }

    def _curation_from_row(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert an entity_curation table row to a dict."""
        return {
            "entity_id": row["entity_id"],
            "status": row["status"],
            "tier": row["tier"],
            "reason": row["reason"],
            "auto_classification": row["auto_classification"],
            "human_override": bool(row["human_override"]),
            "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
            "group_id": row["group_id"],
            "decided_at": row["decided_at"],
            "decided_by": row["decided_by"],
        }

    @staticmethod
    def _decode_config_value(value: str, value_type: str) -> Any:
        """Decode a config value string to its native Python type."""
        if value is None:
            return None
        if value_type == "number":
            try:
                f = float(value)
                return int(f) if f == int(f) else f
            except (ValueError, TypeError):
                return value
        if value_type == "boolean":
            return value.lower() in ("true", "1", "yes")
        return value

    @staticmethod
    def _validate_config_value(value: str, config: Dict[str, Any]) -> None:
        """Validate a config value against constraints.

        Raises ValueError if validation fails.
        """
        vtype = config.get("value_type", "string")
        if vtype == "number":
            try:
                num = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"Expected number, got: {value}")

            min_val = config.get("min_value")
            max_val = config.get("max_value")
            if min_val is not None and num < min_val:
                raise ValueError(
                    f"Value {num} below minimum {min_val}"
                )
            if max_val is not None and num > max_val:
                raise ValueError(
                    f"Value {num} above maximum {max_val}"
                )

        elif vtype == "select":
            options_str = config.get("options", "")
            if options_str:
                valid = [o.strip() for o in options_str.split(",")]
                if value not in valid:
                    raise ValueError(
                        f"Value '{value}' not in options: {valid}"
                    )

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
