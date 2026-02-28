"""SQLite event store for persisting HA state_changed events.

Separate database from hub.db to avoid write contention.
Uses WAL mode for concurrent read access during analysis queries.
"""

import logging
import os
from typing import TypedDict

import aiosqlite

logger = logging.getLogger(__name__)


class StateChangeEvent(TypedDict, total=False):
    """Row dict returned by EventStore query methods."""

    id: int
    timestamp: str
    entity_id: str
    domain: str
    old_state: str | None
    new_state: str | None
    device_id: str | None
    area_id: str | None
    attributes_json: str | None
    context_parent_id: str | None


class EventStore:
    """Async SQLite store for Home Assistant state_changed events."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._reconnect_failed: bool = False

    async def initialize(self) -> None:
        """Create database, enable WAL mode, and ensure schema exists."""
        self._reconnect_failed = False
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state_change_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                old_state TEXT,
                new_state TEXT,
                device_id TEXT,
                area_id TEXT,
                attributes_json TEXT,
                context_parent_id TEXT
            )
        """)

        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sce_ts ON state_change_events(timestamp)")
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sce_entity ON state_change_events(entity_id, timestamp)"
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sce_area ON state_change_events(area_id, timestamp)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sce_domain ON state_change_events(domain, timestamp)")
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sce_context ON state_change_events(context_parent_id)")

        # Migration for existing databases: add context_parent_id if missing
        try:
            await self._conn.execute("ALTER TABLE state_change_events ADD COLUMN context_parent_id TEXT")
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sce_context ON state_change_events(context_parent_id)"
            )
        except Exception as e:
            # "duplicate column name" is the expected path — suppress it silently.
            # Any other error means the schema migration failed, which must not be hidden.
            if "duplicate column" not in str(e).lower():
                logger.error(
                    "event_store migration failed — context_parent_id column: %s — db_path: %s",
                    e,
                    self.db_path,
                )
                raise

        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def shutdown(self) -> None:
        """Close the database connection on shutdown (alias for close())."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def _get_conn(self) -> aiosqlite.Connection:
        """Return a live connection, reconnecting if the connection was dropped.

        This handles the case where the persistent aiosqlite connection is lost
        (e.g. the underlying file descriptor was closed unexpectedly). Rather than
        letting all subsequent operations fail with RuntimeError, we transparently
        re-establish the connection and re-apply the required pragmas.

        If a previous reconnect attempt failed (e.g. disk full), the circuit
        breaker flag prevents a reconnect storm. Call initialize() directly to
        clear the flag and retry.
        """
        if self._conn is None:
            if self._reconnect_failed:
                raise RuntimeError("EventStore connection previously failed to reconnect — call initialize() to retry")
            logger.warning("EventStore: connection was None (dropped?) — reconnecting to %s", self.db_path)
            try:
                await self.initialize()
            except Exception:
                self._reconnect_failed = True
                raise
        return self._conn  # type: ignore[return-value]

    # ── Write methods ───────────────────────────────────────────────────

    async def insert_event(  # noqa: PLR0913 — matches DB column count
        self,
        timestamp: str,
        entity_id: str,
        domain: str,
        old_state: str | None = None,
        new_state: str | None = None,
        device_id: str | None = None,
        area_id: str | None = None,
        attributes_json: str | None = None,
        context_parent_id: str | None = None,
    ) -> None:
        """Insert a single state_changed event."""
        conn = await self._get_conn()
        await conn.execute(
            """INSERT INTO state_change_events
               (timestamp, entity_id, domain, old_state, new_state,
                device_id, area_id, attributes_json, context_parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                entity_id,
                domain,
                old_state,
                new_state,
                device_id,
                area_id,
                attributes_json,
                context_parent_id,
            ),
        )
        await conn.commit()

    async def insert_events_batch(self, events: list[tuple]) -> None:
        """Bulk insert events. Each tuple must match column order:
        (timestamp, entity_id, domain, old_state, new_state,
         device_id, area_id, attributes_json[, context_parent_id]).

        Accepts 8 or 9 element tuples for backward compatibility.
        Missing context_parent_id defaults to None.
        """
        conn = await self._get_conn()
        if not events:
            return
        # Pad 8-element tuples to 9 for backward compatibility
        padded = [e if len(e) >= 9 else (*e, None) for e in events]
        await conn.executemany(
            """INSERT INTO state_change_events
               (timestamp, entity_id, domain, old_state, new_state,
                device_id, area_id, attributes_json, context_parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            padded,
        )
        await conn.commit()

    # ── Read methods ────────────────────────────────────────────────────

    async def query_events(self, start: str, end: str, limit: int = 10000) -> list[dict]:
        """Query events within a time window [start, end)."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM state_change_events
               WHERE timestamp >= ? AND timestamp < ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (start, end, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def query_by_entity(self, entity_id: str, start: str, end: str, limit: int = 10000) -> list[dict]:
        """Query events for a specific entity within a time window."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM state_change_events
               WHERE entity_id = ? AND timestamp >= ? AND timestamp < ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (entity_id, start, end, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def query_by_area(self, area_id: str, start: str, end: str, limit: int = 10000) -> list[dict]:
        """Query events for a specific area within a time window."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM state_change_events
               WHERE area_id = ? AND timestamp >= ? AND timestamp < ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (area_id, start, end, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def query_by_domain(self, domain: str, start: str, end: str, limit: int = 10000) -> list[dict]:
        """Query events for a specific domain within a time window."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM state_change_events
               WHERE domain = ? AND timestamp >= ? AND timestamp < ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (domain, start, end, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def count_events(self, start: str, end: str) -> int:
        """Count events within a time window [start, end)."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT COUNT(*) FROM state_change_events
               WHERE timestamp >= ? AND timestamp < ?""",
            (start, end),
        )
        row = await cursor.fetchone()
        assert row is not None
        return row[0]

    async def query_manual_events(self, start: str, end: str, limit: int = 10000) -> list[dict]:
        """Query events where context_parent_id IS NULL (manual actions only)."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM state_change_events
               WHERE timestamp >= ? AND timestamp < ?
               AND context_parent_id IS NULL
               ORDER BY timestamp ASC
               LIMIT ?""",
            (start, end, limit),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def area_event_summary(self, start: str, end: str) -> dict[str, int]:
        """Aggregate event counts by area_id for performance tiering."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT area_id, COUNT(*) as cnt FROM state_change_events
               WHERE timestamp >= ? AND timestamp < ?
               AND area_id IS NOT NULL
               GROUP BY area_id""",
            (start, end),
        )
        return {row["area_id"]: row["cnt"] for row in await cursor.fetchall()}

    # ── Retention / pruning ─────────────────────────────────────────────

    async def prune_before(self, cutoff: str) -> int:
        """Delete events with timestamp strictly before cutoff. Returns count deleted."""
        conn = await self._get_conn()
        cursor = await conn.execute(
            "DELETE FROM state_change_events WHERE timestamp < ?",
            (cutoff,),
        )
        await conn.commit()
        return cursor.rowcount

    async def total_count(self) -> int:
        """Total number of events in the store."""
        conn = await self._get_conn()
        cursor = await conn.execute("SELECT COUNT(*) FROM state_change_events")
        row = await cursor.fetchone()
        assert row is not None
        return row[0]
