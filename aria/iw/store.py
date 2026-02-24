"""Persistent store for behavioral state definitions, trackers, and co-activations.

Uses aiosqlite for async SQLite access. All data is JSON-serialized via the
model to_dict()/from_dict() methods.
"""

from __future__ import annotations

import json
import logging

import aiosqlite

from aria.iw.models import BehavioralStateDefinition, BehavioralStateTracker

logger = logging.getLogger(__name__)


class BehavioralStateStore:
    """Async SQLite store for I&W behavioral state data.

    Manages three tables:
    - behavioral_state_definitions — BSD definitions (JSON)
    - behavioral_state_trackers    — lifecycle + observation data (JSON)
    - state_co_activations         — pair co-occurrence counts
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection and create tables if they don't exist."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS behavioral_state_definitions (
                id       TEXT PRIMARY KEY,
                name     TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS behavioral_state_trackers (
                definition_id TEXT PRIMARY KEY,
                data_json     TEXT NOT NULL
            )
        """)
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS state_co_activations (
                state_a_id TEXT NOT NULL,
                state_b_id TEXT NOT NULL,
                count      INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (state_a_id, state_b_id)
            )
        """)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ── Definitions ───────────────────────────────────────────────────────────

    async def save_definition(self, definition: BehavioralStateDefinition) -> None:
        """Insert or replace a behavioral state definition."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        data_json = json.dumps(definition.to_dict())
        await self._conn.execute(
            "INSERT OR REPLACE INTO behavioral_state_definitions (id, name, data_json) VALUES (?, ?, ?)",
            (definition.id, definition.name, data_json),
        )
        await self._conn.commit()

    async def get_definition(self, definition_id: str) -> BehavioralStateDefinition | None:
        """Retrieve a single definition by id, or None if not found."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        async with self._conn.execute(
            "SELECT data_json FROM behavioral_state_definitions WHERE id = ?",
            (definition_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return BehavioralStateDefinition.from_dict(json.loads(row["data_json"]))

    async def list_definitions(self) -> list[BehavioralStateDefinition]:
        """Return all definitions."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        async with self._conn.execute("SELECT data_json FROM behavioral_state_definitions") as cursor:
            rows = await cursor.fetchall()
        return [BehavioralStateDefinition.from_dict(json.loads(r["data_json"])) for r in rows]

    async def delete_definition(self, definition_id: str) -> None:
        """Delete a definition and cascade to its tracker and co-activations."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        await self._conn.execute(
            "DELETE FROM behavioral_state_definitions WHERE id = ?",
            (definition_id,),
        )
        await self._conn.execute(
            "DELETE FROM behavioral_state_trackers WHERE definition_id = ?",
            (definition_id,),
        )
        await self._conn.execute(
            "DELETE FROM state_co_activations WHERE state_a_id = ? OR state_b_id = ?",
            (definition_id, definition_id),
        )
        await self._conn.commit()

    # ── Trackers ──────────────────────────────────────────────────────────────

    async def save_tracker(self, tracker: BehavioralStateTracker) -> None:
        """Insert or replace a behavioral state tracker."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        data_json = json.dumps(tracker.to_dict())
        await self._conn.execute(
            "INSERT OR REPLACE INTO behavioral_state_trackers (definition_id, data_json) VALUES (?, ?)",
            (tracker.definition_id, data_json),
        )
        await self._conn.commit()

    async def get_tracker(self, definition_id: str) -> BehavioralStateTracker | None:
        """Retrieve a tracker by definition_id, or None if not found."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        async with self._conn.execute(
            "SELECT data_json FROM behavioral_state_trackers WHERE definition_id = ?",
            (definition_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return BehavioralStateTracker.from_dict(json.loads(row["data_json"]))

    async def list_trackers(self, lifecycle_filter: str | None = None) -> list[BehavioralStateTracker]:
        """Return all trackers, optionally filtered by lifecycle state."""
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        async with self._conn.execute("SELECT data_json FROM behavioral_state_trackers") as cursor:
            rows = await cursor.fetchall()
        trackers = [BehavioralStateTracker.from_dict(json.loads(r["data_json"])) for r in rows]
        if lifecycle_filter is not None:
            trackers = [t for t in trackers if t.lifecycle == lifecycle_filter]
        return trackers

    # ── Co-activations ────────────────────────────────────────────────────────

    async def record_co_activation(self, a_id: str, b_id: str) -> None:
        """Increment the co-activation count for the (a_id, b_id) pair.

        Stores pairs in canonical order (a_id < b_id) to avoid duplicates.
        """
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        # Canonical order so (A,B) and (B,A) map to the same row
        state_a, state_b = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        await self._conn.execute(
            """
            INSERT INTO state_co_activations (state_a_id, state_b_id, count)
            VALUES (?, ?, 1)
            ON CONFLICT(state_a_id, state_b_id) DO UPDATE SET count = count + 1
            """,
            (state_a, state_b),
        )
        await self._conn.commit()

    async def get_co_activations(self, min_count: int = 1) -> list[tuple[str, str, int]]:
        """Return co-activation pairs with count >= min_count.

        Each entry is (state_a_id, state_b_id, count).
        """
        if not self._conn:
            raise RuntimeError("BehavioralStateStore not initialized — call initialize() first")
        async with self._conn.execute(
            "SELECT state_a_id, state_b_id, count FROM state_co_activations WHERE count >= ?",
            (min_count,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [(r["state_a_id"], r["state_b_id"], r["count"]) for r in rows]
