"""Tests for hub.db memory leak fix (issue #140).

Root cause: hub.publish() calls cache.log_event() for *every* event,
including high-frequency state_changed (6/sec × 3050 entities). This
writes 3.6M rows/week to hub.db's events table. prune_events() deletes
rows but never WAL-checkpoints or VACUUMs, so SQLite page cache grows
unchecked in the aria process.

Fix:
  1. Skip logging high-frequency events (state_changed) to hub.db.
  2. WAL checkpoint + VACUUM after each prune run.
"""

import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))


from aria.hub.cache import _HIGH_FREQ_SKIP_EVENTS, CacheManager

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def cache(tmp_path):
    db_path = str(tmp_path / "test_hub.db")
    cm = CacheManager(db_path)
    await cm.initialize()
    yield cm
    await cm.close()


# ---------------------------------------------------------------------------
# Part 1: state_changed must NOT be written to hub.db events table
# ---------------------------------------------------------------------------


class TestHighFreqEventSkip:
    """state_changed must not accumulate in the hub.db events table."""

    def test_skip_set_exists(self):
        """_HIGH_FREQ_SKIP_EVENTS must be exported from cache module."""
        assert isinstance(_HIGH_FREQ_SKIP_EVENTS, frozenset)
        assert "state_changed" in _HIGH_FREQ_SKIP_EVENTS

    @pytest.mark.asyncio
    async def test_state_changed_not_stored(self, cache):
        """log_event('state_changed', ...) must not write to events table."""
        await cache.log_event(
            "state_changed",
            data={"entity_id": "light.kitchen", "new_state": "on"},
        )
        events = await cache.get_events(event_type="state_changed")
        assert events == [], "state_changed must be skipped — found row in hub.db events table"

    @pytest.mark.asyncio
    async def test_hub_events_still_stored(self, cache):
        """Non-high-freq events (cache_updated, module_registered) must still be stored."""
        await cache.log_event("cache_updated", data={"category": "entities"})
        await cache.log_event("module_registered", metadata={"module_id": "presence"})

        cache_events = await cache.get_events(event_type="cache_updated")
        module_events = await cache.get_events(event_type="module_registered")

        assert len(cache_events) >= 1, "cache_updated must still be logged"
        assert len(module_events) >= 1, "module_registered must still be logged"

    @pytest.mark.asyncio
    async def test_high_freq_skip_does_not_raise(self, cache):
        """Skipping a high-freq event must be a silent no-op, not an exception."""
        # Should not raise
        await cache.log_event("state_changed", data={"entity_id": "sensor.temp"})


# ---------------------------------------------------------------------------
# Part 2: WAL checkpoint + VACUUM after prune
# ---------------------------------------------------------------------------


class TestPruneRunsCheckpointAndVacuum:
    """prune_events() must checkpoint the WAL and VACUUM after deleting rows."""

    @pytest.mark.asyncio
    async def test_prune_events_runs_checkpoint(self, cache, tmp_path):
        """After prune_events(), PRAGMA wal_checkpoint must have been executed."""
        executed_pragmas: list[str] = []
        original_execute = cache._conn.execute

        async def spy_execute(sql, *args, **kwargs):
            if "wal_checkpoint" in sql.lower() or "vacuum" in sql.lower():
                executed_pragmas.append(sql.strip().lower())
            return await original_execute(sql, *args, **kwargs)

        cache._conn.execute = spy_execute

        await cache.prune_events(retention_days=7)

        checkpoint_ran = any("wal_checkpoint" in p for p in executed_pragmas)
        assert checkpoint_ran, f"Expected PRAGMA wal_checkpoint after prune; got: {executed_pragmas}"

    @pytest.mark.asyncio
    async def test_prune_events_runs_vacuum(self, cache):
        """After prune_events(), VACUUM must have been executed to reclaim pages."""
        executed_pragmas: list[str] = []
        original_execute = cache._conn.execute

        async def spy_execute(sql, *args, **kwargs):
            if "vacuum" in sql.lower():
                executed_pragmas.append(sql.strip().lower())
            return await original_execute(sql, *args, **kwargs)

        cache._conn.execute = spy_execute

        await cache.prune_events(retention_days=7)

        vacuum_ran = any("vacuum" in p for p in executed_pragmas)
        assert vacuum_ran, f"Expected VACUUM after prune; got: {executed_pragmas}"

    @pytest.mark.asyncio
    async def test_prune_returns_row_count(self, cache):
        """prune_events() must still return the number of rows deleted."""
        from datetime import UTC, datetime, timedelta

        # Insert an old event directly (bypass the skip-list)
        old_ts = (datetime.now(tz=UTC) - timedelta(days=10)).isoformat()
        await cache._conn.execute(
            "INSERT INTO events (timestamp, event_type) VALUES (?, ?)",
            (old_ts, "cache_updated"),
        )
        await cache._conn.commit()

        deleted = await cache.prune_events(retention_days=7)
        assert deleted >= 1, "prune_events must return count of deleted rows"
