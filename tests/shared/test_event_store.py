"""Tests for aria.shared.event_store — EventStore with SQLite persistence."""

import os

import pytest_asyncio

from aria.shared.event_store import EventStore


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create an EventStore with a temp database, initialize, and clean up."""
    db_path = str(tmp_path / "test_events.db")
    es = EventStore(db_path)
    await es.initialize()
    yield es
    await es.close()


# ── Initialization ──────────────────────────────────────────────────────


class TestInitialization:
    async def test_import(self):
        """EventStore can be imported."""
        from aria.shared.event_store import EventStore  # noqa: F811

        assert EventStore is not None

    async def test_creates_db_file(self, tmp_path):
        """initialize() creates the SQLite file on disk."""
        db_path = str(tmp_path / "subdir" / "events.db")
        es = EventStore(db_path)
        await es.initialize()
        assert os.path.exists(db_path)
        await es.close()

    async def test_wal_mode_enabled(self, store):
        """Database uses WAL journal mode."""
        cursor = await store._conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert row[0] == "wal"

    async def test_tables_and_indexes_created(self, store):
        """Schema contains the expected table and indexes."""
        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='state_change_events'"
        )
        assert await cursor.fetchone() is not None

        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_sce_%'"
        )
        indexes = {row[0] for row in await cursor.fetchall()}
        assert indexes == {"idx_sce_ts", "idx_sce_entity", "idx_sce_area", "idx_sce_domain", "idx_sce_context"}


# ── Insert Single Event ─────────────────────────────────────────────────


class TestInsertEvent:
    async def test_insert_and_count(self, store):
        """Single insert increments total count."""
        await store.insert_event(
            timestamp="2025-01-15T10:00:00+00:00",
            entity_id="light.living_room",
            domain="light",
            old_state="off",
            new_state="on",
            device_id="dev_001",
            area_id="living_room",
            attributes_json='{"brightness": 255}',
        )
        assert await store.total_count() == 1

    async def test_insert_with_nulls(self, store):
        """Nullable fields (old_state, device_id, area_id, attributes_json) accept None."""
        await store.insert_event(
            timestamp="2025-01-15T10:00:00+00:00",
            entity_id="sensor.temp",
            domain="sensor",
            old_state=None,
            new_state="22.5",
            device_id=None,
            area_id=None,
            attributes_json=None,
        )
        assert await store.total_count() == 1


# ── Insert Batch ────────────────────────────────────────────────────────


class TestInsertBatch:
    async def test_batch_insert(self, store):
        """Batch insert adds all events."""
        events = [
            ("2025-01-15T10:00:00+00:00", "light.a", "light", "off", "on", None, "room_a", None),
            ("2025-01-15T10:01:00+00:00", "light.b", "light", "on", "off", None, "room_b", None),
            ("2025-01-15T10:02:00+00:00", "sensor.c", "sensor", "20", "21", None, None, None),
        ]
        await store.insert_events_batch(events)
        assert await store.total_count() == 3

    async def test_batch_insert_empty(self, store):
        """Empty batch is a no-op."""
        await store.insert_events_batch([])
        assert await store.total_count() == 0


# ── Query by Time Window ────────────────────────────────────────────────


class TestQueryEvents:
    @pytest_asyncio.fixture(autouse=True)
    async def _seed(self, store):
        events = [
            ("2025-01-15T08:00:00+00:00", "light.a", "light", "off", "on", None, "room_a", None),
            ("2025-01-15T10:00:00+00:00", "light.b", "light", "on", "off", None, "room_a", None),
            ("2025-01-15T12:00:00+00:00", "sensor.c", "sensor", "20", "21", None, "room_b", None),
            ("2025-01-16T08:00:00+00:00", "light.a", "light", "on", "off", None, "room_a", None),
        ]
        await store.insert_events_batch(events)

    async def test_query_full_range(self, store):
        """Query spanning all events returns all."""
        rows = await store.query_events("2025-01-15T00:00:00+00:00", "2025-01-17T00:00:00+00:00")
        assert len(rows) == 4

    async def test_query_partial_range(self, store):
        """Query returns only events within the window (inclusive start, exclusive end by convention)."""
        rows = await store.query_events("2025-01-15T09:00:00+00:00", "2025-01-15T11:00:00+00:00")
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "light.b"

    async def test_query_no_results(self, store):
        """Query outside data range returns empty list."""
        rows = await store.query_events("2024-01-01T00:00:00+00:00", "2024-01-02T00:00:00+00:00")
        assert rows == []

    async def test_query_respects_limit(self, store):
        """Limit caps returned rows."""
        rows = await store.query_events("2025-01-15T00:00:00+00:00", "2025-01-17T00:00:00+00:00", limit=2)
        assert len(rows) == 2


# ── Query by Entity ─────────────────────────────────────────────────────


class TestQueryByEntity:
    @pytest_asyncio.fixture(autouse=True)
    async def _seed(self, store):
        events = [
            ("2025-01-15T08:00:00+00:00", "light.a", "light", "off", "on", None, "room_a", None),
            ("2025-01-15T10:00:00+00:00", "light.b", "light", "on", "off", None, "room_a", None),
            ("2025-01-15T12:00:00+00:00", "light.a", "light", "on", "off", None, "room_a", None),
        ]
        await store.insert_events_batch(events)

    async def test_filter_by_entity(self, store):
        rows = await store.query_by_entity("light.a", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert len(rows) == 2
        assert all(r["entity_id"] == "light.a" for r in rows)

    async def test_entity_not_found(self, store):
        rows = await store.query_by_entity("sensor.nope", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert rows == []


# ── Query by Area ───────────────────────────────────────────────────────


class TestQueryByArea:
    @pytest_asyncio.fixture(autouse=True)
    async def _seed(self, store):
        events = [
            ("2025-01-15T08:00:00+00:00", "light.a", "light", "off", "on", None, "kitchen", None),
            ("2025-01-15T10:00:00+00:00", "light.b", "light", "on", "off", None, "bedroom", None),
            ("2025-01-15T12:00:00+00:00", "sensor.c", "sensor", "20", "21", None, "kitchen", None),
        ]
        await store.insert_events_batch(events)

    async def test_filter_by_area(self, store):
        rows = await store.query_by_area("kitchen", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert len(rows) == 2
        assert all(r["area_id"] == "kitchen" for r in rows)

    async def test_area_not_found(self, store):
        rows = await store.query_by_area("garage", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert rows == []


# ── Query by Domain ─────────────────────────────────────────────────────


class TestQueryByDomain:
    @pytest_asyncio.fixture(autouse=True)
    async def _seed(self, store):
        events = [
            ("2025-01-15T08:00:00+00:00", "light.a", "light", "off", "on", None, "room_a", None),
            ("2025-01-15T10:00:00+00:00", "switch.b", "switch", "on", "off", None, "room_a", None),
            ("2025-01-15T12:00:00+00:00", "light.c", "light", "on", "off", None, "room_b", None),
        ]
        await store.insert_events_batch(events)

    async def test_filter_by_domain(self, store):
        rows = await store.query_by_domain("light", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert len(rows) == 2
        assert all(r["domain"] == "light" for r in rows)

    async def test_domain_not_found(self, store):
        rows = await store.query_by_domain("climate", "2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert rows == []


# ── Count Events ────────────────────────────────────────────────────────


class TestCountEvents:
    async def test_count_empty(self, store):
        count = await store.count_events("2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert count == 0

    async def test_count_with_data(self, store):
        events = [
            ("2025-01-15T08:00:00+00:00", "light.a", "light", "off", "on", None, None, None),
            ("2025-01-15T10:00:00+00:00", "light.b", "light", "on", "off", None, None, None),
            ("2025-01-16T08:00:00+00:00", "light.c", "light", "on", "off", None, None, None),
        ]
        await store.insert_events_batch(events)
        count = await store.count_events("2025-01-15T00:00:00+00:00", "2025-01-15T23:59:59+00:00")
        assert count == 2


# ── Prune & Total Count ─────────────────────────────────────────────────


class TestPruneAndTotalCount:
    async def test_total_count_empty(self, store):
        assert await store.total_count() == 0

    async def test_prune_before(self, store):
        events = [
            ("2025-01-10T08:00:00+00:00", "light.old", "light", "off", "on", None, None, None),
            ("2025-01-15T08:00:00+00:00", "light.mid", "light", "off", "on", None, None, None),
            ("2025-01-20T08:00:00+00:00", "light.new", "light", "off", "on", None, None, None),
        ]
        await store.insert_events_batch(events)
        assert await store.total_count() == 3

        deleted = await store.prune_before("2025-01-15T00:00:00+00:00")
        assert deleted == 1
        assert await store.total_count() == 2

    async def test_prune_nothing(self, store):
        """Pruning with a cutoff before all data deletes nothing."""
        await store.insert_event(
            timestamp="2025-01-15T10:00:00+00:00",
            entity_id="light.a",
            domain="light",
            old_state="off",
            new_state="on",
            device_id=None,
            area_id=None,
            attributes_json=None,
        )
        deleted = await store.prune_before("2025-01-01T00:00:00+00:00")
        assert deleted == 0
        assert await store.total_count() == 1

    async def test_prune_all(self, store):
        """Pruning with a future cutoff deletes everything."""
        events = [
            ("2025-01-10T08:00:00+00:00", "light.a", "light", "off", "on", None, None, None),
            ("2025-01-15T08:00:00+00:00", "light.b", "light", "off", "on", None, None, None),
        ]
        await store.insert_events_batch(events)
        deleted = await store.prune_before("2026-01-01T00:00:00+00:00")
        assert deleted == 2
        assert await store.total_count() == 0


# ── Row Structure ───────────────────────────────────────────────────────


class TestRowStructure:
    async def test_query_returns_dicts_with_expected_keys(self, store):
        """Query results are dicts with all expected columns."""
        await store.insert_event(
            timestamp="2025-01-15T10:00:00+00:00",
            entity_id="light.living_room",
            domain="light",
            old_state="off",
            new_state="on",
            device_id="dev_001",
            area_id="living_room",
            attributes_json='{"brightness": 255}',
        )
        rows = await store.query_events("2025-01-15T00:00:00+00:00", "2025-01-16T00:00:00+00:00")
        assert len(rows) == 1
        row = rows[0]
        expected_keys = {
            "id",
            "timestamp",
            "entity_id",
            "domain",
            "old_state",
            "new_state",
            "device_id",
            "area_id",
            "attributes_json",
            "context_parent_id",
        }
        assert expected_keys == set(row.keys())
        assert row["entity_id"] == "light.living_room"
        assert row["new_state"] == "on"
        assert row["attributes_json"] == '{"brightness": 255}'
