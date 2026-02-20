# Phase 1: Event Store + Entity Graph — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent event store for HA state_changed events and a centralized entity→device→area graph, forming the data foundation for ARIA Roadmap 2.0.

**Architecture:** New `EventStore` (SQLite WAL-mode, separate `events.db`) captures every state_changed event from activity_monitor with resolved device_id/area_id. New `EntityGraph` centralizes the entity→device→area hierarchy currently duplicated across 3 modules. Both are owned by IntelligenceHub and available to all modules.

**Tech Stack:** Python 3.12, aiosqlite, SQLite WAL mode, pytest, FastAPI

**PRD:** `tasks/prd.json` (11 tasks, P1-01 through P1-11)

**Quality Gates:** `.venv/bin/python -m pytest tests/ --timeout=120 -x -q`

---

## Progress Tracking

Initialize `progress.txt` at start of execution. Append after each batch.

---

### Task 1: EventStore class with SQLite schema (PRD P1-01)

**Files:**
- Create: `aria/shared/event_store.py`
- Create: `tests/shared/test_event_store.py`

**Step 1: Write the failing test**

```python
# tests/shared/test_event_store.py
"""Tests for EventStore — persistent HA state_changed event storage."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from aria.shared.event_store import EventStore


@pytest.fixture
def event_store(tmp_path):
    """Create a temporary EventStore for testing."""
    db_path = str(tmp_path / "test_events.db")
    store = EventStore(db_path)
    asyncio.get_event_loop().run_until_complete(store.initialize())
    yield store
    asyncio.get_event_loop().run_until_complete(store.close())


def test_event_store_import():
    """EventStore class can be imported."""
    from aria.shared.event_store import EventStore
    assert EventStore is not None


def test_event_store_creates_db(tmp_path):
    """EventStore creates database file on initialize."""
    db_path = str(tmp_path / "test_events.db")
    store = EventStore(db_path)
    asyncio.get_event_loop().run_until_complete(store.initialize())
    assert Path(db_path).exists()
    asyncio.get_event_loop().run_until_complete(store.close())
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py::test_event_store_import -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.shared.event_store'`

**Step 3: Write minimal implementation**

```python
# aria/shared/event_store.py
"""Persistent storage for HA state_changed events.

Stores every filtered state_changed event from activity_monitor with
resolved device_id and area_id for downstream consumers: ML segment
builder, I&W pattern miner, automation generator, shadow engine.

Uses a separate SQLite database (events.db) to avoid contention with
the hub's cache.db. WAL mode enables concurrent reads during writes.
"""

import os
from datetime import UTC, datetime
from typing import Any

import aiosqlite

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS state_change_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    old_state TEXT,
    new_state TEXT,
    device_id TEXT,
    area_id TEXT,
    attributes_json TEXT
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sce_ts ON state_change_events(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_sce_entity ON state_change_events(entity_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_sce_area ON state_change_events(area_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_sce_domain ON state_change_events(domain, timestamp)",
]


class EventStore:
    """SQLite-backed persistent store for HA state_changed events."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self):
        """Create database and schema."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._conn.execute(idx_sql)
        await self._conn.commit()

    async def close(self):
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py -v --timeout=30`
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add aria/shared/event_store.py tests/shared/test_event_store.py
git commit -m "feat(event-store): add EventStore class with SQLite schema (P1-01)"
```

---

### Task 2: EventStore write and read methods (PRD P1-02)

**Files:**
- Modify: `aria/shared/event_store.py`
- Modify: `tests/shared/test_event_store.py`

**Step 1: Write the failing tests**

Add to `tests/shared/test_event_store.py`:

```python
import pytest_asyncio

@pytest_asyncio.fixture
async def astore(tmp_path):
    """Async EventStore fixture."""
    db_path = str(tmp_path / "test_events.db")
    store = EventStore(db_path)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_insert_and_query(astore):
    """Insert an event and query it back."""
    await astore.insert_event(
        timestamp="2026-02-20T10:00:00",
        entity_id="light.bedroom",
        domain="light",
        old_state="off",
        new_state="on",
        device_id="device_abc",
        area_id="bedroom",
    )
    events = await astore.query_events("2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(events) == 1
    assert events[0]["entity_id"] == "light.bedroom"
    assert events[0]["new_state"] == "on"
    assert events[0]["area_id"] == "bedroom"


@pytest.mark.asyncio
async def test_query_by_entity(astore):
    """Query events filtered by entity_id."""
    await astore.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on")
    await astore.insert_event("2026-02-20T10:05:00", "light.kitchen", "light", "off", "on")
    events = await astore.query_by_entity("light.bedroom", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(events) == 1
    assert events[0]["entity_id"] == "light.bedroom"


@pytest.mark.asyncio
async def test_query_by_area(astore):
    """Query events filtered by area_id."""
    await astore.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on", area_id="bedroom")
    await astore.insert_event("2026-02-20T10:05:00", "light.kitchen", "light", "off", "on", area_id="kitchen")
    events = await astore.query_by_area("bedroom", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(events) == 1
    assert events[0]["area_id"] == "bedroom"


@pytest.mark.asyncio
async def test_query_by_domain(astore):
    """Query events filtered by domain."""
    await astore.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on")
    await astore.insert_event("2026-02-20T10:05:00", "binary_sensor.motion", "binary_sensor", "off", "on")
    events = await astore.query_by_domain("light", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(events) == 1
    assert events[0]["domain"] == "light"


@pytest.mark.asyncio
async def test_count_events(astore):
    """Count events in a time window."""
    await astore.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on")
    await astore.insert_event("2026-02-20T10:05:00", "light.kitchen", "light", "off", "on")
    count = await astore.count_events("2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert count == 2


@pytest.mark.asyncio
async def test_insert_with_attributes(astore):
    """Insert event with attributes_json."""
    await astore.insert_event(
        "2026-02-20T10:00:00", "light.bedroom", "light", "off", "on",
        attributes_json='{"brightness": 200, "color_temp": 4000}',
    )
    events = await astore.query_events("2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert events[0]["attributes_json"] == '{"brightness": 200, "color_temp": 4000}'
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py::test_insert_and_query -v`
Expected: FAIL — `AttributeError: 'EventStore' object has no attribute 'insert_event'`

**Step 3: Write implementation**

Add to `EventStore` class in `aria/shared/event_store.py`:

```python
    async def insert_event(
        self,
        timestamp: str,
        entity_id: str,
        domain: str,
        old_state: str | None = None,
        new_state: str | None = None,
        device_id: str | None = None,
        area_id: str | None = None,
        attributes_json: str | None = None,
    ) -> None:
        """Insert a single state_changed event."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        await self._conn.execute(
            """INSERT INTO state_change_events
               (timestamp, entity_id, domain, old_state, new_state, device_id, area_id, attributes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (timestamp, entity_id, domain, old_state, new_state, device_id, area_id, attributes_json),
        )
        await self._conn.commit()

    async def insert_events_batch(self, events: list[tuple]) -> None:
        """Insert multiple events in a single transaction.

        Each tuple: (timestamp, entity_id, domain, old_state, new_state, device_id, area_id, attributes_json)
        """
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        await self._conn.executemany(
            """INSERT INTO state_change_events
               (timestamp, entity_id, domain, old_state, new_state, device_id, area_id, attributes_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            events,
        )
        await self._conn.commit()

    async def query_events(
        self, start: str, end: str, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Query events in a time window."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            """SELECT * FROM state_change_events
               WHERE timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC LIMIT ?""",
            (start, end, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_by_entity(
        self, entity_id: str, start: str, end: str, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Query events for a specific entity in a time window."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            """SELECT * FROM state_change_events
               WHERE entity_id = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC LIMIT ?""",
            (entity_id, start, end, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_by_area(
        self, area_id: str, start: str, end: str, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Query events for a specific area in a time window."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            """SELECT * FROM state_change_events
               WHERE area_id = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC LIMIT ?""",
            (area_id, start, end, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_by_domain(
        self, domain: str, start: str, end: str, limit: int = 10000
    ) -> list[dict[str, Any]]:
        """Query events for a specific domain in a time window."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            """SELECT * FROM state_change_events
               WHERE domain = ? AND timestamp >= ? AND timestamp <= ?
               ORDER BY timestamp ASC LIMIT ?""",
            (domain, start, end, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def count_events(self, start: str, end: str) -> int:
        """Count events in a time window."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            """SELECT COUNT(*) FROM state_change_events
               WHERE timestamp >= ? AND timestamp <= ?""",
            (start, end),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py -v --timeout=30`
Expected: All PASSED

**Step 5: Commit**

```bash
git add aria/shared/event_store.py tests/shared/test_event_store.py
git commit -m "feat(event-store): add insert and query methods (P1-02)"
```

---

### Task 3: EventStore retention and pruning (PRD P1-03)

**Files:**
- Modify: `aria/shared/event_store.py`
- Modify: `tests/shared/test_event_store.py`

**Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_prune_events(astore):
    """Prune removes events older than cutoff."""
    await astore.insert_event("2026-01-01T10:00:00", "light.old", "light", "off", "on")
    await astore.insert_event("2026-02-20T10:00:00", "light.new", "light", "off", "on")
    pruned = await astore.prune_before("2026-02-01T00:00:00")
    assert pruned == 1
    remaining = await astore.query_events("2025-01-01T00:00:00", "2027-01-01T00:00:00")
    assert len(remaining) == 1
    assert remaining[0]["entity_id"] == "light.new"


@pytest.mark.asyncio
async def test_prune_empty(astore):
    """Prune on empty store returns 0."""
    pruned = await astore.prune_before("2026-02-01T00:00:00")
    assert pruned == 0


@pytest.mark.asyncio
async def test_event_count_total(astore):
    """Total event count across all time."""
    await astore.insert_event("2026-02-20T10:00:00", "light.a", "light", "off", "on")
    await astore.insert_event("2026-02-20T10:05:00", "light.b", "light", "off", "on")
    count = await astore.total_count()
    assert count == 2
```

**Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py::test_prune_events -v`
Expected: FAIL — `AttributeError: 'EventStore' object has no attribute 'prune_before'`

**Step 3: Implement**

Add to `EventStore`:

```python
    async def prune_before(self, cutoff: str) -> int:
        """Delete events older than cutoff timestamp. Returns count deleted."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute(
            "DELETE FROM state_change_events WHERE timestamp < ?", (cutoff,)
        )
        await self._conn.commit()
        return cursor.rowcount

    async def total_count(self) -> int:
        """Total number of events in the store."""
        if not self._conn:
            raise RuntimeError("EventStore not initialized")
        cursor = await self._conn.execute("SELECT COUNT(*) FROM state_change_events")
        row = await cursor.fetchone()
        return row[0] if row else 0
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/shared/test_event_store.py -v --timeout=30`
Expected: All PASSED

**Step 5: Commit**

```bash
git add aria/shared/event_store.py tests/shared/test_event_store.py
git commit -m "feat(event-store): add retention pruning and total count (P1-03)"
```

---

### Task 4: EntityGraph class (PRD P1-04)

**Files:**
- Create: `aria/shared/entity_graph.py`
- Create: `tests/shared/test_entity_graph.py`

**Step 1: Write the failing tests**

```python
# tests/shared/test_entity_graph.py
"""Tests for EntityGraph — centralized entity→device→area resolution."""

import pytest

from aria.shared.entity_graph import EntityGraph


@pytest.fixture
def sample_entities():
    """Sample entity data matching discovery cache format."""
    return {
        "light.bedroom_lamp": {
            "entity_id": "light.bedroom_lamp",
            "device_id": "dev_001",
            "area_id": None,  # inherits from device
        },
        "light.kitchen_main": {
            "entity_id": "light.kitchen_main",
            "device_id": "dev_002",
            "area_id": "kitchen",  # direct area override
        },
        "binary_sensor.bedroom_motion": {
            "entity_id": "binary_sensor.bedroom_motion",
            "device_id": "dev_003",
            "area_id": None,
        },
    }


@pytest.fixture
def sample_devices():
    return {
        "dev_001": {"device_id": "dev_001", "area_id": "bedroom", "name": "Bedroom Lamp"},
        "dev_002": {"device_id": "dev_002", "area_id": "kitchen", "name": "Kitchen Light"},
        "dev_003": {"device_id": "dev_003", "area_id": "bedroom", "name": "Bedroom Motion"},
    }


@pytest.fixture
def sample_areas():
    return [
        {"area_id": "bedroom", "name": "Bedroom"},
        {"area_id": "kitchen", "name": "Kitchen"},
    ]


@pytest.fixture
def graph(sample_entities, sample_devices, sample_areas):
    g = EntityGraph()
    g.update(sample_entities, sample_devices, sample_areas)
    return g


def test_get_area_via_device(graph):
    """Entity without direct area_id resolves through device."""
    assert graph.get_area("light.bedroom_lamp") == "bedroom"


def test_get_area_direct(graph):
    """Entity with direct area_id uses it."""
    assert graph.get_area("light.kitchen_main") == "kitchen"


def test_get_area_unknown_entity(graph):
    """Unknown entity returns None."""
    assert graph.get_area("light.nonexistent") is None


def test_get_device(graph):
    """Resolve entity to device info."""
    device = graph.get_device("light.bedroom_lamp")
    assert device is not None
    assert device["name"] == "Bedroom Lamp"


def test_entities_in_area(graph):
    """Get all entities in an area."""
    bedroom_entities = graph.entities_in_area("bedroom")
    entity_ids = {e["entity_id"] for e in bedroom_entities}
    assert "light.bedroom_lamp" in entity_ids
    assert "binary_sensor.bedroom_motion" in entity_ids
    assert "light.kitchen_main" not in entity_ids


def test_entities_by_domain(graph):
    """Get all entities of a domain."""
    lights = graph.entities_by_domain("light")
    entity_ids = {e["entity_id"] for e in lights}
    assert "light.bedroom_lamp" in entity_ids
    assert "light.kitchen_main" in entity_ids
    assert "binary_sensor.bedroom_motion" not in entity_ids


def test_all_areas(graph):
    """Get list of all known areas."""
    areas = graph.all_areas()
    area_ids = {a["area_id"] for a in areas}
    assert "bedroom" in area_ids
    assert "kitchen" in area_ids


def test_update_refreshes_data(graph, sample_entities, sample_devices):
    """Calling update() refreshes the graph."""
    new_entities = {**sample_entities, "switch.garage": {"entity_id": "switch.garage", "device_id": "dev_004", "area_id": None}}
    new_devices = {**sample_devices, "dev_004": {"device_id": "dev_004", "area_id": "garage", "name": "Garage Switch"}}
    new_areas = [{"area_id": "bedroom", "name": "Bedroom"}, {"area_id": "kitchen", "name": "Kitchen"}, {"area_id": "garage", "name": "Garage"}]
    graph.update(new_entities, new_devices, new_areas)
    assert graph.get_area("switch.garage") == "garage"
```

**Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/shared/test_entity_graph.py::test_get_area_via_device -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement**

```python
# aria/shared/entity_graph.py
"""Centralized entity→device→area graph.

Single source of truth for resolving the HA three-tier hierarchy:
  entity → device → area

Replaces per-module resolution logic in discovery, presence, and
snapshot collector. Updated from discovery cache on cache_updated events.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EntityGraph:
    """In-memory entity→device→area graph built from discovery cache."""

    def __init__(self):
        self._entities: dict[str, dict[str, Any]] = {}
        self._devices: dict[str, dict[str, Any]] = {}
        self._areas: list[dict[str, Any]] = []
        self._area_index: dict[str, list[dict[str, Any]]] = {}  # area_id → [entities]

    def update(
        self,
        entities: dict[str, dict[str, Any]],
        devices: dict[str, dict[str, Any]],
        areas: list[dict[str, Any]],
    ) -> None:
        """Rebuild the graph from fresh discovery data."""
        self._entities = entities
        self._devices = devices
        self._areas = areas
        self._rebuild_area_index()
        logger.debug(
            "EntityGraph updated: %d entities, %d devices, %d areas",
            len(entities), len(devices), len(areas),
        )

    def _rebuild_area_index(self) -> None:
        """Rebuild the area→entities reverse index."""
        self._area_index = {}
        for entity_id, entity in self._entities.items():
            area_id = self._resolve_area(entity)
            if area_id:
                self._area_index.setdefault(area_id, []).append(
                    {**entity, "entity_id": entity_id}
                )

    def _resolve_area(self, entity: dict[str, Any]) -> str | None:
        """Resolve area for an entity: direct area_id, then device fallback."""
        # Entity-level area takes priority
        if entity.get("area_id"):
            return entity["area_id"]
        # Fall back to device's area
        device_id = entity.get("device_id")
        if device_id and device_id in self._devices:
            return self._devices[device_id].get("area_id")
        return None

    def get_area(self, entity_id: str) -> str | None:
        """Get area_id for an entity (entity→device→area chain)."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        return self._resolve_area(entity)

    def get_device(self, entity_id: str) -> dict[str, Any] | None:
        """Get device info for an entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        device_id = entity.get("device_id")
        return self._devices.get(device_id) if device_id else None

    def entities_in_area(self, area_id: str) -> list[dict[str, Any]]:
        """Get all entities in an area."""
        return self._area_index.get(area_id, [])

    def entities_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all entities of a specific domain."""
        return [
            {**e, "entity_id": eid}
            for eid, e in self._entities.items()
            if eid.startswith(f"{domain}.")
        ]

    def all_areas(self) -> list[dict[str, Any]]:
        """Get all known areas."""
        return list(self._areas)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def area_count(self) -> int:
        return len(self._areas)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/shared/test_entity_graph.py -v --timeout=30`
Expected: All PASSED

**Step 5: Commit**

```bash
git add aria/shared/entity_graph.py tests/shared/test_entity_graph.py
git commit -m "feat(entity-graph): add EntityGraph with resolution methods (P1-04)"
```

---

### Task 5: Wire EntityGraph into hub core (PRD P1-05)

**Files:**
- Modify: `aria/hub/core.py` — add `entity_graph` attribute, refresh on cache_updated
- Create: `tests/hub/test_hub_entity_graph.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_hub_entity_graph.py
"""Tests for EntityGraph integration with IntelligenceHub."""

import pytest

from aria.shared.entity_graph import EntityGraph


def test_hub_has_entity_graph(mock_hub):
    """Hub exposes an EntityGraph instance."""
    assert hasattr(mock_hub, "entity_graph")
    assert isinstance(mock_hub.entity_graph, EntityGraph)
```

Note: `mock_hub` is the existing hub fixture from `tests/conftest.py`. Check what fixture name is used — adjust accordingly.

**Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/hub/test_hub_entity_graph.py::test_hub_has_entity_graph -v`
Expected: FAIL — `AttributeError: hub has no attribute 'entity_graph'`

**Step 3: Modify `aria/hub/core.py`**

In `IntelligenceHub.__init__()`, add:
```python
from aria.shared.entity_graph import EntityGraph
# ... in __init__:
self.entity_graph = EntityGraph()
```

In the `set_cache()` method (or via a subscriber), add EntityGraph refresh logic:
```python
# After set_cache writes to SQLite, if category is entities/devices/areas:
if category in ("entities", "devices", "areas"):
    self._refresh_entity_graph()

def _refresh_entity_graph(self):
    """Rebuild entity graph from current cache data."""
    try:
        entities_cache = ... # get from in-memory or SQLite
        devices_cache = ...
        areas_cache = ...
        self.entity_graph.update(
            entities_cache.get("data", {}),
            devices_cache.get("data", {}),
            areas_cache.get("data", []),
        )
    except Exception as e:
        logger.warning("Failed to refresh entity graph: %s", e)
```

Implementation note: The exact integration depends on how `set_cache` is called — it goes through `CacheManager`. The cleanest approach is to subscribe to `cache_updated` events in the hub's own event handler. Read `core.py` carefully for the publish/subscribe pattern before implementing.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_hub_entity_graph.py -v --timeout=30`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/hub/core.py tests/hub/test_hub_entity_graph.py
git commit -m "feat(entity-graph): wire EntityGraph into IntelligenceHub (P1-05)"
```

---

### Task 6: Wire EventStore into hub core (PRD P1-06)

**Files:**
- Modify: `aria/hub/core.py` — add `event_store` attribute
- Create: `tests/hub/test_hub_event_store.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_hub_event_store.py
"""Tests for EventStore integration with IntelligenceHub."""

import pytest

from aria.shared.event_store import EventStore


def test_hub_has_event_store(mock_hub):
    """Hub exposes an EventStore instance."""
    assert hasattr(mock_hub, "event_store")
    assert isinstance(mock_hub.event_store, EventStore)
```

**Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/hub/test_hub_event_store.py::test_hub_has_event_store -v`
Expected: FAIL

**Step 3: Modify `aria/hub/core.py`**

In `IntelligenceHub.__init__()`:
```python
from aria.shared.event_store import EventStore

# EventStore lives alongside hub.db
events_db_path = str(Path(db_path).parent / "events.db")
self.event_store = EventStore(events_db_path)
```

In the hub's `start()` or `initialize()` method:
```python
await self.event_store.initialize()
```

In `shutdown()`:
```python
await self.event_store.close()
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_hub_event_store.py -v --timeout=30`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/hub/core.py tests/hub/test_hub_event_store.py
git commit -m "feat(event-store): wire EventStore into IntelligenceHub (P1-06)"
```

---

### Task 7: Activity monitor persists events to EventStore (PRD P1-07)

**Files:**
- Modify: `aria/modules/activity_monitor.py:360-401` — `_handle_state_changed()`
- Create: `tests/hub/test_activity_event_persistence.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_activity_event_persistence.py
"""Tests for activity_monitor writing events to EventStore."""

import pytest
import pytest_asyncio

from aria.shared.event_store import EventStore


@pytest.mark.asyncio
async def test_state_changed_persists_to_event_store(mock_hub, tmp_path):
    """When activity_monitor handles a state_changed, event appears in EventStore."""
    # Set up EventStore on the mock hub
    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()
    mock_hub.event_store = store

    # Set up EntityGraph with test data
    from aria.shared.entity_graph import EntityGraph
    mock_hub.entity_graph = EntityGraph()
    mock_hub.entity_graph.update(
        {"light.bedroom": {"entity_id": "light.bedroom", "device_id": "dev1", "area_id": None}},
        {"dev1": {"device_id": "dev1", "area_id": "bedroom"}},
        [{"area_id": "bedroom", "name": "Bedroom"}],
    )

    # Import and create activity monitor
    from aria.modules.activity_monitor import ActivityMonitor
    monitor = ActivityMonitor(mock_hub, "http://test:8123", "test-token")

    # Simulate a state_changed event
    monitor._handle_state_changed({
        "entity_id": "light.bedroom",
        "old_state": {"state": "off", "attributes": {"friendly_name": "Bedroom Light"}},
        "new_state": {"state": "on", "attributes": {"friendly_name": "Bedroom Light", "brightness": 200}},
    })

    # Allow async event persist to complete
    import asyncio
    await asyncio.sleep(0.1)

    # Verify event was stored
    events = await store.query_events("2020-01-01T00:00:00", "2030-01-01T00:00:00")
    assert len(events) >= 1
    event = events[0]
    assert event["entity_id"] == "light.bedroom"
    assert event["old_state"] == "off"
    assert event["new_state"] == "on"
    assert event["area_id"] == "bedroom"
    assert event["domain"] == "light"

    await store.close()
```

**Step 2: Run to verify fails**

Run: `.venv/bin/python -m pytest tests/hub/test_activity_event_persistence.py -v`
Expected: FAIL — event not persisted (activity_monitor doesn't write to EventStore yet)

**Step 3: Modify `_handle_state_changed()` in `aria/modules/activity_monitor.py`**

After line 401 (`self._recent_events.append(event)`), add event persistence:

```python
        # Persist to EventStore (non-blocking)
        if hasattr(self.hub, "event_store") and self.hub.event_store:
            try:
                area_id = None
                device_id = None
                if hasattr(self.hub, "entity_graph"):
                    area_id = self.hub.entity_graph.get_area(entity_id)
                    device_info = self.hub.entity_graph.get_device(entity_id)
                    device_id = device_info.get("device_id") if device_info else None
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.hub.event_store.insert_event(
                        timestamp=event["timestamp"],
                        entity_id=entity_id,
                        domain=domain,
                        old_state=from_state,
                        new_state=to_state,
                        device_id=device_id,
                        area_id=area_id,
                        attributes_json=json.dumps(attrs) if attrs else None,
                    )
                )
            except Exception as e:
                self.logger.debug("Event store persist failed: %s", e)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_activity_event_persistence.py -v --timeout=30`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/activity_monitor.py tests/hub/test_activity_event_persistence.py
git commit -m "feat(event-store): activity monitor persists events to EventStore (P1-07)"
```

---

### Task 8: EventStore pruning timer (PRD P1-08)

**Files:**
- Modify: `aria/hub/core.py` — schedule daily pruning
- Modify: `aria/hub/config_defaults.py` — add `events.retention_days` config
- Modify: `tests/hub/test_hub_event_store.py` — add pruning timer test

**Step 1: Write the failing test**

```python
# Add to tests/hub/test_hub_event_store.py

@pytest.mark.asyncio
async def test_pruning_timer(mock_hub, tmp_path):
    """Hub schedules event pruning and it respects retention config."""
    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()

    # Insert old and new events
    await store.insert_event("2025-01-01T10:00:00", "light.old", "light", "off", "on")
    await store.insert_event("2026-02-20T10:00:00", "light.new", "light", "off", "on")

    # Prune with 90-day retention from 2026-02-20
    from datetime import datetime, timedelta
    cutoff = (datetime(2026, 2, 20) - timedelta(days=90)).isoformat()
    pruned = await store.prune_before(cutoff)
    assert pruned == 1

    remaining = await store.query_events("2020-01-01", "2030-01-01")
    assert len(remaining) == 1

    await store.close()
```

**Step 2: Run to verify it fails/passes as appropriate**

**Step 3: Add config default**

Add to `aria/hub/config_defaults.py` `CONFIG_DEFAULTS` list:
```python
{
    "key": "events.retention_days",
    "default_value": "90",
    "value_type": "number",
    "label": "Event Retention (Days)",
    "description": "How many days of raw state_changed events to keep in the event store.",
    "category": "Event Store",
    "min_value": 7,
    "max_value": 365,
    "step": 1,
},
```

**Step 4: Schedule pruning in hub startup**

In the hub's initialization (after EventStore is initialized), schedule:
```python
await self.schedule_task(
    task_id="event_store_prune",
    coro=self._prune_event_store,
    interval=timedelta(hours=24),
    run_immediately=False,
)
```

Where `_prune_event_store` reads `events.retention_days` from config and calls `event_store.prune_before()`.

**Step 5: Commit**

```bash
git add aria/hub/core.py aria/hub/config_defaults.py tests/hub/test_hub_event_store.py
git commit -m "feat(event-store): add daily pruning timer with configurable retention (P1-08)"
```

---

### Task 9: EventStore API endpoints (PRD P1-09)

**Files:**
- Modify: `aria/hub/api.py` — add `/api/events` endpoint
- Create: `tests/hub/test_api_events.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_api_events.py
"""Tests for EventStore HTTP API endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_get_events_endpoint(test_client, mock_hub, tmp_path):
    """GET /api/events returns events from the store."""
    import asyncio
    from aria.shared.event_store import EventStore

    store = EventStore(str(tmp_path / "events.db"))
    asyncio.get_event_loop().run_until_complete(store.initialize())
    asyncio.get_event_loop().run_until_complete(
        store.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on", area_id="bedroom")
    )
    mock_hub.event_store = store

    response = test_client.get("/api/events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["entity_id"] == "light.bedroom"

    asyncio.get_event_loop().run_until_complete(store.close())


def test_get_events_with_filters(test_client, mock_hub, tmp_path):
    """GET /api/events supports entity_id, area_id, domain filters."""
    import asyncio
    from aria.shared.event_store import EventStore

    store = EventStore(str(tmp_path / "events.db"))
    asyncio.get_event_loop().run_until_complete(store.initialize())
    asyncio.get_event_loop().run_until_complete(
        store.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on", area_id="bedroom")
    )
    asyncio.get_event_loop().run_until_complete(
        store.insert_event("2026-02-20T10:05:00", "light.kitchen", "light", "off", "on", area_id="kitchen")
    )
    mock_hub.event_store = store

    # Filter by area
    response = test_client.get("/api/events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59&area_id=bedroom")
    assert response.status_code == 200
    assert len(response.json()["events"]) == 1

    asyncio.get_event_loop().run_until_complete(store.close())
```

Note: Adjust test fixtures to match the existing test pattern in `tests/hub/`. Check `conftest.py` for `test_client` and `mock_hub` fixture names.

**Step 2: Run to verify fails**

**Step 3: Add endpoint to `aria/hub/api.py`**

```python
@app.get("/api/events")
async def get_events(
    start: str,
    end: str,
    entity_id: str | None = None,
    area_id: str | None = None,
    domain: str | None = None,
    limit: int = 1000,
):
    """Query state_changed events from the event store."""
    if not hasattr(hub, "event_store") or not hub.event_store:
        return JSONResponse({"error": "Event store not available"}, status_code=503)

    if entity_id:
        events = await hub.event_store.query_by_entity(entity_id, start, end, limit)
    elif area_id:
        events = await hub.event_store.query_by_area(area_id, start, end, limit)
    elif domain:
        events = await hub.event_store.query_by_domain(domain, start, end, limit)
    else:
        events = await hub.event_store.query_events(start, end, limit)

    return {"events": events, "count": len(events)}
```

Also add `GET /api/events/stats`:
```python
@app.get("/api/events/stats")
async def get_event_stats():
    """Get event store statistics."""
    if not hasattr(hub, "event_store") or not hub.event_store:
        return JSONResponse({"error": "Event store not available"}, status_code=503)
    total = await hub.event_store.total_count()
    return {"total_events": total}
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api_events.py -v --timeout=30`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_events.py
git commit -m "feat(event-store): add /api/events query endpoints (P1-09)"
```

---

### Task 10: Integration test — full event flow (PRD P1-10)

**Files:**
- Create: `tests/integration/test_event_flow.py`

**Step 1: Write the integration test**

```python
# tests/integration/test_event_flow.py
"""Integration test: state_changed → EventStore → API → EntityGraph resolution."""

import asyncio
import json

import pytest

from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore


@pytest.mark.asyncio
async def test_full_event_flow(tmp_path):
    """End-to-end: event insertion → query → area resolution."""
    # 1. Set up EventStore
    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()

    # 2. Set up EntityGraph
    graph = EntityGraph()
    graph.update(
        entities={
            "light.bedroom_lamp": {"entity_id": "light.bedroom_lamp", "device_id": "dev1", "area_id": None},
            "binary_sensor.kitchen_motion": {"entity_id": "binary_sensor.kitchen_motion", "device_id": "dev2", "area_id": None},
        },
        devices={
            "dev1": {"device_id": "dev1", "area_id": "bedroom", "name": "Bedroom Lamp"},
            "dev2": {"device_id": "dev2", "area_id": "kitchen", "name": "Kitchen Motion"},
        },
        areas=[
            {"area_id": "bedroom", "name": "Bedroom"},
            {"area_id": "kitchen", "name": "Kitchen"},
        ],
    )

    # 3. Simulate activity_monitor persisting events (with area resolution)
    test_events = [
        ("light.bedroom_lamp", "light", "off", "on"),
        ("binary_sensor.kitchen_motion", "binary_sensor", "off", "on"),
        ("light.bedroom_lamp", "light", "on", "off"),
    ]
    for entity_id, domain, old, new in test_events:
        area_id = graph.get_area(entity_id)
        device = graph.get_device(entity_id)
        device_id = device.get("device_id") if device else None
        await store.insert_event(
            timestamp=f"2026-02-20T10:{test_events.index((entity_id, domain, old, new)):02d}:00",
            entity_id=entity_id,
            domain=domain,
            old_state=old,
            new_state=new,
            device_id=device_id,
            area_id=area_id,
        )

    # 4. Verify: query all events
    all_events = await store.query_events("2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(all_events) == 3

    # 5. Verify: query by area
    bedroom_events = await store.query_by_area("bedroom", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(bedroom_events) == 2  # lamp on + lamp off
    assert all(e["area_id"] == "bedroom" for e in bedroom_events)

    kitchen_events = await store.query_by_area("kitchen", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(kitchen_events) == 1

    # 6. Verify: query by domain
    light_events = await store.query_by_domain("light", "2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert len(light_events) == 2

    # 7. Verify: entity graph consistency
    assert graph.get_area("light.bedroom_lamp") == "bedroom"
    assert graph.get_area("binary_sensor.kitchen_motion") == "kitchen"
    bedroom_entities = graph.entities_in_area("bedroom")
    assert any(e["entity_id"] == "light.bedroom_lamp" for e in bedroom_entities)

    # 8. Verify: count
    total = await store.count_events("2026-02-20T00:00:00", "2026-02-20T23:59:59")
    assert total == 3

    await store.close()
```

**Step 2: Run**

Run: `.venv/bin/python -m pytest tests/integration/test_event_flow.py -v --timeout=60`
Expected: PASS (all prior tasks should make this pass)

**Step 3: Commit**

```bash
git add tests/integration/test_event_flow.py
git commit -m "test: add end-to-end event flow integration test (P1-10)"
```

---

### Task 11: Regression check — existing tests pass (PRD P1-11)

**Step 1: Check available memory**

Run: `free -h | awk '/Mem:/{print $7}'`
If < 4G: run targeted suites instead of full.

**Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ --timeout=120 -x -q`
Expected: ~1584+ tests PASSED (new tests add to the count)

**Step 3: If any fail, fix regressions**

Focus on: `tests/hub/` tests that mock IntelligenceHub — they may need `event_store` and `entity_graph` attributes added to mock fixtures in `tests/conftest.py`.

**Step 4: Final commit**

```bash
git add -A  # stage any fixture updates
git commit -m "test: fix mock hub fixtures for EventStore/EntityGraph, all tests pass (P1-11)"
```

---

## Quality Gates

Run between every batch of tasks:

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

## Batch Grouping

| Batch | Tasks | Description |
|-------|-------|-------------|
| 1 | Tasks 1-3 | EventStore class (standalone, no hub changes) |
| 2 | Task 4 | EntityGraph class (standalone, no hub changes) |
| 3 | Tasks 5-6 | Wire both into hub core |
| 4 | Tasks 7-8 | Activity monitor integration + pruning timer |
| 5 | Task 9 | API endpoints |
| 6 | Tasks 10-11 | Integration test + regression check |

Run quality gates between each batch.
