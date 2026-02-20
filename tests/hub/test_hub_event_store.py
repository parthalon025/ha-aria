# tests/hub/test_hub_event_store.py
"""Tests for EventStore integration with IntelligenceHub."""

from datetime import datetime, timedelta

import pytest

from aria.hub.core import IntelligenceHub
from aria.shared.event_store import EventStore


def test_hub_has_event_store(tmp_path):
    """Hub exposes an EventStore instance."""
    hub = IntelligenceHub(str(tmp_path / "hub.db"))
    assert hasattr(hub, "event_store")
    assert isinstance(hub.event_store, EventStore)


@pytest.mark.asyncio
async def test_event_store_initialized_on_hub_start(tmp_path):
    """EventStore is initialized when hub starts."""
    hub = IntelligenceHub(str(tmp_path / "hub.db"))
    await hub.initialize()
    try:
        # EventStore should be usable (insert should not raise)
        await hub.event_store.insert_event(
            timestamp="2026-02-20T10:00:00",
            entity_id="light.test",
            domain="light",
            old_state="off",
            new_state="on",
        )
        count = await hub.event_store.total_count()
        assert count == 1
    finally:
        await hub.shutdown()


@pytest.mark.asyncio
async def test_event_store_closed_on_hub_shutdown(tmp_path):
    """EventStore is closed when hub shuts down."""
    hub = IntelligenceHub(str(tmp_path / "hub.db"))
    await hub.initialize()
    await hub.shutdown()
    # After shutdown, EventStore conn should be None
    assert hub.event_store._conn is None


@pytest.mark.asyncio
async def test_pruning_timer(tmp_path):
    """Hub prune_event_store method removes old events based on retention."""
    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()

    # Insert old and new events
    await store.insert_event("2025-01-01T10:00:00", "light.old", "light", "off", "on")
    await store.insert_event("2026-02-20T10:00:00", "light.new", "light", "off", "on")

    # Prune with 90-day retention from 2026-02-20
    cutoff = (datetime(2026, 2, 20) - timedelta(days=90)).isoformat()
    pruned = await store.prune_before(cutoff)
    assert pruned == 1

    remaining = await store.query_events("2020-01-01", "2030-01-01")
    assert len(remaining) == 1
    assert remaining[0]["entity_id"] == "light.new"

    await store.close()


@pytest.mark.asyncio
async def test_hub_prune_event_store_method(tmp_path):
    """Hub._prune_event_store reads config and prunes accordingly."""
    hub = IntelligenceHub(str(tmp_path / "hub.db"))
    await hub.initialize()
    try:
        # Insert old and recent events
        await hub.event_store.insert_event("2025-01-01T10:00:00", "light.old", "light", "off", "on")
        await hub.event_store.insert_event("2026-02-20T10:00:00", "light.new", "light", "off", "on")

        # Run the prune method
        await hub._prune_event_store()

        # Old event should be pruned (default 90-day retention)
        remaining = await hub.event_store.query_events("2020-01-01", "2030-01-01")
        assert len(remaining) == 1
        assert remaining[0]["entity_id"] == "light.new"
    finally:
        await hub.shutdown()
