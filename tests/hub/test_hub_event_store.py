# tests/hub/test_hub_event_store.py
"""Tests for EventStore integration with IntelligenceHub."""

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
