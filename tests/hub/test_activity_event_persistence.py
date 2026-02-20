# tests/hub/test_activity_event_persistence.py
"""Tests for activity_monitor writing events to EventStore."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore


@pytest.mark.asyncio
async def test_state_changed_persists_to_event_store(tmp_path):
    """When activity_monitor handles a state_changed, event appears in EventStore."""
    # Set up a mock hub with real EventStore and EntityGraph
    mock_hub = MagicMock()
    mock_hub.subscribe = MagicMock()
    mock_hub.publish = AsyncMock()
    mock_hub.cache = MagicMock()
    mock_hub.cache.get = AsyncMock(return_value=None)
    mock_hub.cache.get_config_value = AsyncMock(return_value=None)
    mock_hub.cache.get_included_entity_ids = AsyncMock(return_value=set())
    mock_hub.cache.get_all_curation = AsyncMock(return_value=[])
    mock_hub.is_running = MagicMock(return_value=True)
    mock_hub.schedule_task = AsyncMock()
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_cache = AsyncMock(return_value=None)

    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()
    mock_hub.event_store = store

    graph = EntityGraph()
    graph.update(
        {"light.bedroom": {"entity_id": "light.bedroom", "device_id": "dev1", "area_id": None}},
        {"dev1": {"device_id": "dev1", "area_id": "bedroom"}},
        [{"area_id": "bedroom", "name": "Bedroom"}],
    )
    mock_hub.entity_graph = graph

    # Create activity monitor
    from aria.modules.activity_monitor import ActivityMonitor

    monitor = ActivityMonitor(mock_hub, "http://test:8123", "test-token")

    # Simulate a state_changed event
    monitor._handle_state_changed(
        {
            "entity_id": "light.bedroom",
            "old_state": {"state": "off", "attributes": {"friendly_name": "Bedroom Light"}},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Bedroom Light", "brightness": 200}},
        }
    )

    # Allow async task to complete
    await asyncio.sleep(0.3)

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


@pytest.mark.asyncio
async def test_state_changed_includes_device_id(tmp_path):
    """EventStore event includes device_id from entity graph."""
    mock_hub = MagicMock()
    mock_hub.subscribe = MagicMock()
    mock_hub.publish = AsyncMock()
    mock_hub.cache = MagicMock()
    mock_hub.cache.get = AsyncMock(return_value=None)
    mock_hub.cache.get_config_value = AsyncMock(return_value=None)
    mock_hub.cache.get_included_entity_ids = AsyncMock(return_value=set())
    mock_hub.cache.get_all_curation = AsyncMock(return_value=[])
    mock_hub.is_running = MagicMock(return_value=True)
    mock_hub.schedule_task = AsyncMock()
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_cache = AsyncMock(return_value=None)

    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()
    mock_hub.event_store = store

    graph = EntityGraph()
    graph.update(
        {"switch.kitchen": {"entity_id": "switch.kitchen", "device_id": "dev42", "area_id": None}},
        {"dev42": {"device_id": "dev42", "area_id": "kitchen"}},
        [{"area_id": "kitchen", "name": "Kitchen"}],
    )
    mock_hub.entity_graph = graph

    from aria.modules.activity_monitor import ActivityMonitor

    monitor = ActivityMonitor(mock_hub, "http://test:8123", "test-token")

    monitor._handle_state_changed(
        {
            "entity_id": "switch.kitchen",
            "old_state": {"state": "on", "attributes": {"friendly_name": "Kitchen Switch"}},
            "new_state": {"state": "off", "attributes": {"friendly_name": "Kitchen Switch"}},
        }
    )

    await asyncio.sleep(0.3)

    events = await store.query_events("2020-01-01T00:00:00", "2030-01-01T00:00:00")
    assert len(events) >= 1
    assert events[0]["device_id"] == "dev42"
    assert events[0]["area_id"] == "kitchen"

    await store.close()


@pytest.mark.asyncio
async def test_state_changed_without_event_store(tmp_path):
    """Activity monitor works fine when hub has no event_store."""
    mock_hub = MagicMock()
    mock_hub.subscribe = MagicMock()
    mock_hub.publish = AsyncMock()
    mock_hub.cache = MagicMock()
    mock_hub.cache.get = AsyncMock(return_value=None)
    mock_hub.cache.get_config_value = AsyncMock(return_value=None)
    mock_hub.cache.get_included_entity_ids = AsyncMock(return_value=set())
    mock_hub.cache.get_all_curation = AsyncMock(return_value=[])
    mock_hub.is_running = MagicMock(return_value=True)
    mock_hub.schedule_task = AsyncMock()
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_cache = AsyncMock(return_value=None)

    # No event_store attribute
    del mock_hub.event_store

    from aria.modules.activity_monitor import ActivityMonitor

    monitor = ActivityMonitor(mock_hub, "http://test:8123", "test-token")

    # Should not raise
    monitor._handle_state_changed(
        {
            "entity_id": "light.bedroom",
            "old_state": {"state": "off", "attributes": {"friendly_name": "Bedroom Light"}},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Bedroom Light"}},
        }
    )

    await asyncio.sleep(0.1)
    # No assertion needed — just verifying no exception
