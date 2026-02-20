# tests/integration/test_event_flow.py
"""Integration test: state_changed -> EventStore -> API -> EntityGraph resolution."""

import pytest

from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore


@pytest.mark.asyncio
async def test_full_event_flow(tmp_path):
    """End-to-end: event insertion -> query -> area resolution."""
    # 1. Set up EventStore
    store = EventStore(str(tmp_path / "events.db"))
    await store.initialize()

    # 2. Set up EntityGraph
    graph = EntityGraph()
    graph.update(
        entities={
            "light.bedroom_lamp": {
                "entity_id": "light.bedroom_lamp",
                "device_id": "dev1",
                "area_id": None,
            },
            "binary_sensor.kitchen_motion": {
                "entity_id": "binary_sensor.kitchen_motion",
                "device_id": "dev2",
                "area_id": None,
            },
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
    for idx, (entity_id, domain, old, new) in enumerate(test_events):
        area_id = graph.get_area(entity_id)
        device = graph.get_device(entity_id)
        device_id = device.get("device_id") if device else None
        await store.insert_event(
            timestamp=f"2026-02-20T10:{idx:02d}:00",
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
