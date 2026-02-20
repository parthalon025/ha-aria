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


def test_get_area_orphan_entity(graph):
    """Entity with no device_id and no area_id returns None."""
    graph._entities["sensor.orphan"] = {"entity_id": "sensor.orphan", "device_id": None, "area_id": None}
    assert graph.get_area("sensor.orphan") is None


def test_get_device_unknown_device(graph):
    """Entity with device_id not in devices dict returns None."""
    graph._entities["sensor.mystery"] = {"entity_id": "sensor.mystery", "device_id": "dev_unknown", "area_id": None}
    assert graph.get_device("sensor.mystery") is None


def test_empty_graph():
    """Fresh EntityGraph before update() returns safe defaults."""
    g = EntityGraph()
    assert g.get_area("anything") is None
    assert g.get_device("anything") is None
    assert g.entities_in_area("anywhere") == []
    assert g.entities_by_domain("light") == []
    assert g.all_areas() == []
    assert g.entity_count == 0


def test_counter_properties(graph):
    """Counter properties reflect current graph state."""
    assert graph.entity_count == 3
    assert graph.device_count == 3
    assert graph.area_count == 2


def test_update_refreshes_data(graph, sample_entities, sample_devices):
    """Calling update() refreshes the graph."""
    new_entities = {
        **sample_entities,
        "switch.garage": {"entity_id": "switch.garage", "device_id": "dev_004", "area_id": None},
    }
    new_devices = {
        **sample_devices,
        "dev_004": {"device_id": "dev_004", "area_id": "garage", "name": "Garage Switch"},
    }
    new_areas = [
        {"area_id": "bedroom", "name": "Bedroom"},
        {"area_id": "kitchen", "name": "Kitchen"},
        {"area_id": "garage", "name": "Garage"},
    ]
    graph.update(new_entities, new_devices, new_areas)
    assert graph.get_area("switch.garage") == "garage"
