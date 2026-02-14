"""Integration tests for the organic discovery pipeline end-to-end.

Exercises the full OrganicDiscoveryModule with a real IntelligenceHub, real
SQLite cache, real HDBSCAN clustering, and realistic synthetic entity data.
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from aria.hub.core import IntelligenceHub
from aria.modules.organic_discovery.module import OrganicDiscoveryModule


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_entity(entity_id: str, domain: str, state: str = "on",
                 device_class: str | None = None,
                 device_id: str | None = None,
                 area_id: str | None = None,
                 unit_of_measurement: str | None = None,
                 attributes: dict | None = None) -> dict:
    """Build a single synthetic entity dict matching discovery cache format."""
    entity = {
        "entity_id": entity_id,
        "domain": domain,
        "state": state,
        "attributes": attributes or {},
    }
    if device_class is not None:
        entity["device_class"] = device_class
    if device_id is not None:
        entity["device_id"] = device_id
    if area_id is not None:
        entity["area_id"] = area_id
    if unit_of_measurement is not None:
        entity["unit_of_measurement"] = unit_of_measurement
    return entity


def build_synthetic_entities() -> list[dict]:
    """Build ~60 synthetic entities across 4 domains, clearly separated.

    Groups:
    - 18 lights (living room, bedroom, kitchen) — domain=light, no device_class
    - 15 switches — domain=switch, no device_class
    - 15 power sensors — domain=sensor, device_class=power, unit=W
    - 12 binary sensors — domain=binary_sensor, device_class=motion
    """
    entities = []

    # --- Lights (18) ---
    rooms = ["living_room", "bedroom", "kitchen"]
    for i, room in enumerate(rooms):
        for j in range(6):
            idx = i * 6 + j
            entities.append(_make_entity(
                entity_id=f"light.{room}_{j}",
                domain="light",
                state="on" if j % 2 == 0 else "off",
                device_id=f"device_light_{idx}",
                attributes={"brightness": 200 + j * 10, "color_temp": 350},
            ))

    # --- Switches (15) ---
    for i in range(15):
        entities.append(_make_entity(
            entity_id=f"switch.outlet_{i}",
            domain="switch",
            state="on" if i % 3 == 0 else "off",
            device_id=f"device_switch_{i}",
        ))

    # --- Power sensors (15) ---
    for i in range(15):
        entities.append(_make_entity(
            entity_id=f"sensor.power_{i}",
            domain="sensor",
            state=str(100 + i * 15),
            device_class="power",
            unit_of_measurement="W",
            device_id=f"device_sensor_{i}",
        ))

    # --- Binary sensors / motion (12) ---
    for i in range(12):
        entities.append(_make_entity(
            entity_id=f"binary_sensor.motion_{i}",
            domain="binary_sensor",
            state="on" if i % 4 == 0 else "off",
            device_class="motion",
            device_id=f"device_binary_{i}",
        ))

    return entities


def build_devices(entities: list[dict]) -> dict:
    """Build devices dict from entities, assigning areas by domain group."""
    area_map = {
        "light": "living_room",
        "switch": "garage",
        "sensor": "utility_room",
        "binary_sensor": "hallway",
    }
    devices = {}
    for entity in entities:
        device_id = entity.get("device_id")
        if device_id and device_id not in devices:
            domain = entity["domain"]
            devices[device_id] = {
                "id": device_id,
                "name": f"Device {device_id}",
                "area_id": area_map.get(domain, "unknown"),
                "manufacturer": f"mfr_{domain}",
            }
    return devices


def build_seed_capabilities(entities: list[dict]) -> dict:
    """Build seed capabilities that match light and power sensor entities."""
    light_ids = [e["entity_id"] for e in entities if e["domain"] == "light"]
    power_ids = [e["entity_id"] for e in entities if e["domain"] == "sensor"]
    return {
        "lighting": {
            "available": True,
            "entities": light_ids,
            "total_count": len(light_ids),
            "can_predict": True,
            "source": "seed",
            "usefulness": 100,
            "layer": "domain",
            "status": "promoted",
            "description": "All lighting entities",
        },
        "power_monitoring": {
            "available": True,
            "entities": power_ids,
            "total_count": len(power_ids),
            "can_predict": False,
            "source": "seed",
            "usefulness": 85,
            "layer": "domain",
            "status": "promoted",
            "description": "Power monitoring sensors",
        },
    }


def build_activity_rates(entities: list[dict]) -> dict:
    """Build activity_summary cache data with entity_activity rates."""
    entity_activity = {}
    rate_by_domain = {
        "light": 25.0,
        "switch": 10.0,
        "sensor": 40.0,
        "binary_sensor": 15.0,
    }
    for entity in entities:
        eid = entity["entity_id"]
        domain = entity["domain"]
        base_rate = rate_by_domain.get(domain, 5.0)
        entity_activity[eid] = {"daily_avg_changes": base_rate}
    return {"entity_activity": entity_activity}


def build_logbook_entries(entities: list[dict], n_windows: int = 30) -> list[dict]:
    """Build synthetic logbook entries with temporal co-occurrence.

    Creates n_windows time windows. In each window, a correlated group of
    entities fires together — lights and motion sensors fire in the same
    windows (simulating "someone enters a room").
    """
    entries = []
    base_time = datetime(2026, 2, 10, 8, 0, 0)

    light_ids = [e["entity_id"] for e in entities if e["domain"] == "light"][:8]
    motion_ids = [e["entity_id"] for e in entities if e["domain"] == "binary_sensor"][:8]
    correlated_group = light_ids + motion_ids

    switch_ids = [e["entity_id"] for e in entities if e["domain"] == "switch"][:8]

    for w in range(n_windows):
        window_time = base_time + timedelta(minutes=w * 15)

        # Correlated group fires together every window
        for eid in correlated_group:
            entries.append({
                "entity_id": eid,
                "state": "on",
                "when": (window_time + timedelta(seconds=len(entries) % 10)).isoformat(),
            })

        # Switches fire together but in alternating windows
        if w % 2 == 0:
            for eid in switch_ids:
                entries.append({
                    "entity_id": eid,
                    "state": "on",
                    "when": (window_time + timedelta(seconds=5)).isoformat(),
                })

    return entries


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db():
    """Create a temp directory for the SQLite DB, clean up after test."""
    tmp_dir = tempfile.mkdtemp(prefix="aria_test_")
    db_path = os.path.join(tmp_dir, "hub.db")
    yield db_path
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def synthetic_entities():
    return build_synthetic_entities()


@pytest.fixture()
def synthetic_devices(synthetic_entities):
    return build_devices(synthetic_entities)


@pytest.fixture()
def seed_capabilities(synthetic_entities):
    return build_seed_capabilities(synthetic_entities)


@pytest.fixture()
def activity_rates(synthetic_entities):
    return build_activity_rates(synthetic_entities)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_with_synthetic_entities(
    tmp_db, synthetic_entities, synthetic_devices, seed_capabilities, activity_rates,
):
    """End-to-end: populate cache, run discovery, verify outputs."""
    hub = IntelligenceHub(tmp_db)
    await hub.initialize()

    try:
        # Pre-populate cache
        await hub.set_cache("entities", synthetic_entities)
        await hub.set_cache("devices", synthetic_devices)
        await hub.set_cache("capabilities", seed_capabilities)
        await hub.set_cache("activity_summary", activity_rates)

        # Create module and run discovery (skip initialize to avoid schedule_task loop)
        module = OrganicDiscoveryModule(hub)
        # Patch _load_logbook to return empty (no logbook files in test env)
        module._load_logbook = AsyncMock(return_value=[])
        result = await module.run_discovery()

        # --- Assertions ---

        # 1. Result summary is populated
        assert result is not None
        assert "clusters_found" in result
        assert "organic_caps" in result
        assert "total_merged" in result

        # 2. Capabilities cache was written
        caps_entry = await hub.get_cache("capabilities")
        assert caps_entry is not None
        caps = caps_entry["data"]
        assert len(caps) > 0

        # 3. Seed capabilities are preserved with source="seed"
        assert "lighting" in caps
        assert caps["lighting"]["source"] == "seed"
        assert caps["lighting"]["status"] == "promoted"
        assert "power_monitoring" in caps
        assert caps["power_monitoring"]["source"] == "seed"

        # 4. At least one organic capability was discovered
        organic = {k: v for k, v in caps.items() if v.get("source") == "organic"}
        assert len(organic) > 0, "Expected at least one organic capability from 60 entities"

        # 5. Organic capabilities have required fields
        for name, cap in organic.items():
            assert "usefulness" in cap, f"Missing usefulness for {name}"
            assert isinstance(cap["usefulness"], int), f"Usefulness should be int for {name}"
            assert 0 <= cap["usefulness"] <= 100, f"Usefulness out of range for {name}"
            assert "usefulness_components" in cap
            assert cap["status"] == "candidate"
            assert cap["source"] == "organic"
            assert "entities" in cap
            assert len(cap["entities"]) > 0

        # 6. Discovery history was recorded
        history_entry = await hub.get_cache("discovery_history")
        assert history_entry is not None
        history = history_entry["data"]
        assert len(history) >= 1
        assert history[-1]["clusters_found"] >= 1

        # 7. Total merged = seeds + organic
        assert result["total_merged"] == len(caps)
        assert result["total_merged"] >= 3  # at least 2 seeds + 1 organic

    finally:
        await hub.shutdown()


@pytest.mark.asyncio
async def test_pipeline_with_behavioral_data(
    tmp_db, synthetic_entities, synthetic_devices, seed_capabilities, activity_rates,
):
    """Layer 2: behavioral clustering from logbook co-occurrence data."""
    hub = IntelligenceHub(tmp_db)
    await hub.initialize()

    try:
        await hub.set_cache("entities", synthetic_entities)
        await hub.set_cache("devices", synthetic_devices)
        await hub.set_cache("capabilities", seed_capabilities)
        await hub.set_cache("activity_summary", activity_rates)

        logbook_entries = build_logbook_entries(synthetic_entities, n_windows=40)

        module = OrganicDiscoveryModule(hub)
        module._load_logbook = AsyncMock(return_value=logbook_entries)
        result = await module.run_discovery()

        # Get capabilities
        caps_entry = await hub.get_cache("capabilities")
        caps = caps_entry["data"]

        # Find behavioral capabilities
        behavioral = {
            k: v for k, v in caps.items()
            if v.get("source") == "organic" and v.get("layer") == "behavioral"
        }

        # With 40 windows of correlated light+motion activity, HDBSCAN should
        # find at least one behavioral cluster
        assert len(behavioral) > 0, (
            f"Expected behavioral capabilities from {len(logbook_entries)} logbook entries. "
            f"All caps: {list(caps.keys())}"
        )

        # Behavioral capabilities should have temporal_pattern
        for name, cap in behavioral.items():
            assert "temporal_pattern" in cap, f"Missing temporal_pattern on {name}"

    finally:
        await hub.shutdown()


@pytest.mark.asyncio
async def test_autonomy_modes_integration(
    tmp_db, synthetic_entities, synthetic_devices, seed_capabilities, activity_rates,
):
    """Auto-promote mode: high-usefulness capabilities get promoted."""
    hub = IntelligenceHub(tmp_db)
    await hub.initialize()

    try:
        await hub.set_cache("entities", synthetic_entities)
        await hub.set_cache("devices", synthetic_devices)
        await hub.set_cache("capabilities", seed_capabilities)
        await hub.set_cache("activity_summary", activity_rates)

        module = OrganicDiscoveryModule(hub)
        module._load_logbook = AsyncMock(return_value=[])

        # Set auto_promote with very low threshold and 0-day streak so any
        # organic capability with nonzero usefulness gets promoted
        module.settings["autonomy_mode"] = "auto_promote"
        module.settings["promote_threshold"] = 1
        module.settings["promote_streak_days"] = 0

        # Seed history so stability_streak >= 0 (which matches promote_streak_days=0)
        result = await module.run_discovery()

        caps_entry = await hub.get_cache("capabilities")
        caps = caps_entry["data"]

        organic = {k: v for k, v in caps.items() if v.get("source") == "organic"}
        assert len(organic) > 0, "Need organic capabilities to test auto-promote"

        promoted = {k: v for k, v in organic.items() if v.get("status") == "promoted"}
        assert len(promoted) > 0, (
            f"Expected at least one promoted organic capability with threshold=1 and streak=0. "
            f"Organic caps: {[(k, v.get('usefulness'), v.get('stability_streak')) for k, v in organic.items()]}"
        )

        # Promoted capabilities should have promoted_at date
        for name, cap in promoted.items():
            assert cap["promoted_at"] is not None, f"promoted_at missing for {name}"

    finally:
        await hub.shutdown()


@pytest.mark.asyncio
async def test_seed_validation_integration(
    tmp_db, synthetic_entities, synthetic_devices, seed_capabilities, activity_rates,
):
    """Seed capabilities survive discovery and retain source='seed'."""
    hub = IntelligenceHub(tmp_db)
    await hub.initialize()

    try:
        await hub.set_cache("entities", synthetic_entities)
        await hub.set_cache("devices", synthetic_devices)
        await hub.set_cache("capabilities", seed_capabilities)
        await hub.set_cache("activity_summary", activity_rates)

        module = OrganicDiscoveryModule(hub)
        module._load_logbook = AsyncMock(return_value=[])
        result = await module.run_discovery()

        caps_entry = await hub.get_cache("capabilities")
        caps = caps_entry["data"]

        # Seeds must survive merge
        assert "lighting" in caps
        assert "power_monitoring" in caps

        # Source must remain "seed" (not overwritten by organic)
        assert caps["lighting"]["source"] == "seed"
        assert caps["power_monitoring"]["source"] == "seed"

        # Seeds always have status "promoted"
        assert caps["lighting"]["status"] == "promoted"
        assert caps["power_monitoring"]["status"] == "promoted"

        # Seeds preserve their entity lists
        assert len(caps["lighting"]["entities"]) == 18  # all lights
        assert len(caps["power_monitoring"]["entities"]) == 15  # all power sensors

        # Seed validation is in history record
        assert "seed_validation" in result

        # History was persisted
        history_entry = await hub.get_cache("discovery_history")
        assert history_entry is not None
        history = history_entry["data"]
        assert len(history) >= 1
        latest = history[-1]
        assert "seed_validation" in latest

    finally:
        await hub.shutdown()
