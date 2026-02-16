"""Shared test fixtures for ARIA engine test suite."""

import pytest

from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.config import (
    AppConfig,
    HAConfig,
    HolidayConfig,
    ModelConfig,
    OllamaConfig,
    PathConfig,
    SafetyConfig,
    WeatherConfig,
)
from aria.engine.features.time_features import build_time_features
from aria.engine.storage.data_store import DataStore

# --- Common test data ---

SAMPLE_STATES = [
    {
        "entity_id": "sensor.usp_pdu_pro_ac_power_consumption",
        "state": "156.5",
        "attributes": {"unit_of_measurement": "W", "friendly_name": "USP PDU Pro AC Power Consumption"},
    },
    {
        "entity_id": "person.justin",
        "state": "home",
        "attributes": {"friendly_name": "Justin", "source": "device_tracker.ipad_pro"},
    },
    {"entity_id": "person.lisa", "state": "not_home", "attributes": {"friendly_name": "Lisa"}},
    {
        "entity_id": "climate.bedroom",
        "state": "cool",
        "attributes": {"current_temperature": 72, "temperature": 68, "friendly_name": "Bedroom"},
    },
    {
        "entity_id": "lock.back_door",
        "state": "unlocked",
        "attributes": {"battery_level": 58, "friendly_name": "Back Door"},
    },
    {"entity_id": "light.atrium", "state": "on", "attributes": {"brightness": 143, "friendly_name": "Atrium"}},
    {"entity_id": "light.office", "state": "off", "attributes": {"friendly_name": "Office"}},
    {
        "entity_id": "automation.arrive_justin",
        "state": "on",
        "attributes": {"friendly_name": "Arrive Justin", "last_triggered": "2026-02-10T22:23:00"},
    },
    {
        "entity_id": "sensor.luda_battery",
        "state": "71",
        "attributes": {"unit_of_measurement": "%", "friendly_name": "TARS Battery"},
    },
    {
        "entity_id": "sensor.luda_charger_power",
        "state": "4.0",
        "attributes": {"unit_of_measurement": "kW", "friendly_name": "TARS Charger power"},
    },
    {
        "entity_id": "sensor.luda_range",
        "state": "199.3",
        "attributes": {"unit_of_measurement": "mi", "friendly_name": "TARS Range"},
    },
    {
        "entity_id": "binary_sensor.hue_motion_sensor_2_motion",
        "state": "off",
        "attributes": {"device_class": "motion", "friendly_name": "Closet motion Motion"},
    },
    {"entity_id": "device_tracker.iphonea17", "state": "home", "attributes": {"friendly_name": "iPhonea17"}},
]

EXTENDED_STATES = [
    {
        "entity_id": "binary_sensor.front_door",
        "state": "off",
        "attributes": {"device_class": "door", "friendly_name": "Front Door"},
    },
    {
        "entity_id": "binary_sensor.garage_door_sensor",
        "state": "on",
        "attributes": {"device_class": "garage_door", "friendly_name": "Garage Door"},
    },
    {
        "entity_id": "binary_sensor.kitchen_window",
        "state": "off",
        "attributes": {"device_class": "window", "friendly_name": "Kitchen Window"},
    },
    {
        "entity_id": "binary_sensor.motion_1",
        "state": "on",
        "attributes": {"device_class": "motion", "friendly_name": "Motion 1"},
    },
    {
        "entity_id": "lock.back_door",
        "state": "locked",
        "attributes": {"battery_level": 58, "friendly_name": "Back Door Lock"},
    },
    {
        "entity_id": "sensor.hue_motion_battery",
        "state": "82",
        "attributes": {"device_class": "battery", "unit_of_measurement": "%", "friendly_name": "Hue Motion Battery"},
    },
    {"entity_id": "device_tracker.iphone", "state": "home", "attributes": {}},
    {"entity_id": "device_tracker.ipad", "state": "home", "attributes": {}},
    {"entity_id": "device_tracker.macbook", "state": "not_home", "attributes": {}},
    {"entity_id": "device_tracker.unknown1", "state": "unavailable", "attributes": {}},
    {"entity_id": "media_player.living_room", "state": "playing", "attributes": {"friendly_name": "Living Room"}},
    {"entity_id": "media_player.atrium", "state": "idle", "attributes": {"friendly_name": "Atrium"}},
    {
        "entity_id": "sun.sun",
        "state": "above_horizon",
        "attributes": {
            "next_rising": "2026-02-11T06:42:00+00:00",
            "next_setting": "2026-02-10T17:58:00+00:00",
            "elevation": 32.5,
        },
    },
    {
        "entity_id": "vacuum.roborock",
        "state": "docked",
        "attributes": {"battery_level": 100, "friendly_name": "Roborock"},
    },
]


# --- Fixtures ---


@pytest.fixture
def holidays_config():
    return HolidayConfig()


@pytest.fixture
def empty_snapshot(holidays_config):
    return build_empty_snapshot("2026-02-10", holidays_config)


@pytest.fixture
def tmp_paths(tmp_path):
    """Create a PathConfig pointed at a temp directory."""
    paths = PathConfig(
        data_dir=tmp_path / "intelligence",
        logbook_path=tmp_path / "current.json",
    )
    paths.ensure_dirs()
    return paths


@pytest.fixture
def store(tmp_paths):
    """Create a DataStore backed by temp directories."""
    return DataStore(tmp_paths)


@pytest.fixture
def app_config(tmp_paths):
    """Create an AppConfig with temp paths."""
    return AppConfig(
        ha=HAConfig(),
        paths=tmp_paths,
        model=ModelConfig(),
        ollama=OllamaConfig(),
        weather=WeatherConfig(),
        safety=SafetyConfig(),
        holidays=HolidayConfig(),
    )


def make_snapshot(date_str, power=150, lights_on=30, devices_home=50):
    """Helper to build a snapshot with specific metric values."""
    snap = build_empty_snapshot(date_str, HolidayConfig())
    snap["power"]["total_watts"] = power
    snap["lights"]["on"] = lights_on
    snap["occupancy"]["device_count_home"] = devices_home
    snap["entities"]["unavailable"] = 900
    snap["logbook_summary"] = {"useful_events": 2500}
    return snap


def make_synthetic_snapshots(n=50):
    """Generate N synthetic snapshots with known patterns."""
    import random

    random.seed(42)
    snapshots = []
    for i in range(n):
        date_str = f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
        snap = build_empty_snapshot(date_str, HolidayConfig())
        temp = 50 + i * 0.8 + random.gauss(0, 3)
        snap["weather"] = {"temp_f": temp, "humidity_pct": 50 + random.gauss(0, 5), "wind_mph": 8}
        snap["power"]["total_watts"] = 100 + temp * 2 + random.gauss(0, 5)
        snap["lights"]["on"] = 5 + (3 if snap["is_weekend"] else 0) + random.randint(0, 2)
        snap["occupancy"]["people_home_count"] = 2
        snap["occupancy"]["device_count_home"] = 60 + random.randint(-5, 5)
        snap["motion"] = {"active_count": random.randint(0, 3)}
        snap["media"] = {"total_active": random.randint(0, 1)}
        snap["ev"] = {"TARS": {"battery_pct": 50 + random.randint(0, 50), "is_charging": False}}
        snap["entities"]["unavailable"] = 900
        snap["logbook_summary"] = {"useful_events": 2500}
        snap["time_features"] = build_time_features(f"{date_str}T16:00:00", None, date_str)
        snapshots.append(snap)
    return snapshots
