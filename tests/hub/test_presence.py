"""Unit tests for PresenceModule.

Tests signal management, Frigate event handling, HA state change processing,
room resolution, presence state flushing, and Bayesian fusion integration.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.engine.analysis.occupancy import SENSOR_CONFIG
from aria.hub.constants import CACHE_PRESENCE
from aria.modules.presence import (
    DEFAULT_CAMERA_ROOMS,
    SIGNAL_STALE_S,
    PresenceModule,
)

# ============================================================================
# Mock Hub
# ============================================================================


class MockCacheManager:
    """Mock cache manager for presence tests."""

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def get_cache(self, category: str):
        return self._cache.get(category)

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = data


class MockHub:
    """Lightweight hub mock for presence module tests."""

    def __init__(self):
        self._cache_data: dict[str, Any] = {}
        self._running = True
        self._scheduled_tasks: list[dict[str, Any]] = []
        self._published: list = []
        self.cache = MockCacheManager()

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache_data[category] = data
        self.cache._cache[category] = data

    async def get_cache(self, category: str):
        return self._cache_data.get(category) or self.cache._cache.get(category)

    def is_running(self) -> bool:
        return self._running

    async def schedule_task(self, task_id: str, coro, interval=None, run_immediately=False):
        self._scheduled_tasks.append(
            {
                "task_id": task_id,
                "interval": interval,
                "run_immediately": run_immediately,
            }
        )

    async def publish(self, event_type: str, data: Any):
        self._published.append((event_type, data))


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def module(hub):
    return PresenceModule(
        hub=hub,
        ha_url="http://192.168.1.35:8123",
        ha_token="test_token",
        mqtt_host="192.168.1.35",
        mqtt_port=1883,
        mqtt_user="frigate",
        mqtt_password="test_password",
    )


# ============================================================================
# Initialization
# ============================================================================


class TestInitialization:
    """Test module initialization and capability declaration."""

    def test_module_id(self, module):
        assert module.module_id == "presence"

    def test_default_camera_rooms(self, module):
        assert module.camera_rooms == DEFAULT_CAMERA_ROOMS

    def test_custom_camera_rooms(self, hub):
        custom = {"my_cam": "living_room"}
        m = PresenceModule(hub, "http://x", "tok", camera_rooms=custom)
        assert m.camera_rooms == custom

    async def test_initialize_schedules_tasks(self, module):
        await module.initialize()
        task_ids = [t["task_id"] for t in module.hub._scheduled_tasks]
        assert "presence_mqtt_listener" in task_ids
        assert "presence_ws_listener" in task_ids
        assert "presence_state_flush" in task_ids

    def test_capabilities_declared(self):
        caps = PresenceModule.CAPABILITIES
        assert len(caps) == 1
        assert caps[0].id == "presence_tracking"
        assert caps[0].layer == "hub"
        assert caps[0].module == "presence"


# ============================================================================
# Signal Management
# ============================================================================


class TestSignalManagement:
    """Test adding and retrieving signals."""

    def test_add_signal(self, module):
        now = datetime.now()
        module._add_signal("living_room", "motion", 0.9, "test", now)
        assert len(module._room_signals["living_room"]) == 1

    def test_add_multiple_signals(self, module):
        now = datetime.now()
        module._add_signal("living_room", "motion", 0.9, "motion1", now)
        module._add_signal("living_room", "camera_person", 0.95, "cam1", now)
        module._add_signal("kitchen", "door", 0.7, "door1", now)
        assert len(module._room_signals["living_room"]) == 2
        assert len(module._room_signals["kitchen"]) == 1

    def test_get_active_signals_fresh(self, module):
        now = datetime.now()
        module._add_signal("room", "motion", 0.9, "fresh", now)
        active = module._get_active_signals("room", now)
        assert len(active) == 1
        assert active[0][0] == "motion"

    def test_get_active_signals_stale(self, module):
        stale_time = datetime.now() - timedelta(seconds=SIGNAL_STALE_S + 60)
        module._add_signal("room", "motion", 0.9, "stale", stale_time)
        active = module._get_active_signals("room", datetime.now())
        assert len(active) == 0

    def test_get_active_signals_decay(self, module):
        # Signal halfway through its decay period should be reduced
        half_decay = SENSOR_CONFIG.get("motion", {}).get("decay_seconds", 300) / 2
        past = datetime.now() - timedelta(seconds=half_decay)
        module._add_signal("room", "motion", 0.9, "decaying", past)
        active = module._get_active_signals("room", datetime.now())
        if active:
            # Value should be reduced but still positive
            assert active[0][1] < 0.9
            assert active[0][1] > 0.0

    def test_get_active_signals_camera_face_no_decay(self, module):
        # camera_face has decay_seconds: 0 — should not decay
        now = datetime.now()
        module._add_signal("room", "camera_face", 0.99, "face", now)
        active = module._get_active_signals("room", now)
        assert len(active) == 1
        assert active[0][1] == 0.99

    def test_get_active_signals_empty_room(self, module):
        active = module._get_active_signals("nonexistent_room", datetime.now())
        assert active == []


# ============================================================================
# Frigate Event Handling
# ============================================================================


class TestFrigateEvents:
    """Test MQTT event processing from Frigate."""

    async def test_handle_person_detection(self, module):
        event = {
            "after": {
                "camera": "driveway",
                "label": "person",
                "score": 0.87,
            }
        }
        await module._handle_frigate_event(event)
        signals = module._room_signals.get("driveway", [])
        assert len(signals) == 1
        assert signals[0][0] == "camera_person"
        assert signals[0][1] <= 0.99  # Capped at 0.99

    async def test_handle_face_recognition(self, module):
        event = {
            "after": {
                "camera": "carters_room",
                "label": "person",
                "score": 0.9,
                "sub_label": ["Carter"],
                "sub_label_score": 0.92,
            }
        }
        await module._handle_frigate_event(event)
        signals = module._room_signals.get("carters_room", [])
        # Should have both person + face signals
        types = [s[0] for s in signals]
        assert "camera_person" in types
        assert "camera_face" in types
        # Should track identified person
        assert "Carter" in module._identified_persons
        assert module._identified_persons["Carter"]["room"] == "carters_room"

    async def test_handle_event_no_after(self, module):
        await module._handle_frigate_event({})
        assert len(module._room_signals) == 0

    async def test_handle_event_non_person(self, module):
        event = {
            "after": {
                "camera": "driveway",
                "label": "car",
                "score": 0.8,
            }
        }
        await module._handle_frigate_event(event)
        assert len(module._room_signals) == 0

    async def test_handle_person_count(self, module):
        await module._handle_person_count("backyard", 2)
        signals = module._room_signals.get("backyard", [])
        assert len(signals) == 1
        assert "2 person(s)" in signals[0][2]

    async def test_handle_person_count_zero(self, module):
        await module._handle_person_count("backyard", 0)
        assert len(module._room_signals.get("backyard", [])) == 0

    async def test_handle_person_count_invalid(self, module):
        await module._handle_person_count("backyard", "invalid")
        assert len(module._room_signals.get("backyard", [])) == 0

    async def test_camera_room_mapping(self, module):
        event = {
            "after": {
                "camera": "panoramic",
                "label": "person",
                "score": 0.9,
            }
        }
        await module._handle_frigate_event(event)
        # panoramic maps to "backyard" in DEFAULT_CAMERA_ROOMS
        assert "backyard" in module._room_signals

    async def test_unknown_camera_uses_camera_name(self, module):
        event = {
            "after": {
                "camera": "unknown_cam",
                "label": "person",
                "score": 0.9,
            }
        }
        await module._handle_frigate_event(event)
        assert "unknown_cam" in module._room_signals

    async def test_mqtt_message_routing_events(self, module):
        payload = {
            "after": {
                "camera": "driveway",
                "label": "person",
                "score": 0.85,
            }
        }
        await module._handle_mqtt_message("frigate/events", payload)
        assert "driveway" in module._room_signals

    async def test_mqtt_message_routing_person_count(self, module):
        await module._handle_mqtt_message("frigate/driveway/person", 1)
        assert "driveway" in module._room_signals


# ============================================================================
# HA State Change Handling
# ============================================================================


class TestHAStateChanges:
    """Test processing of Home Assistant state change events."""

    async def test_motion_sensor_on(self, module):
        module._resolve_room = AsyncMock(return_value="living_room")
        data = {
            "new_state": {
                "entity_id": "binary_sensor.living_room_motion",
                "state": "on",
                "attributes": {"device_class": "motion"},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("living_room", [])
        assert len(signals) == 1
        assert signals[0][0] == "motion"

    async def test_motion_sensor_off_ignored(self, module):
        module._resolve_room = AsyncMock(return_value="living_room")
        data = {
            "new_state": {
                "entity_id": "binary_sensor.living_room_motion",
                "state": "off",
                "attributes": {"device_class": "motion"},
            }
        }
        await module._handle_ha_state_change(data)
        assert len(module._room_signals.get("living_room", [])) == 0

    async def test_light_turned_on(self, module):
        module._resolve_room = AsyncMock(return_value="kitchen")
        data = {
            "new_state": {
                "entity_id": "light.kitchen",
                "state": "on",
                "attributes": {},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("kitchen", [])
        assert len(signals) == 1
        assert signals[0][0] == "light_interaction"

    async def test_dimmer_press(self, module):
        module._resolve_room = AsyncMock(return_value="bedroom")
        data = {
            "new_state": {
                "entity_id": "event.hue_dimmer_bedroom",
                "state": "",
                "attributes": {"event_type": "initial_press"},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("bedroom", [])
        assert len(signals) == 1
        assert signals[0][0] == "dimmer_press"

    async def test_person_home(self, module):
        module._resolve_room = AsyncMock(return_value="overall")
        data = {
            "new_state": {
                "entity_id": "person.justin",
                "state": "home",
                "attributes": {},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("overall", [])
        assert len(signals) == 1
        assert signals[0][0] == "device_tracker"
        assert signals[0][1] == 0.9  # home = high

    async def test_person_away(self, module):
        module._resolve_room = AsyncMock(return_value="overall")
        data = {
            "new_state": {
                "entity_id": "person.justin",
                "state": "not_home",
                "attributes": {},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("overall", [])
        assert len(signals) == 1
        assert signals[0][1] == 0.1  # away = low

    async def test_door_sensor(self, module):
        module._resolve_room = AsyncMock(return_value="garage")
        data = {
            "new_state": {
                "entity_id": "binary_sensor.garage_door",
                "state": "on",
                "attributes": {"device_class": "door"},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("garage", [])
        assert len(signals) == 1
        assert signals[0][0] == "door"

    async def test_no_new_state_ignored(self, module):
        await module._handle_ha_state_change({"old_state": {}})
        assert len(module._room_signals) == 0

    async def test_unresolved_room_ignored(self, module):
        module._resolve_room = AsyncMock(return_value=None)
        data = {
            "new_state": {
                "entity_id": "binary_sensor.unknown",
                "state": "on",
                "attributes": {"device_class": "motion"},
            }
        }
        await module._handle_ha_state_change(data)
        assert len(module._room_signals) == 0


# ============================================================================
# Room Resolution
# ============================================================================


class TestRoomResolution:
    """Test entity-to-room resolution logic."""

    async def test_resolve_from_entity_cache(self, module):
        module.hub._cache_data["entities"] = {
            "binary_sensor.living_room_motion": {
                "area_id": "living_room",
            }
        }
        room = await module._resolve_room("binary_sensor.living_room_motion", {})
        assert room == "living_room"

    async def test_resolve_from_device_cache(self, module):
        module.hub._cache_data["entities"] = {
            "binary_sensor.sensor1": {
                "device_id": "dev123",
            }
        }
        module.hub._cache_data["devices"] = {
            "dev123": {"area_id": "kitchen"},
        }
        room = await module._resolve_room("binary_sensor.sensor1", {})
        assert room == "kitchen"

    async def test_resolve_fallback_bedroom(self, module):
        room = await module._resolve_room("light.master_bedroom_lamp", {})
        assert room == "bedroom"

    async def test_resolve_fallback_front_door(self, module):
        room = await module._resolve_room("binary_sensor.front_door_motion", {})
        assert room == "front_door"

    async def test_resolve_fallback_doorbell(self, module):
        room = await module._resolve_room("camera.doorbell", {})
        assert room == "front_door"

    async def test_resolve_unknown_returns_none(self, module):
        room = await module._resolve_room("sensor.unknown_thing", {})
        assert room is None

    async def test_resolve_room_unwraps_cache_data(self, module):
        """_resolve_room should look inside the 'data' wrapper from hub cache."""
        module.hub.get_cache = AsyncMock(
            return_value={
                "category": "entities",
                "data": {
                    "binary_sensor.closet_motion": {
                        "area_id": "closet",
                        "device_id": "dev123",
                    }
                },
                "version": 1,
            }
        )
        room = await module._resolve_room("binary_sensor.closet_motion", {})
        assert room == "closet"

    async def test_resolve_room_device_fallback_unwraps_cache(self, module):
        """Device fallback in _resolve_room should also unwrap cache data."""

        async def mock_get_cache(key):
            if key == "entities":
                return {
                    "category": "entities",
                    "data": {
                        "light.kitchen_lamp": {
                            "device_id": "dev456",
                        }
                    },
                }
            elif key == "devices":
                return {
                    "category": "devices",
                    "data": {"dev456": {"area_id": "kitchen"}},
                }
            return None

        module.hub.get_cache = AsyncMock(side_effect=mock_get_cache)
        room = await module._resolve_room("light.kitchen_lamp", {})
        assert room == "kitchen"

    async def test_person_home_bypasses_room_resolution(self, module):
        """Person entities should NOT call _resolve_room."""
        module._resolve_room = AsyncMock(return_value=None)  # Would block if called
        data = {
            "new_state": {
                "entity_id": "person.justin",
                "state": "home",
                "attributes": {},
            }
        }
        await module._handle_ha_state_change(data)
        signals = module._room_signals.get("overall", [])
        assert len(signals) == 1
        assert signals[0][0] == "device_tracker"
        assert signals[0][1] == 0.9
        # _resolve_room should NOT have been called
        module._resolve_room.assert_not_called()


# ============================================================================
# Presence State Flush
# ============================================================================


class TestFlushPresenceState:
    """Test the periodic presence state calculation and caching."""

    async def test_flush_empty(self, module):
        await module._flush_presence_state()
        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert cached is not None
        assert cached["rooms"] == {}
        assert cached["occupied_rooms"] == []

    async def test_flush_with_signals(self, module):
        now = datetime.now()
        module._add_signal("living_room", "motion", 0.9, "motion detected", now)
        module._add_signal("living_room", "camera_person", 0.95, "person seen", now)
        await module._flush_presence_state()

        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "living_room" in cached["rooms"]
        room_data = cached["rooms"]["living_room"]
        assert room_data["probability"] > 0.5
        assert room_data["confidence"] != "none"
        assert len(room_data["signals"]) == 2

    async def test_flush_publishes_event(self, module):
        now = datetime.now()
        module._add_signal("room", "motion", 0.9, "test", now)
        await module._flush_presence_state()
        assert len(module.hub._published) == 1
        assert module.hub._published[0][0] == "presence_updated"

    async def test_flush_prunes_stale_signals(self, module):
        stale = datetime.now() - timedelta(seconds=SIGNAL_STALE_S + 60)
        module._add_signal("room", "motion", 0.9, "stale", stale)
        now = datetime.now()
        module._add_signal("room", "motion", 0.8, "fresh", now)
        await module._flush_presence_state()
        # After flush, stale signal should be pruned
        assert len(module._room_signals["room"]) == 1

    async def test_flush_tracks_identified_persons(self, module):
        now = datetime.now()
        module._identified_persons["Carter"] = {
            "room": "carters_room",
            "last_seen": now.isoformat(),
            "confidence": 0.92,
            "camera": "carters_room",
        }
        module._add_signal("carters_room", "camera_face", 0.92, "Carter", now)
        await module._flush_presence_state()

        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "Carter" in cached["identified_persons"]
        assert cached["identified_persons"]["Carter"]["room"] == "carters_room"

    async def test_flush_occupied_rooms_threshold(self, module):
        now = datetime.now()
        # High-confidence signal
        module._add_signal("living_room", "camera_person", 0.95, "person", now)
        module._add_signal("living_room", "motion", 0.9, "motion", now)
        await module._flush_presence_state()

        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "living_room" in cached["occupied_rooms"]

    async def test_flush_includes_mqtt_status(self, module):
        await module._flush_presence_state()
        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "mqtt_connected" in cached
        assert cached["mqtt_connected"] is False  # Not connected in tests

    async def test_flush_includes_camera_rooms(self, module):
        await module._flush_presence_state()
        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "camera_rooms" in cached
        assert cached["camera_rooms"] == DEFAULT_CAMERA_ROOMS


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    async def test_face_recognition_string_sublabel(self, module):
        """Handle sub_label as string instead of list."""
        event = {
            "after": {
                "camera": "driveway",
                "label": "person",
                "score": 0.9,
                "sub_label": ["Justin"],
                "sub_label_score": 0.88,
            }
        }
        await module._handle_frigate_event(event)
        assert "Justin" in module._identified_persons

    async def test_score_capped_at_099(self, module):
        event = {
            "after": {
                "camera": "driveway",
                "label": "person",
                "score": 1.0,
            }
        }
        await module._handle_frigate_event(event)
        signal = module._room_signals["driveway"][0]
        assert signal[1] <= 0.99

    async def test_resolve_room_cache_exception(self, module):
        """Cache errors should not propagate."""
        module.hub.get_cache = AsyncMock(side_effect=RuntimeError("broken"))
        # Should fall through to fallback parsing
        room = await module._resolve_room("light.bedroom_lamp", {})
        assert room == "bedroom"

    async def test_stale_identified_persons_excluded(self, module):
        stale = datetime.now() - timedelta(seconds=SIGNAL_STALE_S + 60)
        module._identified_persons["OldPerson"] = {
            "room": "kitchen",
            "last_seen": stale.isoformat(),
            "confidence": 0.9,
            "camera": "kitchen_cam",
        }
        await module._flush_presence_state()
        cached = module.hub.cache._cache.get(CACHE_PRESENCE)
        assert "OldPerson" not in cached["identified_persons"]

    def test_default_camera_rooms_coverage(self):
        """All expected cameras should be mapped."""
        expected = {"driveway", "backyard", "panoramic", "front_doorbell", "carters_room", "collins_room", "pool"}
        assert set(DEFAULT_CAMERA_ROOMS.keys()) == expected
