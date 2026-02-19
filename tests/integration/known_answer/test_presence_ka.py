"""Known-answer tests for PresenceModule.

Validates Frigate person-detection event handling, room signal accumulation,
presence cache structure, and golden snapshot stability against deterministic
fixture events.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.hub.constants import CACHE_PRESENCE
from aria.modules.presence import PresenceModule
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Deterministic fixture events (Frigate MQTT format)
# ---------------------------------------------------------------------------

CAMERA_ROOMS = {
    "front_door": "Hallway",
    "living_room": "Living Room",
    "kitchen": "Kitchen",
}

# Freeze time for deterministic Bayesian output
FROZEN_NOW = datetime(2026, 2, 19, 10, 0, 0)


def _freeze_datetime(monkeypatch, target_dt):
    """Patch datetime in presence module to return fixed time."""
    mock_dt = MagicMock(wraps=datetime)
    mock_dt.now.return_value = target_dt
    mock_dt.utcnow.return_value = target_dt
    mock_dt.fromisoformat = datetime.fromisoformat  # preserve real parser
    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
    monkeypatch.setattr("aria.modules.presence.datetime", mock_dt)
    return mock_dt


def _make_frigate_event(camera: str, score: float, event_id: str, **kwargs) -> dict:
    """Build a Frigate MQTT event payload (frigate/events topic).

    Optional kwargs: sub_label, sub_label_score, has_snapshot.
    """
    after = {
        "id": event_id,
        "camera": camera,
        "label": "person",
        "score": score,
        "sub_label": kwargs.get("sub_label"),
        "sub_label_score": kwargs.get("sub_label_score", 0.0),
        "has_snapshot": kwargs.get("has_snapshot", False),
    }
    return {"after": after}


FIXTURE_EVENTS = [
    # Person detected at front door
    _make_frigate_event("front_door", 0.85, "evt-001"),
    # Person detected in living room
    _make_frigate_event("living_room", 0.92, "evt-002"),
    # Face recognized in living room
    _make_frigate_event(
        "living_room",
        0.90,
        "evt-003",
        sub_label=["alice"],
        sub_label_score=0.95,
        has_snapshot=True,
    ),
    # Another person at front door
    _make_frigate_event("front_door", 0.78, "evt-004"),
    # Person in kitchen
    _make_frigate_event("kitchen", 0.88, "evt-005"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def presence(hub):
    """Create a PresenceModule without starting MQTT/WS listeners."""
    mod = PresenceModule(
        hub=hub,
        ha_url="http://test-host:8123",
        ha_token="test-token",
        mqtt_host="test-host",
        mqtt_port=1883,
        camera_rooms=CAMERA_ROOMS,
    )

    # Stub initialize — it starts MQTT, WS, timers we don't need
    mod.initialize = AsyncMock()

    # Stub hub.publish so event publishing doesn't error
    hub.publish = AsyncMock()

    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_processes_person_event(presence, hub, monkeypatch):
    """Feed Frigate person detection events, flush, verify presence cache updates."""
    # Feed events through the internal handler (bypassing MQTT transport)
    _freeze_datetime(monkeypatch, FROZEN_NOW)
    for event in FIXTURE_EVENTS:
        await presence._handle_frigate_event(event)

    # Verify room signals were recorded
    assert len(presence._room_signals) > 0, "Room signals should be populated"

    # Check signals exist for all rooms with cameras
    signaled_rooms = set(presence._room_signals.keys())
    assert "Hallway" in signaled_rooms, "Hallway should have signals (front_door camera)"
    assert "Living Room" in signaled_rooms, "Living Room should have signals"
    assert "Kitchen" in signaled_rooms, "Kitchen should have signals"

    # Verify recent detections ring buffer
    assert len(presence._recent_detections) == len(FIXTURE_EVENTS), (
        f"Expected {len(FIXTURE_EVENTS)} recent detections, got {len(presence._recent_detections)}"
    )

    # Verify face recognition tracked
    assert "alice" in presence._identified_persons, "alice should be identified"
    assert presence._identified_persons["alice"]["room"] == "Living Room"
    assert presence._identified_persons["alice"]["confidence"] == 0.95

    # Flush presence state to cache (same frozen time already active via monkeypatch)
    await presence._flush_presence_state()

    # Verify cache written
    cached = await hub.get_cache(CACHE_PRESENCE)
    assert cached is not None, "presence cache should exist after flush"

    data = cached["data"]
    assert "rooms" in data, "presence data should contain 'rooms'"
    assert "occupied_rooms" in data, "presence data should contain 'occupied_rooms'"
    assert "identified_persons" in data, "presence data should contain 'identified_persons'"
    assert "mqtt_connected" in data, "presence data should contain 'mqtt_connected'"
    assert "recent_detections" in data, "presence data should contain 'recent_detections'"

    # Occupied rooms should include rooms with high-confidence detections
    rooms_data = data["rooms"]
    assert "Living Room" in rooms_data, "Living Room should be in room results"
    assert rooms_data["Living Room"]["probability"] > 0.5, (
        "Living Room probability should be > 0.5 with person + face signals"
    )

    # alice should appear in identified_persons
    assert "alice" in data["identified_persons"]


@pytest.mark.asyncio
async def test_golden_snapshot(presence, hub, update_golden, monkeypatch):
    """Golden comparison of presence state after feeding multiple person events."""
    # Feed all fixture events with frozen time
    _freeze_datetime(monkeypatch, FROZEN_NOW)
    for event in FIXTURE_EVENTS:
        await presence._handle_frigate_event(event)

    # Flush to cache with same frozen time (monkeypatch still active)
    await presence._flush_presence_state()

    cached = await hub.get_cache(CACHE_PRESENCE)
    assert cached is not None

    data = cached["data"]

    # Extract stable fields (strip volatile timestamp)
    rooms_stable = {}
    for room_name, room_data in sorted(data["rooms"].items()):
        rooms_stable[room_name] = {
            "probability": room_data["probability"],
            "confidence": room_data["confidence"],
            "signal_count": len(room_data["signals"]),
            "signal_types": sorted({s["type"] for s in room_data["signals"]}),
            "person_count": len(room_data["persons"]),
        }

    identified_stable = {}
    for name, info in sorted(data["identified_persons"].items()):
        identified_stable[name] = {
            "room": info["room"],
            "confidence": info["confidence"],
        }

    detection_cameras = [d["camera"] for d in data["recent_detections"]]

    golden_data = {
        "rooms": rooms_stable,
        "occupied_rooms": sorted(data["occupied_rooms"]),
        "identified_persons": identified_stable,
        "mqtt_connected": data["mqtt_connected"],
        "camera_rooms": dict(sorted(data["camera_rooms"].items())),
        "face_recognition_enabled": data["face_recognition"]["enabled"],
        "recent_detection_count": len(data["recent_detections"]),
        "recent_detection_cameras": detection_cameras,
    }

    golden_compare(golden_data, "presence_state", update=update_golden)
