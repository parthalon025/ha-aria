"""Known-answer tests for ActivityMonitor.

Validates event buffering, activity log flushing, entity counting in summaries,
and golden snapshot stability against deterministic fixture events.
"""

from unittest.mock import AsyncMock

import pytest

from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY
from aria.modules.activity_monitor import ActivityMonitor
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Deterministic fixture events
# ---------------------------------------------------------------------------


def _make_event(entity_id: str, old_state: str, new_state: str, timestamp: str) -> dict:
    """Build a state_changed event dict matching HA WebSocket format.

    ActivityMonitor._handle_state_changed expects:
      data["entity_id"], data["new_state"]["state"], data["old_state"]["state"],
      data["new_state"]["attributes"]
    """
    return {
        "entity_id": entity_id,
        "old_state": {"state": old_state, "attributes": {}},
        "new_state": {"state": new_state, "attributes": {}},
    }


FIXTURE_EVENTS = [
    _make_event("light.living_room", "off", "on", "2026-02-19T08:00:00"),
    _make_event("sensor.temperature", "21.0", "22.5", "2026-02-19T08:01:00"),
    _make_event("binary_sensor.motion", "off", "on", "2026-02-19T08:02:00"),
    _make_event("light.kitchen", "off", "on", "2026-02-19T08:05:00"),
    _make_event("light.living_room", "on", "off", "2026-02-19T08:30:00"),
    _make_event("binary_sensor.motion", "on", "off", "2026-02-19T08:35:00"),
    _make_event("sensor.temperature", "22.5", "23.0", "2026-02-19T09:00:00"),
    _make_event("switch.smart_plug", "off", "on", "2026-02-19T09:15:00"),
]

# sensor.temperature is in CONDITIONAL_DOMAINS and only tracked when
# device_class == "power".  Our fixtures have no device_class, so sensor
# events are filtered.  Expected tracked entities:
EXPECTED_TRACKED_ENTITIES = {
    "light.living_room",
    "binary_sensor.motion",
    "light.kitchen",
    "switch.smart_plug",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def monitor(hub):
    """Create an ActivityMonitor without starting WS or timers."""
    mon = ActivityMonitor(hub=hub, ha_url="http://test-host:8123", ha_token="test-token")

    # Stub out initialize — it schedules WS listener and timers we don't need
    mon.initialize = AsyncMock()

    # Stub hub.publish so fire-and-forget event publishing doesn't error
    hub.publish = AsyncMock()

    # Prevent snapshot subprocess from running
    mon._maybe_trigger_snapshot = lambda: None

    return mon


def _feed_events(monitor: ActivityMonitor, events: list[dict]) -> None:
    """Feed a list of fixture events into the monitor's handler."""
    for evt in events:
        monitor._handle_state_changed(evt)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_produce_activity_log(monitor, hub):
    """Feed state_changed events, flush buffer, verify activity_log cache has data."""
    _feed_events(monitor, FIXTURE_EVENTS)

    # Buffer should have events (minus filtered sensor.temperature)
    assert len(monitor._activity_buffer) > 0, "Buffer should contain events after feeding"

    # Flush buffer into cache
    await monitor._flush_activity_buffer()

    # Read activity_log from cache
    cached = await hub.get_cache(CACHE_ACTIVITY_LOG)
    assert cached is not None, "activity_log cache should exist after flush"

    data = cached["data"]
    assert "windows" in data, "activity_log should contain 'windows' key"
    assert len(data["windows"]) == 1, "Single flush should produce one window"

    window = data["windows"][0]
    assert window["event_count"] == 6, (
        f"Expected 6 tracked events (8 total minus 2 sensor.temperature), got {window['event_count']}"
    )
    assert "by_domain" in window
    assert "by_entity" in window


@pytest.mark.asyncio
async def test_entity_count_in_summary(monitor, hub):
    """After events, verify activity_summary reflects distinct tracked entities."""
    _feed_events(monitor, FIXTURE_EVENTS)
    await monitor._flush_activity_buffer()

    # Summary cache is updated at end of flush
    cached = await hub.get_cache(CACHE_ACTIVITY_SUMMARY)
    assert cached is not None, "activity_summary cache should exist after flush"

    summary = cached["data"]

    # Check recent_activity contains only tracked entities
    recent_entities = {item["entity"] for item in summary.get("recent_activity", [])}
    assert recent_entities == EXPECTED_TRACKED_ENTITIES, (
        f"Expected entities {EXPECTED_TRACKED_ENTITIES}, got {recent_entities}"
    )

    # Verify occupancy structure exists
    assert "occupancy" in summary
    assert "anyone_home" in summary["occupancy"]


@pytest.mark.asyncio
async def test_golden_snapshot(monitor, hub, update_golden):
    """Golden comparison of activity log and summary output."""
    _feed_events(monitor, FIXTURE_EVENTS)
    await monitor._flush_activity_buffer()

    # --- Activity log golden ---
    log_cached = await hub.get_cache(CACHE_ACTIVITY_LOG)
    assert log_cached is not None

    log_data = log_cached["data"]
    # Strip volatile fields for deterministic comparison
    log_stable = {
        "window_count": len(log_data["windows"]),
        "events_today": log_data.get("events_today", 0),
        "window_event_count": log_data["windows"][0]["event_count"],
        "window_by_domain": log_data["windows"][0]["by_domain"],
        "window_by_entity": log_data["windows"][0]["by_entity"],
        "window_occupancy": log_data["windows"][0]["occupancy"],
    }

    golden_compare(log_stable, "activity_monitor_log", update=update_golden)

    # --- Activity summary golden ---
    summary_cached = await hub.get_cache(CACHE_ACTIVITY_SUMMARY)
    assert summary_cached is not None

    summary = summary_cached["data"]
    # Extract stable fields only (timestamps, trend, predictions are volatile)
    recent_entities = sorted({item["entity"] for item in summary.get("recent_activity", [])})
    summary_stable = {
        "occupancy_anyone_home": summary["occupancy"]["anyone_home"],
        "occupancy_people": summary["occupancy"]["people"],
        "recent_entity_ids": recent_entities,
        "recent_activity_count": len(summary.get("recent_activity", [])),
        "snapshot_today_count": summary["snapshot_status"]["today_count"],
        "websocket_connected": summary["websocket"]["connected"],
    }

    golden_compare(summary_stable, "activity_monitor_summary", update=update_golden)
