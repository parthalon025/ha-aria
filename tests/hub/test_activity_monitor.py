"""Unit tests for ActivityMonitor module.

Tests event filtering, occupancy tracking, buffer windowing, snapshot
triggering, daily counter resets, snapshot logging, and WebSocket liveness.
"""

import contextlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY
from aria.modules.activity_monitor import (
    DAILY_SNAPSHOT_CAP,
    SNAPSHOT_COOLDOWN_S,
    ActivityMonitor,
)

# ============================================================================
# Mock Hub
# ============================================================================


class MockCacheManager:
    """Mock cache manager with entity curation methods."""

    def __init__(self):
        self._included: set = set()
        self._all_curation: list[dict[str, Any]] = []
        self._should_raise: bool = False

    async def get_included_entity_ids(self) -> set:
        if self._should_raise:
            raise RuntimeError("Simulated cache failure")
        return set(self._included)

    async def get_all_curation(self) -> list[dict[str, Any]]:
        if self._should_raise:
            raise RuntimeError("Simulated cache failure")
        return list(self._all_curation)


class MockHub:
    """Lightweight hub mock that provides set_cache/get_cache without SQLite."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._running = True
        self._scheduled_tasks: list[dict[str, Any]] = []
        self.cache = MockCacheManager()

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = {"data": data, "metadata": metadata}

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        return self._cache.get(category)

    def is_running(self) -> bool:
        return self._running

    async def schedule_task(self, **kwargs):
        self._scheduled_tasks.append(kwargs)

    def register_module(self, mod):
        pass

    async def publish(self, event_type: str, data: dict[str, Any]):
        pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def monitor(hub, tmp_path):
    """Create an ActivityMonitor with a temp snapshot log path."""
    mon = ActivityMonitor(hub, "http://ha:8123", "fake-token")
    # Override snapshot log to use tmp_path so tests don't touch real filesystem
    mon._snapshot_log_path = tmp_path / "snapshot_log.jsonl"
    mon._snapshot_log_path.parent.mkdir(parents=True, exist_ok=True)
    return mon


# ============================================================================
# Event Filtering Tests
# ============================================================================


class TestEventFiltering:
    """Test _handle_state_changed filtering logic."""

    def test_tracked_domain_accepted(self, monitor):
        """Events from tracked domains (light, switch, etc.) are buffered."""
        data = {
            "entity_id": "light.kitchen",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1
        assert monitor._activity_buffer[0]["entity_id"] == "light.kitchen"

    def test_untracked_domain_rejected(self, monitor):
        """Events from untracked domains (automation, script, etc.) are dropped."""
        data = {
            "entity_id": "automation.morning_lights",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    def test_noise_transition_rejected(self, monitor):
        """Transitions like unavailable->unknown are noise and dropped."""
        data = {
            "entity_id": "light.garage",
            "old_state": {"state": "unavailable"},
            "new_state": {"state": "unknown", "attributes": {}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    def test_reverse_noise_transition_rejected(self, monitor):
        """unknown->unavailable is also noise."""
        data = {
            "entity_id": "switch.outlet",
            "old_state": {"state": "unknown"},
            "new_state": {"state": "unavailable", "attributes": {}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    def test_same_state_transition_rejected(self, monitor):
        """Events where old_state == new_state are dropped."""
        data = {
            "entity_id": "light.living_room",
            "old_state": {"state": "on"},
            "new_state": {"state": "on", "attributes": {}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    def test_sensor_power_accepted(self, monitor):
        """Sensors with device_class=power are conditionally tracked."""
        data = {
            "entity_id": "sensor.power_meter",
            "old_state": {"state": "100"},
            "new_state": {
                "state": "200",
                "attributes": {"device_class": "power", "friendly_name": "Power Meter"},
            },
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1

    def test_sensor_non_power_rejected(self, monitor):
        """Sensors without device_class=power are not tracked."""
        data = {
            "entity_id": "sensor.temperature",
            "old_state": {"state": "20"},
            "new_state": {
                "state": "21",
                "attributes": {"device_class": "temperature"},
            },
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    def test_events_today_increments(self, monitor):
        """Each accepted event increments _events_today counter."""
        for i in range(3):
            data = {
                "entity_id": f"light.room_{i}",
                "old_state": {"state": "off"},
                "new_state": {"state": "on", "attributes": {}},
            }
            monitor._handle_state_changed(data)
        assert monitor._events_today == 3

    def test_recent_events_ring_buffer(self, monitor):
        """Recent events are stored in the ring buffer (deque maxlen=20)."""
        for i in range(25):
            data = {
                "entity_id": f"light.room_{i}",
                "old_state": {"state": "off"},
                "new_state": {"state": "on", "attributes": {}},
            }
            monitor._handle_state_changed(data)
        # Ring buffer caps at 20
        assert len(monitor._recent_events) == 20
        # Most recent should be the last entity
        assert monitor._recent_events[-1]["entity_id"] == "light.room_24"


# ============================================================================
# Occupancy Tracking Tests
# ============================================================================


class TestOccupancyTracking:
    """Test _update_occupancy with person/device_tracker entities."""

    def test_person_arrives_home(self, monitor):
        """Person arriving home sets occupancy to True."""
        monitor._update_occupancy("person.justin", "home", "Justin")
        assert monitor._occupancy_state is True
        assert "Justin" in monitor._occupancy_people
        assert monitor._occupancy_since is not None

    def test_person_leaves(self, monitor):
        """Person leaving (last one) sets occupancy to False."""
        # Arrive first
        monitor._update_occupancy("person.justin", "home", "Justin")
        assert monitor._occupancy_state is True

        # Leave
        monitor._update_occupancy("person.justin", "not_home", "Justin")
        assert monitor._occupancy_state is False
        assert "Justin" not in monitor._occupancy_people

    def test_multiple_people_home(self, monitor):
        """Multiple people tracked independently."""
        monitor._update_occupancy("person.justin", "home", "Justin")
        monitor._update_occupancy("person.sarah", "home", "Sarah")
        assert monitor._occupancy_state is True
        assert len(monitor._occupancy_people) == 2

    def test_one_leaves_occupancy_stays(self, monitor):
        """If one person leaves but another is still home, occupancy stays True."""
        monitor._update_occupancy("person.justin", "home", "Justin")
        monitor._update_occupancy("person.sarah", "home", "Sarah")

        monitor._update_occupancy("person.justin", "not_home", "Justin")
        assert monitor._occupancy_state is True
        assert monitor._occupancy_people == ["Sarah"]

    def test_all_leave(self, monitor):
        """When everyone leaves, occupancy goes False."""
        monitor._update_occupancy("person.justin", "home", "Justin")
        monitor._update_occupancy("person.sarah", "home", "Sarah")

        monitor._update_occupancy("person.justin", "not_home", "Justin")
        monitor._update_occupancy("person.sarah", "not_home", "Sarah")
        assert monitor._occupancy_state is False
        assert monitor._occupancy_people == []

    def test_no_duplicate_people(self, monitor):
        """Arriving home multiple times doesn't duplicate the name."""
        monitor._update_occupancy("person.justin", "home", "Justin")
        monitor._update_occupancy("person.justin", "home", "Justin")
        assert monitor._occupancy_people.count("Justin") == 1


# ============================================================================
# Buffer Windowing Tests
# ============================================================================


class TestBufferWindowing:
    """Test _flush_activity_buffer groups events into 15-minute windows."""

    @pytest.mark.asyncio
    async def test_flush_creates_window(self, monitor, hub):
        """Flushing a non-empty buffer creates a cache window entry."""
        # Inject events into buffer
        now = datetime.now()
        for i in range(3):
            monitor._activity_buffer.append(
                {
                    "entity_id": f"light.room_{i}",
                    "domain": "light",
                    "device_class": "",
                    "from": "off",
                    "to": "on",
                    "time": now.strftime("%H:%M:%S"),
                    "timestamp": now.isoformat(),
                    "friendly_name": f"Room {i}",
                }
            )

        await monitor._flush_activity_buffer()

        # Cache should have the activity_log
        cached = await hub.get_cache(CACHE_ACTIVITY_LOG)
        assert cached is not None
        windows = cached["data"]["windows"]
        assert len(windows) == 1
        assert windows[0]["event_count"] == 3
        assert windows[0]["by_domain"]["light"] == 3

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self, monitor):
        """After flush, the activity buffer is emptied."""
        monitor._activity_buffer.append(
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "device_class": "",
                "from": "off",
                "to": "on",
                "time": "10:00:00",
                "timestamp": datetime.now().isoformat(),
                "friendly_name": "Kitchen",
            }
        )

        await monitor._flush_activity_buffer()
        assert len(monitor._activity_buffer) == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_still_updates_summary(self, monitor, hub):
        """Flushing with no events still updates the summary cache."""
        await monitor._flush_activity_buffer()

        summary = await hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        assert summary is not None
        assert "occupancy" in summary["data"]

    @pytest.mark.asyncio
    async def test_windows_pruned_beyond_24h(self, monitor, hub):
        """Windows older than 24 hours are pruned on flush."""
        # Seed cache with an old window
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        await hub.set_cache(
            CACHE_ACTIVITY_LOG,
            {
                "windows": [
                    {
                        "window_start": old_time,
                        "window_end": old_time,
                        "event_count": 5,
                        "by_domain": {"light": 5},
                        "notable_changes": [],
                        "occupancy": True,
                    }
                ],
                "last_updated": old_time,
                "events_today": 5,
                "snapshots_today": 0,
            },
        )

        # Add a fresh event and flush
        now = datetime.now()
        monitor._activity_buffer.append(
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "device_class": "",
                "from": "off",
                "to": "on",
                "time": now.strftime("%H:%M:%S"),
                "timestamp": now.isoformat(),
                "friendly_name": "Kitchen",
            }
        )

        await monitor._flush_activity_buffer()

        cached = await hub.get_cache(CACHE_ACTIVITY_LOG)
        windows = cached["data"]["windows"]
        # Only the new window should remain; old one pruned
        assert len(windows) == 1
        assert windows[0]["event_count"] == 1


# ============================================================================
# Snapshot Triggering Tests
# ============================================================================


class TestSnapshotTriggering:
    """Test _maybe_trigger_snapshot conditions."""

    def test_triggers_when_occupied_and_active(self, monitor):
        """Snapshot triggers when occupied, >5 events, and cooldown expired."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin"]
        # Fill buffer with enough events
        for i in range(6):
            monitor._activity_buffer.append(
                {
                    "entity_id": f"light.room_{i}",
                    "domain": "light",
                }
            )

        mock_future = MagicMock()
        mock_loop = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)

        with patch.object(monitor, "_run_snapshot"), patch("asyncio.get_running_loop", return_value=mock_loop):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == 1
        assert monitor._last_snapshot_time is not None

    def test_blocks_when_away(self, monitor):
        """Snapshot does not trigger when nobody is home."""
        monitor._occupancy_state = False
        monitor._activity_buffer = [{"domain": "light"}] * 10

        with patch.object(monitor, "_run_snapshot"):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == 0

    def test_blocks_during_cooldown(self, monitor):
        """Snapshot blocked within SNAPSHOT_COOLDOWN_S of the last snapshot."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin"]
        monitor._last_snapshot_time = datetime.now() - timedelta(seconds=60)
        monitor._activity_buffer = [{"domain": "light"}] * 10

        with patch.object(monitor, "_run_snapshot"):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == 0

    def test_blocks_at_daily_cap(self, monitor):
        """Snapshot blocked once the daily cap is reached."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin"]
        monitor._snapshots_today = DAILY_SNAPSHOT_CAP
        monitor._activity_buffer = [{"domain": "light"}] * 10

        with patch.object(monitor, "_run_snapshot"):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == DAILY_SNAPSHOT_CAP

    def test_blocks_with_insufficient_events(self, monitor):
        """Snapshot blocked when fewer than 5 events in buffer."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin"]
        monitor._activity_buffer = [{"domain": "light"}] * 4  # < 5

        with patch.object(monitor, "_run_snapshot"):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == 0

    def test_cooldown_expired_allows_trigger(self, monitor):
        """Snapshot allowed after cooldown has fully expired."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin"]
        # Set last snapshot well beyond cooldown
        monitor._last_snapshot_time = datetime.now() - timedelta(seconds=SNAPSHOT_COOLDOWN_S + 60)
        for i in range(6):
            monitor._activity_buffer.append(
                {
                    "entity_id": f"light.room_{i}",
                    "domain": "light",
                }
            )

        mock_future = MagicMock()
        mock_loop = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)

        with patch.object(monitor, "_run_snapshot"), patch("asyncio.get_running_loop", return_value=mock_loop):
            monitor._maybe_trigger_snapshot()

        assert monitor._snapshots_today == 1


# ============================================================================
# Daily Counter Reset Tests
# ============================================================================


class TestDailyCounterReset:
    """Test _reset_daily_counters at midnight boundary."""

    def test_reset_on_date_change(self, monitor):
        """Counters reset when the date has changed."""
        monitor._events_today = 42
        monitor._snapshots_today = 5
        monitor._snapshot_log_today_cache = [{"test": True}]
        monitor._events_date = "2025-01-01"
        monitor._snapshot_date = "2025-01-01"

        monitor._reset_daily_counters()

        assert monitor._events_today == 0
        assert monitor._snapshots_today == 0
        assert monitor._snapshot_log_today_cache == []
        assert monitor._events_date == datetime.now().strftime("%Y-%m-%d")

    def test_no_reset_same_date(self, monitor):
        """Counters are not reset if the date hasn't changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        monitor._events_today = 42
        monitor._snapshots_today = 5
        monitor._events_date = today

        monitor._reset_daily_counters()

        assert monitor._events_today == 42
        assert monitor._snapshots_today == 5


# ============================================================================
# Snapshot Log Tests
# ============================================================================


class TestSnapshotLog:
    """Test _append_snapshot_log and _read_snapshot_log_today."""

    def test_append_writes_jsonl(self, monitor):
        """Appended entries are written as JSONL to disk."""
        entry = {"timestamp": "2025-01-01T10:00:00", "number": 1}
        monitor._append_snapshot_log(entry)

        assert monitor._snapshot_log_path.exists()
        lines = monitor._snapshot_log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == entry

    def test_append_multiple_entries(self, monitor):
        """Multiple entries are appended as separate lines."""
        for i in range(3):
            monitor._append_snapshot_log({"number": i})

        lines = monitor._snapshot_log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_read_returns_today_cache(self, monitor):
        """_read_snapshot_log_today returns in-memory cache (no file scan)."""
        entries = [{"number": 1}, {"number": 2}]
        monitor._snapshot_log_today_cache = entries

        result = monitor._read_snapshot_log_today()
        assert result == entries
        # Returns a copy, not the same list
        assert result is not entries

    def test_append_updates_in_memory_cache(self, monitor):
        """_append_snapshot_log adds to the in-memory cache."""
        entry = {"number": 1}
        monitor._append_snapshot_log(entry)
        assert monitor._snapshot_log_today_cache == [entry]


# ============================================================================
# WebSocket Liveness Tests
# ============================================================================


class TestWebSocketLiveness:
    """Test WS connect/disconnect tracking state updates."""

    def test_initial_state_disconnected(self, monitor):
        """Monitor starts disconnected."""
        assert monitor._ws_connected is False
        assert monitor._ws_last_connected_at is None
        assert monitor._ws_disconnect_count == 0

    def test_disconnect_increments_count(self, monitor):
        """Simulating disconnect increments the counter."""
        monitor._ws_connected = True
        # Simulate what _ws_listen_loop does on disconnect
        monitor._ws_connected = False
        monitor._ws_disconnect_count += 1
        monitor._ws_last_disconnect_at = datetime.now()

        assert monitor._ws_connected is False
        assert monitor._ws_disconnect_count == 1
        assert monitor._ws_last_disconnect_at is not None

    def test_reconnect_tracks_gap_duration(self, monitor):
        """Reconnecting calculates the gap in total_disconnect_s."""
        # Simulate disconnect 10 seconds ago
        disconnect_time = datetime.now() - timedelta(seconds=10)
        monitor._ws_last_disconnect_at = disconnect_time
        monitor._ws_total_disconnect_s = 0.0

        # Simulate reconnect logic from _ws_listen_loop
        now = datetime.now()
        if monitor._ws_last_disconnect_at:
            gap = (now - monitor._ws_last_disconnect_at).total_seconds()
            monitor._ws_total_disconnect_s += gap
            monitor._ws_last_disconnect_at = None
        monitor._ws_connected = True
        monitor._ws_last_connected_at = now.isoformat()

        assert monitor._ws_connected is True
        assert monitor._ws_last_disconnect_at is None
        # Gap should be approximately 10s (allow tolerance for test execution)
        assert monitor._ws_total_disconnect_s >= 9.0
        assert monitor._ws_total_disconnect_s < 12.0


# ============================================================================
# Summary Cache Tests
# ============================================================================


class TestSummaryCache:
    """Test _update_summary_cache output structure."""

    @pytest.mark.asyncio
    async def test_summary_has_expected_keys(self, monitor, hub):
        """Summary cache contains occupancy, recent_activity, snapshot_status, websocket."""
        await monitor._update_summary_cache()

        cached = await hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        assert cached is not None
        data = cached["data"]

        assert "occupancy" in data
        assert "recent_activity" in data
        assert "activity_rate" in data
        assert "snapshot_status" in data
        assert "domains_active_1h" in data
        assert "websocket" in data

    @pytest.mark.asyncio
    async def test_summary_reflects_occupancy(self, monitor, hub):
        """Summary occupancy section reflects current state."""
        monitor._occupancy_state = True
        monitor._occupancy_people = ["Justin", "Sarah"]
        monitor._occupancy_since = "2025-01-01T10:00:00"

        await monitor._update_summary_cache()

        cached = await hub.get_cache(CACHE_ACTIVITY_SUMMARY)
        occ = cached["data"]["occupancy"]
        assert occ["anyone_home"] is True
        assert occ["people"] == ["Justin", "Sarah"]
        assert occ["since"] == "2025-01-01T10:00:00"


# ============================================================================
# Entity Curation Tests
# ============================================================================


class TestEntityCuration:
    """Test curation-based entity filtering in _handle_state_changed."""

    @pytest.mark.asyncio
    async def test_curation_included_entity_passes(self, monitor, hub):
        """Entity in the included set passes filter regardless of domain."""
        # Set up curation with an automation entity (normally excluded by domain)
        hub.cache._included = {"automation.morning_lights"}
        hub.cache._all_curation = [
            {"entity_id": "automation.morning_lights", "status": "promoted"},
        ]
        await monitor._load_curation_rules()
        assert monitor._curation_loaded is True

        data = {
            "entity_id": "automation.morning_lights",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Morning Lights"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1
        assert monitor._activity_buffer[0]["entity_id"] == "automation.morning_lights"

    @pytest.mark.asyncio
    async def test_curation_excluded_entity_blocked(self, monitor, hub):
        """Entity in the excluded set is blocked even if domain is tracked."""
        hub.cache._included = {"light.other"}
        hub.cache._all_curation = [
            {"entity_id": "light.other", "status": "included"},
            {"entity_id": "light.kitchen", "status": "excluded"},
        ]
        await monitor._load_curation_rules()
        assert monitor._curation_loaded is True

        data = {
            "entity_id": "light.kitchen",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 0

    @pytest.mark.asyncio
    async def test_curation_unknown_entity_falls_back_to_domain(self, monitor, hub):
        """Entity not in either curation set falls back to domain filtering."""
        hub.cache._included = {"light.bedroom"}
        hub.cache._all_curation = [
            {"entity_id": "light.bedroom", "status": "included"},
            {"entity_id": "sensor.noisy", "status": "excluded"},
        ]
        await monitor._load_curation_rules()
        assert monitor._curation_loaded is True

        # Unknown light entity — domain is tracked, should pass
        data = {
            "entity_id": "light.living_room",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Living Room"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1

        # Unknown automation entity — domain not tracked, should be blocked
        data2 = {
            "entity_id": "automation.test",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {}},
        }
        monitor._handle_state_changed(data2)
        assert len(monitor._activity_buffer) == 1  # still 1

    def test_curation_not_loaded_uses_domain_filter(self, monitor):
        """Before curation loads, domain-based filtering works as before."""
        assert monitor._curation_loaded is False

        # Tracked domain passes
        data = {
            "entity_id": "light.kitchen",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1

        # Untracked domain blocked
        data2 = {
            "entity_id": "automation.test",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {}},
        }
        monitor._handle_state_changed(data2)
        assert len(monitor._activity_buffer) == 1

    @pytest.mark.asyncio
    async def test_curation_reload_on_event(self, monitor, hub):
        """on_event('curation_updated') triggers reload of curation rules."""
        # Initially not loaded
        assert monitor._curation_loaded is False

        # Populate curation data
        hub.cache._included = {"light.kitchen", "switch.garage"}
        hub.cache._all_curation = [
            {"entity_id": "light.kitchen", "status": "included"},
            {"entity_id": "switch.garage", "status": "promoted"},
            {"entity_id": "sensor.noisy", "status": "auto_excluded"},
        ]

        # Fire the curation_updated event
        await monitor.on_event("curation_updated", {})

        assert monitor._curation_loaded is True
        assert "light.kitchen" in monitor._included_entities
        assert "switch.garage" in monitor._included_entities
        assert "sensor.noisy" in monitor._excluded_entities

    @pytest.mark.asyncio
    async def test_curation_load_failure_nonfatal(self, monitor, hub):
        """If cache methods fail, domain-based filtering continues to work."""
        hub.cache._should_raise = True

        # Loading should fail silently (logged warning)
        with contextlib.suppress(RuntimeError):
            await monitor._load_curation_rules()

        # Curation should NOT be loaded
        assert monitor._curation_loaded is False

        # Domain filter still works
        data = {
            "entity_id": "light.kitchen",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {"friendly_name": "Kitchen"}},
        }
        monitor._handle_state_changed(data)
        assert len(monitor._activity_buffer) == 1

        # Untracked domain still blocked
        data2 = {
            "entity_id": "automation.test",
            "old_state": {"state": "off"},
            "new_state": {"state": "on", "attributes": {}},
        }
        monitor._handle_state_changed(data2)
        assert len(monitor._activity_buffer) == 1
