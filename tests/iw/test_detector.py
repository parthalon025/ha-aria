"""Tests for the IW real-time detector module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.iw.models import (
    BehavioralStateDefinition,
    BehavioralStateTracker,
    Indicator,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_definition(  # noqa: PLR0913
    def_id: str = "def-1",
    trigger_entity: str = "light.kitchen",
    trigger_state: str = "on",
    confirming: list[tuple[str, str]] | None = None,
    person: str | None = None,
    duration_minutes: float = 10.0,
) -> BehavioralStateDefinition:
    """Build a minimal BehavioralStateDefinition for testing."""
    confirming = confirming or [("binary_sensor.motion_kitchen", "on")]
    trigger = Indicator(
        entity_id=trigger_entity,
        role="trigger",
        mode="state_change",
        expected_state=trigger_state,
        confidence=0.8,
    )
    conf_indicators = [
        Indicator(
            entity_id=eid,
            role="confirming",
            mode="state_change",
            expected_state=state,
            max_delay_seconds=300,
            confidence=0.7,
        )
        for eid, state in confirming
    ]
    return BehavioralStateDefinition(
        id=def_id,
        name=f"Test {def_id}",
        trigger=trigger,
        trigger_preconditions=[],
        confirming=conf_indicators,
        deviations=[],
        areas=frozenset(["kitchen"]),
        day_types=frozenset(["weekday"]),
        person_attribution=person,
        typical_duration_minutes=duration_minutes,
        expected_outcomes=(),
    )


def _make_hub() -> MagicMock:
    """Create a mock IntelligenceHub with subscribe/unsubscribe/cache/event_store."""
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.unsubscribe = MagicMock()
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(side_effect=_config_defaults)
    hub.event_store = MagicMock()
    hub.event_store.query_events = AsyncMock(return_value=[])
    return hub


def _config_defaults(key: str, fallback: Any = None) -> Any:
    """Return config defaults for test purposes."""
    defaults = {
        "iw.expiry_check_interval_seconds": 10,
        "iw.min_match_ratio": 0.5,
        "iw.cold_start_replay_minutes": 15,
    }
    return defaults.get(key, fallback)


def _make_store(definitions: list[BehavioralStateDefinition] | None = None) -> MagicMock:
    """Create a mock BehavioralStateStore."""
    store = MagicMock()
    store.list_definitions = AsyncMock(return_value=definitions or [])
    store.save_tracker = AsyncMock()
    store.get_tracker = AsyncMock(return_value=None)
    store.record_co_activation = AsyncMock()
    return store


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _iso_offset(minutes: float) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


# ── Task 5 Tests ──────────────────────────────────────────────────────────


class TestTriggerCreatesActiveState:
    """test_trigger_creates_active_state — state_changed matching trigger → ActiveState created."""

    @pytest.mark.asyncio
    async def test_trigger_creates_active_state(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition()
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Simulate a state_changed event matching the trigger
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )

        assert len(detector._active_states) == 1
        active = detector._active_states[0]
        assert active.definition_id == "def-1"
        assert active.matched_confirming == []
        assert "binary_sensor.motion_kitchen" in active.pending_confirming

        await detector.shutdown()


class TestConfirmingUpdatesActiveState:
    """test_confirming_updates_active_state — confirming event updates matched_confirming."""

    @pytest.mark.asyncio
    async def test_confirming_updates_active_state(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition()
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Trigger
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )

        # Confirming
        await detector._on_state_changed(
            {
                "entity_id": "binary_sensor.motion_kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "binary_sensor",
            }
        )

        assert len(detector._active_states) == 1
        active = detector._active_states[0]
        assert "binary_sensor.motion_kitchen" in active.matched_confirming
        assert active.pending_confirming == []

        await detector.shutdown()


class TestWindowExpiryRecordsObservation:
    """test_window_expiry_records_observation — expired ActiveState with sufficient match_ratio → tracker updated."""

    @pytest.mark.asyncio
    async def test_window_expiry_records_observation(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition(duration_minutes=0.0)  # expires immediately
        hub = _make_hub()
        store = _make_store([defn])
        # Return a tracker to update
        tracker = BehavioralStateTracker(definition_id="def-1")
        store.get_tracker = AsyncMock(return_value=tracker)

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Trigger
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )

        # Confirm the indicator so match_ratio > threshold
        await detector._on_state_changed(
            {
                "entity_id": "binary_sensor.motion_kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "binary_sensor",
            }
        )

        # Force the window to be expired
        for state in detector._active_states:
            state.window_expires = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()

        # Run expiry check
        await detector._check_expiry()

        # Active state should be removed
        assert len(detector._active_states) == 0
        # Tracker should have been saved with an observation
        store.save_tracker.assert_called_once()
        saved_tracker = store.save_tracker.call_args[0][0]
        assert saved_tracker.observation_count == 1

        await detector.shutdown()


class TestWindowExpiryDiscardsLowMatch:
    """test_window_expiry_discards_low_match — expired ActiveState with low match_ratio → discarded."""

    @pytest.mark.asyncio
    async def test_window_expiry_discards_low_match(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition(
            confirming=[
                ("binary_sensor.motion_kitchen", "on"),
                ("switch.fan_kitchen", "on"),
            ],
        )
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Trigger (creates ActiveState with 2 pending, 0 matched → ratio=0.0)
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )

        # Don't confirm anything — ratio stays 0.0

        # Force expiry
        for state in detector._active_states:
            state.window_expires = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()

        await detector._check_expiry()

        assert len(detector._active_states) == 0
        store.save_tracker.assert_not_called()

        await detector.shutdown()


class TestEntityIndexLookup:
    """test_entity_index_lookup — events for non-indexed entities are ignored O(1)."""

    @pytest.mark.asyncio
    async def test_entity_index_lookup(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition()
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Event for an entity NOT in any definition
        await detector._on_state_changed(
            {
                "entity_id": "sensor.outdoor_temperature",
                "new_state": "25.3",
                "old_state": "24.8",
                "domain": "sensor",
            }
        )

        # No active states should be created
        assert len(detector._active_states) == 0

        await detector.shutdown()


class TestPersonDepartureTerminates:
    """test_person_departure_terminates — person.X goes away → attributed ActiveStates terminated."""

    @pytest.mark.asyncio
    async def test_person_departure_terminates(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition(person="person.justin")
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Trigger
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )
        assert len(detector._active_states) == 1

        # Person departs
        await detector._on_state_changed(
            {
                "entity_id": "person.justin",
                "new_state": "away",
                "old_state": "home",
                "domain": "person",
            }
        )

        assert len(detector._active_states) == 0

        await detector.shutdown()


class TestDomainFilter:
    """test_domain_filter — detector subscribes only to domains present in definitions."""

    @pytest.mark.asyncio
    async def test_domain_filter(self) -> None:
        from aria.iw.detector import IWDetector

        defn = _make_definition()
        hub = _make_hub()
        store = _make_store([defn])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # The domain set should include light, binary_sensor (from definitions)
        # and person (always included for departure detection)
        assert "light" in detector._domain_set
        assert "binary_sensor" in detector._domain_set
        assert "person" in detector._domain_set

        # A domain NOT in definitions should not be in the set
        assert "climate" not in detector._domain_set

        await detector.shutdown()


class TestMultipleActiveStates:
    """test_multiple_active_states — two overlapping definitions can be active simultaneously."""

    @pytest.mark.asyncio
    async def test_multiple_active_states(self) -> None:
        from aria.iw.detector import IWDetector

        defn1 = _make_definition(def_id="def-1", trigger_entity="light.kitchen")
        defn2 = _make_definition(
            def_id="def-2",
            trigger_entity="light.kitchen",
            trigger_state="on",
            confirming=[("switch.exhaust_kitchen", "on")],
        )
        hub = _make_hub()
        store = _make_store([defn1, defn2])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        # Both share the same trigger entity
        await detector._on_state_changed(
            {
                "entity_id": "light.kitchen",
                "new_state": "on",
                "old_state": "off",
                "domain": "light",
            }
        )

        assert len(detector._active_states) == 2
        ids = {s.definition_id for s in detector._active_states}
        assert ids == {"def-1", "def-2"}

        await detector.shutdown()


class TestDefinitionRefresh:
    """test_definition_refresh — after refresh_definitions(), new definitions are picked up."""

    @pytest.mark.asyncio
    async def test_definition_refresh(self) -> None:
        from aria.iw.detector import IWDetector

        defn1 = _make_definition(def_id="def-1")
        hub = _make_hub()
        store = _make_store([defn1])

        detector = IWDetector("iw_detector", hub, store)
        await detector.initialize()

        assert "light.kitchen" in detector._entity_index

        # Add a new definition
        defn2 = _make_definition(
            def_id="def-2",
            trigger_entity="light.bedroom",
            confirming=[("binary_sensor.motion_bedroom", "on")],
        )
        store.list_definitions = AsyncMock(return_value=[defn1, defn2])

        await detector.refresh_definitions()

        assert "light.bedroom" in detector._entity_index
        assert "binary_sensor.motion_bedroom" in detector._entity_index

        await detector.shutdown()
