"""Tests for aria.iw.models — TDD: written before implementation."""

from __future__ import annotations

import pytest

from aria.iw.models import ActiveState, BehavioralStateDefinition, BehavioralStateTracker, Indicator

# ── Indicator ─────────────────────────────────────────────────────────────────


class TestIndicatorStateChange:
    def test_state_change_mode(self) -> None:
        ind = Indicator(
            entity_id="binary_sensor.door",
            role="trigger",
            mode="state_change",
            expected_state="on",
        )
        assert ind.entity_id == "binary_sensor.door"
        assert ind.role == "trigger"
        assert ind.mode == "state_change"
        assert ind.expected_state == "on"
        assert ind.quiet_seconds is None
        assert ind.threshold_value is None
        assert ind.threshold_direction is None
        assert ind.max_delay_seconds == 0
        assert ind.confidence == 0.0

    def test_confirming_role(self) -> None:
        ind = Indicator(
            entity_id="light.living_room",
            role="confirming",
            mode="state_change",
            expected_state="on",
        )
        assert ind.role == "confirming"

    def test_deviation_role(self) -> None:
        ind = Indicator(
            entity_id="sensor.motion",
            role="deviation",
            mode="state_change",
            expected_state="off",
        )
        assert ind.role == "deviation"


class TestIndicatorQuietPeriod:
    def test_quiet_period_mode(self) -> None:
        ind = Indicator(
            entity_id="binary_sensor.motion",
            role="confirming",
            mode="quiet_period",
            quiet_seconds=300,
        )
        assert ind.mode == "quiet_period"
        assert ind.quiet_seconds == 300

    def test_quiet_period_defaults(self) -> None:
        ind = Indicator(
            entity_id="sensor.x",
            role="trigger",
            mode="quiet_period",
        )
        assert ind.quiet_seconds is None


class TestIndicatorThreshold:
    def test_threshold_mode_above(self) -> None:
        ind = Indicator(
            entity_id="sensor.temperature",
            role="trigger",
            mode="threshold",
            threshold_value=22.5,
            threshold_direction="above",
        )
        assert ind.mode == "threshold"
        assert ind.threshold_value == 22.5
        assert ind.threshold_direction == "above"

    def test_threshold_mode_below(self) -> None:
        ind = Indicator(
            entity_id="sensor.humidity",
            role="confirming",
            mode="threshold",
            threshold_value=60.0,
            threshold_direction="below",
        )
        assert ind.threshold_direction == "below"


class TestIndicatorValidation:
    def test_invalid_role_raises(self) -> None:
        with pytest.raises(ValueError, match="role"):
            Indicator(
                entity_id="sensor.x",
                role="bad_role",
                mode="state_change",
            )

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="bad_mode",
            )

    def test_frozen_immutability(self) -> None:
        ind = Indicator(entity_id="sensor.x", role="trigger", mode="state_change")
        with pytest.raises((AttributeError, TypeError)):
            ind.entity_id = "sensor.y"  # type: ignore[misc]


class TestIndicatorSerialization:
    def test_to_dict_roundtrip(self) -> None:
        ind = Indicator(
            entity_id="binary_sensor.door",
            role="trigger",
            mode="threshold",
            threshold_value=1.5,
            threshold_direction="above",
            max_delay_seconds=30,
            confidence=0.8,
        )
        d = ind.to_dict()
        restored = Indicator.from_dict(d)
        assert restored == ind

    def test_to_dict_contains_all_fields(self) -> None:
        ind = Indicator(entity_id="sensor.x", role="trigger", mode="state_change")
        d = ind.to_dict()
        assert "entity_id" in d
        assert "role" in d
        assert "mode" in d


# ── BehavioralStateDefinition ─────────────────────────────────────────────────


class TestBehavioralStateDefinition:
    def _trigger(self) -> Indicator:
        return Indicator(entity_id="binary_sensor.door", role="trigger", mode="state_change")

    def test_minimal_creation(self) -> None:
        trigger = self._trigger()
        bsd = BehavioralStateDefinition(
            id="bsd-001",
            name="Morning Routine",
            trigger=trigger,
            trigger_preconditions=[],
            confirming=[],
            deviations=[],
            areas=frozenset({"kitchen"}),
            day_types=frozenset({"weekday"}),
            person_attribution=None,
            typical_duration_minutes=30.0,
            expected_outcomes=(),
        )
        assert bsd.id == "bsd-001"
        assert bsd.name == "Morning Routine"
        assert bsd.trigger == trigger
        assert bsd.areas == frozenset({"kitchen"})
        assert bsd.composite_of == ()

    def test_with_indicators(self) -> None:
        trigger = self._trigger()
        confirming = Indicator(entity_id="light.kitchen", role="confirming", mode="state_change", expected_state="on")
        deviation = Indicator(entity_id="sensor.motion", role="deviation", mode="quiet_period", quiet_seconds=60)

        bsd = BehavioralStateDefinition(
            id="bsd-002",
            name="Cooking",
            trigger=trigger,
            trigger_preconditions=[],
            confirming=[confirming],
            deviations=[deviation],
            areas=frozenset({"kitchen"}),
            day_types=frozenset({"weekday", "weekend"}),
            person_attribution="person.alice",
            typical_duration_minutes=45.0,
            expected_outcomes=({"entity_id": "light.kitchen", "state": "on"},),
        )
        assert len(bsd.confirming) == 1
        assert len(bsd.deviations) == 1

    def test_frozen_immutability(self) -> None:
        trigger = self._trigger()
        bsd = BehavioralStateDefinition(
            id="bsd-003",
            name="Test",
            trigger=trigger,
            trigger_preconditions=[],
            confirming=[],
            deviations=[],
            areas=frozenset(),
            day_types=frozenset(),
            person_attribution=None,
            typical_duration_minutes=0.0,
            expected_outcomes=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            bsd.name = "Changed"  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        trigger = self._trigger()
        confirming = Indicator(entity_id="light.x", role="confirming", mode="state_change")
        bsd = BehavioralStateDefinition(
            id="bsd-004",
            name="Evening Relaxation",
            trigger=trigger,
            trigger_preconditions=[],
            confirming=[confirming],
            deviations=[],
            areas=frozenset({"living_room"}),
            day_types=frozenset({"weekend"}),
            person_attribution=None,
            typical_duration_minutes=120.0,
            expected_outcomes=({"entity_id": "light.sofa", "state": "on"},),
            composite_of=("bsd-001", "bsd-002"),
        )
        d = bsd.to_dict()
        restored = BehavioralStateDefinition.from_dict(d)
        assert restored.id == bsd.id
        assert restored.name == bsd.name
        assert restored.areas == bsd.areas
        assert restored.composite_of == bsd.composite_of
        assert len(restored.confirming) == 1


# ── BehavioralStateTracker ────────────────────────────────────────────────────


class TestBehavioralStateTracker:
    def test_default_lifecycle(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        assert tracker.lifecycle == "seed"
        assert tracker.observation_count == 0
        assert tracker.consistency == 0.0

    def test_invalid_lifecycle_raises(self) -> None:
        with pytest.raises(ValueError, match="lifecycle"):
            BehavioralStateTracker(definition_id="bsd-001", lifecycle="invalid")

    def test_valid_lifecycles(self) -> None:
        for lc in ("seed", "emerging", "confirmed", "mature", "dormant", "retired"):
            t = BehavioralStateTracker(definition_id="bsd-001", lifecycle=lc)
            assert t.lifecycle == lc

    def test_record_observation_increments_count(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.record_observation("2026-01-01T08:00:00", 1.0)
        assert tracker.observation_count == 1

    def test_record_observation_updates_last_seen(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.record_observation("2026-01-01T08:00:00", 1.0)
        assert tracker.last_seen == "2026-01-01T08:00:00"

    def test_record_observation_sets_first_seen(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.record_observation("2026-01-01T08:00:00", 1.0)
        assert tracker.first_seen == "2026-01-01T08:00:00"

    def test_record_observation_running_average(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.record_observation("2026-01-01T08:00:00", 1.0)
        tracker.record_observation("2026-01-02T08:00:00", 0.5)
        # Running average: (1.0 + 0.5) / 2 = 0.75
        assert abs(tracker.consistency - 0.75) < 1e-9

    def test_record_multiple_observations(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        for i in range(5):
            tracker.record_observation(f"2026-01-0{i + 1}T08:00:00", 0.8)
        assert tracker.observation_count == 5
        assert abs(tracker.consistency - 0.8) < 1e-9

    def test_json_roundtrip(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="emerging")
        tracker.record_observation("2026-01-01T08:00:00", 0.9)
        d = tracker.to_dict()
        restored = BehavioralStateTracker.from_dict(d)
        assert restored.definition_id == tracker.definition_id
        assert restored.lifecycle == tracker.lifecycle
        assert restored.observation_count == tracker.observation_count
        assert abs(restored.consistency - tracker.consistency) < 1e-9


# ── ActiveState ───────────────────────────────────────────────────────────────


class TestActiveState:
    def test_empty_match_ratio(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=[],
            pending_confirming=[],
            window_expires="2026-01-01T08:01:00",
        )
        assert active.match_ratio == 0.0

    def test_partial_match_ratio(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=["light.x"],
            pending_confirming=["light.y", "light.z"],
            window_expires="2026-01-01T08:01:00",
        )
        # 1 matched / (1 + 2) total = 0.333...
        assert abs(active.match_ratio - (1 / 3)) < 1e-9

    def test_full_match_ratio(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=["light.x", "light.y"],
            pending_confirming=[],
            window_expires="2026-01-01T08:01:00",
        )
        assert active.match_ratio == 1.0
