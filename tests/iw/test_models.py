"""Tests for aria.iw.models — behavioral state data models."""

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

    def test_quiet_period_missing_seconds_raises(self) -> None:
        """#146: quiet_period mode requires quiet_seconds."""
        with pytest.raises(ValueError, match="quiet_seconds"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="quiet_period",
            )


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

    def test_threshold_missing_value_raises(self) -> None:
        """#146: threshold mode requires threshold_value."""
        with pytest.raises(ValueError, match="threshold_value"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="threshold",
                threshold_direction="above",
            )

    def test_threshold_missing_direction_raises(self) -> None:
        """#146: threshold mode requires threshold_direction."""
        with pytest.raises(ValueError, match="threshold_direction"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="threshold",
                threshold_value=50.0,
            )


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

    def test_confidence_out_of_range_raises(self) -> None:
        """#146: confidence must be [0.0, 1.0]."""
        with pytest.raises(ValueError, match="confidence"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="state_change",
                confidence=1.5,
            )

    def test_confidence_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Indicator(
                entity_id="sensor.x",
                role="trigger",
                mode="state_change",
                confidence=-0.1,
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

    def test_empty_id_raises(self) -> None:
        """#145: id must be non-empty."""
        trigger = self._trigger()
        with pytest.raises(ValueError, match="id"):
            BehavioralStateDefinition(
                id="",
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

    def test_wrong_trigger_role_raises(self) -> None:
        """#145: trigger indicator must have role='trigger'."""
        bad_trigger = Indicator(entity_id="sensor.x", role="confirming", mode="state_change")
        with pytest.raises(ValueError, match="trigger must have role='trigger'"):
            BehavioralStateDefinition(
                id="bsd-bad",
                name="Bad",
                trigger=bad_trigger,
                trigger_preconditions=[],
                confirming=[],
                deviations=[],
                areas=frozenset(),
                day_types=frozenset(),
                person_attribution=None,
                typical_duration_minutes=0.0,
                expected_outcomes=(),
            )

    def test_wrong_confirming_role_raises(self) -> None:
        """#145: confirming indicators must have role='confirming'."""
        trigger = self._trigger()
        bad_confirming = Indicator(entity_id="sensor.x", role="trigger", mode="state_change")
        with pytest.raises(ValueError, match="confirming indicator must have role='confirming'"):
            BehavioralStateDefinition(
                id="bsd-bad",
                name="Bad",
                trigger=trigger,
                trigger_preconditions=[],
                confirming=[bad_confirming],
                deviations=[],
                areas=frozenset(),
                day_types=frozenset(),
                person_attribution=None,
                typical_duration_minutes=0.0,
                expected_outcomes=(),
            )

    def test_wrong_deviation_role_raises(self) -> None:
        """#145: deviation indicators must have role='deviation'."""
        trigger = self._trigger()
        bad_deviation = Indicator(entity_id="sensor.x", role="trigger", mode="state_change")
        with pytest.raises(ValueError, match="deviation indicator must have role='deviation'"):
            BehavioralStateDefinition(
                id="bsd-bad",
                name="Bad",
                trigger=trigger,
                trigger_preconditions=[],
                confirming=[],
                deviations=[bad_deviation],
                areas=frozenset(),
                day_types=frozenset(),
                person_attribution=None,
                typical_duration_minutes=0.0,
                expected_outcomes=(),
            )


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


class TestBehavioralStateTrackerLifecycle:
    """#147: Lifecycle state machine validation."""

    def test_valid_transition_seed_to_emerging(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.transition_lifecycle("emerging", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "emerging"

    def test_valid_transition_emerging_to_confirmed(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="emerging")
        tracker.transition_lifecycle("confirmed", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "confirmed"

    def test_valid_transition_confirmed_to_mature(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="confirmed")
        tracker.transition_lifecycle("mature", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "mature"

    def test_valid_transition_mature_to_dormant(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="mature")
        tracker.transition_lifecycle("dormant", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "dormant"

    def test_valid_transition_dormant_to_emerging(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="dormant")
        tracker.transition_lifecycle("emerging", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "emerging"

    def test_invalid_transition_mature_to_seed(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="mature")
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            tracker.transition_lifecycle("seed", "2026-01-01T08:00:00")

    def test_invalid_transition_retired_to_anything(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="retired")
        with pytest.raises(ValueError, match="Invalid lifecycle transition"):
            tracker.transition_lifecycle("seed", "2026-01-01T08:00:00")

    def test_invalid_target_lifecycle_raises(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        with pytest.raises(ValueError, match="lifecycle"):
            tracker.transition_lifecycle("invalid", "2026-01-01T08:00:00")

    def test_transition_records_history(self) -> None:
        tracker = BehavioralStateTracker(definition_id="bsd-001")
        tracker.transition_lifecycle("emerging", "2026-01-01T08:00:00")
        assert len(tracker.lifecycle_history) == 1
        assert tracker.lifecycle_history[0]["from"] == "seed"
        assert tracker.lifecycle_history[0]["to"] == "emerging"
        assert tracker.lifecycle_history[0]["timestamp"] == "2026-01-01T08:00:00"

    def test_rejection_demotion_emerging_to_seed(self) -> None:
        """emerging can go back to seed on rejection."""
        tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle="emerging")
        tracker.transition_lifecycle("seed", "2026-01-01T08:00:00")
        assert tracker.lifecycle == "seed"

    def test_any_state_can_retire(self) -> None:
        """All non-retired states should be able to transition to retired."""
        for lc in ("seed", "emerging", "confirmed", "mature", "dormant"):
            tracker = BehavioralStateTracker(definition_id="bsd-001", lifecycle=lc)
            tracker.transition_lifecycle("retired", "2026-01-01T08:00:00")
            assert tracker.lifecycle == "retired"


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


class TestActiveStateInvariants:
    """#144: ActiveState disjointness and confirm_indicator."""

    def test_disjoint_invariant_enforced(self) -> None:
        with pytest.raises(ValueError, match="disjoint"):
            ActiveState(
                definition_id="bsd-001",
                trigger_time="2026-01-01T08:00:00",
                matched_confirming=["light.x"],
                pending_confirming=["light.x", "light.y"],
                window_expires="2026-01-01T08:01:00",
            )

    def test_confirm_indicator_moves_to_matched(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=[],
            pending_confirming=["light.x", "light.y"],
            window_expires="2026-01-01T08:01:00",
        )
        active.confirm_indicator("light.x")
        assert "light.x" in active.matched_confirming
        assert "light.x" not in active.pending_confirming
        assert abs(active.match_ratio - 0.5) < 1e-9

    def test_confirm_indicator_not_pending_raises(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=["light.x"],
            pending_confirming=["light.y"],
            window_expires="2026-01-01T08:01:00",
        )
        with pytest.raises(ValueError, match="not in pending_confirming"):
            active.confirm_indicator("light.z")

    def test_confirm_all_indicators(self) -> None:
        active = ActiveState(
            definition_id="bsd-001",
            trigger_time="2026-01-01T08:00:00",
            matched_confirming=[],
            pending_confirming=["light.x", "light.y"],
            window_expires="2026-01-01T08:01:00",
        )
        active.confirm_indicator("light.x")
        active.confirm_indicator("light.y")
        assert active.match_ratio == 1.0
        assert len(active.pending_confirming) == 0


class TestStoreGuard:
    """#185: BehavioralStateStore raises RuntimeError when not initialized."""

    @pytest.mark.asyncio
    async def test_save_definition_without_init_raises(self) -> None:
        from aria.iw.store import BehavioralStateStore

        store = BehavioralStateStore("/tmp/nonexistent.db")
        trigger = Indicator(entity_id="sensor.x", role="trigger", mode="state_change")
        defn = BehavioralStateDefinition(
            id="test",
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
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.save_definition(defn)

    @pytest.mark.asyncio
    async def test_list_definitions_without_init_raises(self) -> None:
        from aria.iw.store import BehavioralStateStore

        store = BehavioralStateStore("/tmp/nonexistent.db")
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.list_definitions()
