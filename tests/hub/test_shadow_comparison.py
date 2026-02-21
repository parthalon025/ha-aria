"""Unit tests for Shadow Comparison Engine.

Tests duplicate detection (exact, superset, subset), conflict detection
(opposite action, parameter conflict), gap detection (cross-area, disabled),
and EntityGraph integration for area-based set comparison.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.automation.models import ShadowResult
from aria.shared.entity_graph import EntityGraph
from aria.shared.shadow_comparison import compare_candidate

# ============================================================================
# Helpers
# ============================================================================


def make_entity_graph(entities=None, devices=None, areas=None) -> EntityGraph:
    """Create an EntityGraph with test data."""
    graph = EntityGraph()
    graph.update(
        entities=entities or {},
        devices=devices or {},
        areas=areas or [],
    )
    return graph


def make_candidate(
    trigger: list | None = None,
    action: list | None = None,
    condition: list | None = None,
    alias: str = "ARIA: Evening Lights",
    id: str = "aria_gen_abc123",
) -> dict[str, Any]:
    """Build a candidate automation."""
    return {
        "id": id,
        "alias": alias,
        "trigger": trigger or [{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
        "action": action or [{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        "condition": condition or [],
        "mode": "single",
    }


def make_ha_automation(  # noqa: PLR0913
    id: str = "automation.evening_lights",
    alias: str = "Evening Lights",
    trigger: list | None = None,
    action: list | None = None,
    condition: list | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Build an existing HA automation."""
    return {
        "id": id,
        "alias": alias,
        "trigger": trigger or [{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
        "action": action or [{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        "condition": condition or [],
        "mode": "single",
        "enabled": enabled,
    }


# Default entity graph with bedroom entities
def default_graph() -> EntityGraph:
    return make_entity_graph(
        entities={
            "light.bedroom_main": {"device_id": "dev1", "area_id": None},
            "light.bedroom_lamp": {"device_id": "dev2", "area_id": None},
            "light.hallway": {"device_id": "dev3", "area_id": None},
            "binary_sensor.front_door": {"device_id": "dev4", "area_id": None},
        },
        devices={
            "dev1": {"area_id": "bedroom"},
            "dev2": {"area_id": "bedroom"},
            "dev3": {"area_id": "hallway"},
            "dev4": {"area_id": "living_room"},
        },
        areas=[{"area_id": "bedroom"}, {"area_id": "hallway"}, {"area_id": "living_room"}],
    )


# ============================================================================
# No Existing Automations — New Status
# ============================================================================


class TestNewAutomation:
    """Test when there are no existing automations."""

    def test_no_existing_returns_new(self):
        """Candidate with no existing automations is 'new'."""
        candidate = make_candidate()
        result = compare_candidate(candidate, [], default_graph())

        assert isinstance(result, ShadowResult)
        assert result.status == "new"
        assert result.duplicate_score == 0.0

    def test_empty_ha_automations_returns_new(self):
        """Empty HA automation list means candidate is new."""
        candidate = make_candidate()
        result = compare_candidate(candidate, [], default_graph())
        assert result.status == "new"


# ============================================================================
# Exact Duplicate Detection
# ============================================================================


class TestExactDuplicate:
    """Test exact duplicate detection (same trigger + same targets)."""

    def test_exact_duplicate_same_trigger_same_action(self):
        """Same trigger entity + same action → duplicate."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "duplicate"
        assert result.duplicate_score >= 0.9

    def test_exact_duplicate_entity_list_target(self):
        """Same trigger + same entity list targets → duplicate."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[
                {"service": "light.turn_on", "target": {"entity_id": ["light.bedroom_main", "light.bedroom_lamp"]}}
            ],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[
                    {"service": "light.turn_on", "target": {"entity_id": ["light.bedroom_main", "light.bedroom_lamp"]}}
                ],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "duplicate"
        assert result.duplicate_score >= 0.9

    def test_duplicate_high_score(self):
        """Exact duplicate should have a high duplicate_score."""
        candidate = make_candidate()
        existing = [make_ha_automation()]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.duplicate_score >= 0.9

    def test_different_trigger_not_duplicate(self):
        """Different trigger entity → not duplicate."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.back_door", "to": "on"}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status != "duplicate"


# ============================================================================
# Superset Detection
# ============================================================================


class TestSupersetDetection:
    """Test superset detection (ARIA targets ⊃ existing)."""

    def test_candidate_superset_of_existing(self):
        """Candidate with more targets than existing → superset flagged."""
        # Existing: just bedroom main light
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"entity_id": "light.bedroom_main"}}],
            )
        ]
        # Candidate: bedroom main + bedroom lamp
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[
                {"service": "light.turn_on", "target": {"entity_id": ["light.bedroom_main", "light.bedroom_lamp"]}}
            ],
        )

        result = compare_candidate(candidate, existing, default_graph())
        # Superset should not be suppressed — flagged with explanation
        assert result.status != "duplicate"
        assert "expands" in result.reason.lower() or result.status == "new"


# ============================================================================
# Subset Detection
# ============================================================================


class TestSubsetDetection:
    """Test subset detection (ARIA targets ⊂ existing → suppress)."""

    def test_candidate_subset_of_existing(self):
        """Candidate with fewer targets than existing → suppress as duplicate."""
        # Existing: bedroom main + bedroom lamp
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[
                    {"service": "light.turn_on", "target": {"entity_id": ["light.bedroom_main", "light.bedroom_lamp"]}}
                ],
            )
        ]
        # Candidate: just bedroom main
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"entity_id": "light.bedroom_main"}}],
        )

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "duplicate"
        assert result.duplicate_score >= 0.5


# ============================================================================
# Conflict Detection — Opposite Action
# ============================================================================


class TestConflictOppositeAction:
    """Test conflict detection for opposite actions."""

    def test_opposite_service_detected(self):
        """Same trigger + opposite service (turn_on vs turn_off) → conflict."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_off", "target": {"area_id": "bedroom"}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "conflict"
        assert result.conflicting_automation is not None

    def test_opposite_service_switch(self):
        """switch.turn_on vs switch.turn_off → conflict."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "switch.turn_on", "target": {"entity_id": "switch.bedroom_fan"}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "switch.turn_off", "target": {"entity_id": "switch.bedroom_fan"}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "conflict"


# ============================================================================
# Conflict Detection — Parameter Conflict
# ============================================================================


class TestConflictParameterDifference:
    """Test conflict detection for parameter differences."""

    def test_brightness_conflict(self):
        """Same trigger+target+service, different brightness > 20% → conflict."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}, "data": {"brightness_pct": 100}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}, "data": {"brightness_pct": 30}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "conflict"

    def test_minor_brightness_difference_not_conflict(self):
        """Same trigger+target+service, brightness difference ≤ 20% → not conflict."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}, "data": {"brightness_pct": 80}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}, "data": {"brightness_pct": 90}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status != "conflict"


# ============================================================================
# Gap Detection — Cross Area
# ============================================================================


class TestGapDetectionCrossArea:
    """Test cross-area gap detection."""

    def test_cross_area_gap_fill(self):
        """Existing automation for bedroom but candidate for hallway → gap_fill."""
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
            )
        ]
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "hallway"}}],
        )

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "gap_fill"
        assert result.gap_source_automation is not None

    def test_no_gap_when_area_already_covered(self):
        """Same trigger, same area → not a gap fill."""
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
            )
        ]
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status != "gap_fill"


# ============================================================================
# Gap Detection — Disabled Automations
# ============================================================================


class TestDisabledAutomationGap:
    """Test that disabled automations are NOT treated as duplicates."""

    def test_disabled_not_duplicate(self):
        """Disabled HA automation with same config → not duplicate, candidate is new or gap_fill."""
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
                enabled=False,
            )
        ]
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )

        result = compare_candidate(candidate, existing, default_graph())
        # Disabled automations should not cause suppression
        assert result.status != "duplicate"
        assert "disabled" in result.reason.lower() or "improved" in result.reason.lower() or result.status == "new"


# ============================================================================
# Area Resolution via EntityGraph
# ============================================================================


class TestEntityGraphAreaResolution:
    """Test EntityGraph-based area resolution for set comparison."""

    def test_area_id_expands_to_entities(self):
        """area_id target resolves to entity set via EntityGraph."""
        graph = default_graph()

        # Candidate targets area → should resolve to bedroom entities
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        # Existing targets specific entities in bedroom
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[
                    {"service": "light.turn_on", "target": {"entity_id": ["light.bedroom_main", "light.bedroom_lamp"]}}
                ],
            )
        ]

        result = compare_candidate(candidate, existing, graph)
        # Area resolves to same entities → duplicate
        assert result.status == "duplicate"
        assert result.duplicate_score >= 0.9

    def test_area_vs_partial_entity_list(self):
        """Area target resolving to more entities than explicit list → not exact dup."""
        graph = default_graph()

        # Candidate targets full area
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        # Existing targets only one entity in bedroom
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"entity_id": "light.bedroom_main"}}],
            )
        ]

        result = compare_candidate(candidate, existing, graph)
        # Candidate is a superset — should NOT be suppressed as duplicate
        assert result.status != "duplicate" or result.duplicate_score < 1.0


# ============================================================================
# ShadowResult Structure
# ============================================================================


class TestShadowResultStructure:
    """Test that compare_candidate returns properly structured ShadowResult."""

    def test_result_has_candidate(self):
        """Result contains the original candidate dict."""
        candidate = make_candidate()
        result = compare_candidate(candidate, [], default_graph())
        assert result.candidate == candidate

    def test_result_has_reason(self):
        """Result always includes a human-readable reason."""
        candidate = make_candidate()
        result = compare_candidate(candidate, [], default_graph())
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_duplicate_score_range(self):
        """duplicate_score is always between 0.0 and 1.0."""
        candidate = make_candidate()
        result = compare_candidate(candidate, [], default_graph())
        assert 0.0 <= result.duplicate_score <= 1.0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and defensive behavior."""

    def test_candidate_with_no_triggers(self):
        """Candidate with empty triggers doesn't crash."""
        candidate = make_candidate(trigger=[])
        result = compare_candidate(candidate, [make_ha_automation()], default_graph())
        assert isinstance(result, ShadowResult)

    def test_candidate_with_no_actions(self):
        """Candidate with empty actions doesn't crash."""
        candidate = make_candidate(action=[])
        result = compare_candidate(candidate, [make_ha_automation()], default_graph())
        assert isinstance(result, ShadowResult)

    def test_existing_with_no_triggers(self):
        """Existing automation with empty triggers doesn't crash."""
        candidate = make_candidate()
        existing = [make_ha_automation(trigger=[])]
        result = compare_candidate(candidate, existing, default_graph())
        assert isinstance(result, ShadowResult)

    def test_multiple_existing_automations(self):
        """compare_candidate checks all existing automations."""
        candidate = make_candidate(
            trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        existing = [
            make_ha_automation(
                id="auto.unrelated",
                trigger=[{"platform": "time", "at": "06:00:00"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "kitchen"}}],
            ),
            make_ha_automation(
                id="auto.match",
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
            ),
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status == "duplicate"

    def test_none_entity_graph(self):
        """None entity_graph falls back gracefully."""
        candidate = make_candidate()
        # Pass None entity graph — should still work for direct comparisons
        result = compare_candidate(candidate, [], None)
        assert isinstance(result, ShadowResult)
        assert result.status == "new"

    def test_time_trigger_vs_state_trigger_not_duplicate(self):
        """Different trigger platforms are not duplicates."""
        candidate = make_candidate(
            trigger=[{"platform": "time", "at": "21:00:00"}],
            action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        )
        existing = [
            make_ha_automation(
                trigger=[{"platform": "state", "entity_id": "binary_sensor.front_door", "to": "on"}],
                action=[{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
            )
        ]

        result = compare_candidate(candidate, existing, default_graph())
        assert result.status != "duplicate"
        assert result.status != "conflict"
