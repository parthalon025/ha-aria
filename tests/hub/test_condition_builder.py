"""Tests for condition builder — presence, time, weekday, illuminance, safety."""

from unittest.mock import MagicMock

from aria.automation.condition_builder import SAFETY_CONDITIONS, build_conditions
from aria.automation.models import ChainLink, DetectionResult


def _detection(  # noqa: PLR0913 — test helper
    trigger_entity="binary_sensor.bedroom_motion",
    action_entities=None,
    area_id="bedroom",
    day_type="workday",
    confidence=0.85,
    entity_chain=None,
):
    if action_entities is None:
        action_entities = ["light.bedroom"]
    if entity_chain is None:
        entity_chain = [
            ChainLink(entity_id=trigger_entity, state="on", offset_seconds=0),
            ChainLink(entity_id=action_entities[0], state="on", offset_seconds=30),
        ]
    return DetectionResult(
        source="pattern",
        trigger_entity=trigger_entity,
        action_entities=action_entities,
        entity_chain=entity_chain,
        area_id=area_id,
        confidence=confidence,
        recency_weight=0.9,
        observation_count=47,
        first_seen="2026-01-01T06:30:00",
        last_seen="2026-02-19T06:45:00",
        day_type=day_type,
        combined_score=0.8,
    )


def _mock_entity_graph():
    graph = MagicMock()
    graph.entities_in_area.return_value = [
        "binary_sensor.bedroom_motion",
        "light.bedroom",
        "sensor.bedroom_illuminance",
        "person.justin",
    ]
    return graph


class TestWeekdayCondition:
    """Test weekday condition generation from day_type."""

    def test_workday_generates_weekday_condition(self):
        conditions = build_conditions(_detection(day_type="workday"), _mock_entity_graph())
        weekday_conds = [c for c in conditions if c.get("condition") == "time" and "weekday" in c]
        assert len(weekday_conds) == 1
        assert set(weekday_conds[0]["weekday"]) == {"mon", "tue", "wed", "thu", "fri"}

    def test_weekend_generates_weekend_condition(self):
        conditions = build_conditions(_detection(day_type="weekend"), _mock_entity_graph())
        weekday_conds = [c for c in conditions if c.get("condition") == "time" and "weekday" in c]
        assert len(weekday_conds) == 1
        assert set(weekday_conds[0]["weekday"]) == {"sat", "sun"}

    def test_all_day_type_no_weekday_condition(self):
        conditions = build_conditions(_detection(day_type="all"), _mock_entity_graph())
        weekday_conds = [c for c in conditions if c.get("condition") == "time" and "weekday" in c]
        assert len(weekday_conds) == 0


class TestPresenceCondition:
    """Test presence condition when person entity is in area."""

    def test_presence_added_for_light_action(self):
        conditions = build_conditions(_detection(), _mock_entity_graph())
        presence_conds = [
            c for c in conditions if c.get("condition") == "state" and "person." in c.get("entity_id", "")
        ]
        assert len(presence_conds) >= 1
        assert presence_conds[0]["state"] == '"home"'

    def test_no_presence_if_no_person_in_area(self):
        graph = MagicMock()
        graph.entities_in_area.return_value = ["light.bedroom", "sensor.bedroom_temp"]
        conditions = build_conditions(_detection(), graph)
        presence_conds = [
            c for c in conditions if c.get("condition") == "state" and "person." in c.get("entity_id", "")
        ]
        assert len(presence_conds) == 0


class TestIlluminanceCondition:
    """Test illuminance condition for light actions."""

    def test_illuminance_added_for_light_action(self):
        conditions = build_conditions(_detection(), _mock_entity_graph())
        illum_conds = [
            c for c in conditions if c.get("condition") == "numeric_state" and "illuminance" in c.get("entity_id", "")
        ]
        assert len(illum_conds) >= 1
        assert "below" in illum_conds[0]

    def test_no_illuminance_for_switch_action(self):
        det = _detection(action_entities=["switch.coffee_maker"])
        det.entity_chain[1] = ChainLink(entity_id="switch.coffee_maker", state="on", offset_seconds=30)
        conditions = build_conditions(det, _mock_entity_graph())
        illum_conds = [
            c for c in conditions if c.get("condition") == "numeric_state" and "illuminance" in c.get("entity_id", "")
        ]
        assert len(illum_conds) == 0

    def test_no_illuminance_if_no_sensor_in_area(self):
        graph = MagicMock()
        graph.entities_in_area.return_value = ["light.bedroom", "person.justin"]
        conditions = build_conditions(_detection(), graph)
        illum_conds = [
            c for c in conditions if c.get("condition") == "numeric_state" and "illuminance" in c.get("entity_id", "")
        ]
        assert len(illum_conds) == 0


class TestSafetyConditions:
    """Test safety condition defaults."""

    def test_safety_conditions_dict_exists(self):
        assert isinstance(SAFETY_CONDITIONS, dict)
        assert "light.turn_on" in SAFETY_CONDITIONS

    def test_conditions_list_returned(self):
        conditions = build_conditions(_detection(), _mock_entity_graph())
        assert isinstance(conditions, list)
        for c in conditions:
            assert isinstance(c, dict)
            assert "condition" in c


class TestEmptyConditions:
    """Test edge cases for condition generation."""

    def test_no_area_still_works(self):
        det = _detection(area_id=None)
        graph = MagicMock()
        graph.entities_in_area.return_value = []
        conditions = build_conditions(det, graph)
        assert isinstance(conditions, list)
