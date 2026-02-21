"""Tests for template engine — composes full HA automation dict."""

from unittest.mock import MagicMock

from aria.automation.models import ChainLink, DetectionResult
from aria.automation.template_engine import AutomationTemplate


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
        "light.bedroom",
        "binary_sensor.bedroom_motion",
        "sensor.bedroom_illuminance",
        "person.alice",
    ]
    return graph


class TestAutomationStructure:
    """Test the full automation dict has all required HA fields."""

    def test_has_required_keys(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert "id" in result
        assert "alias" in result
        assert "description" in result
        assert "triggers" in result
        assert "conditions" in result
        assert "actions" in result
        assert "mode" in result

    def test_id_is_string(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert isinstance(result["id"], str)
        assert len(result["id"]) > 0

    def test_triggers_is_list(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert isinstance(result["triggers"], list)
        assert len(result["triggers"]) >= 1

    def test_conditions_is_list(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert isinstance(result["conditions"], list)

    def test_actions_is_list(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert isinstance(result["actions"], list)
        assert len(result["actions"]) >= 1


class TestModeSelection:
    """Test automation mode selection based on action type."""

    def test_default_mode_single(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert result["mode"] == "single"

    def test_notification_mode_queued(self):
        det = _detection(action_entities=["notify.mobile_app"])
        det.entity_chain[1] = ChainLink(entity_id="notify.mobile_app", state="on", offset_seconds=30)
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(det)
        assert result["mode"] == "queued"


class TestAliasGeneration:
    """Test human-readable alias generation."""

    def test_alias_contains_area(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert "bedroom" in result["alias"].lower()

    def test_alias_is_human_readable(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert isinstance(result["alias"], str)
        assert len(result["alias"]) > 5


class TestDescriptionGeneration:
    """Test description includes useful metadata."""

    def test_description_mentions_source(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        assert "pattern" in result["description"].lower() or "aria" in result["description"].lower()

    def test_description_includes_confidence(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        # Description should mention confidence or observation count
        desc = result["description"].lower()
        assert "confidence" in desc or "observation" in desc or "85" in desc


class TestIdGeneration:
    """Test automation ID uniqueness and format."""

    def test_id_contains_entity_hint(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        # ID should relate to the trigger or area
        assert "bedroom" in result["id"] or "motion" in result["id"]

    def test_different_detections_different_ids(self):
        template = AutomationTemplate(_mock_entity_graph())
        r1 = template.build(_detection(trigger_entity="binary_sensor.bedroom_motion"))
        r2 = template.build(_detection(trigger_entity="binary_sensor.kitchen_motion"))
        assert r1["id"] != r2["id"]


class TestIntegrationWiring:
    """Test that template engine correctly calls all three builders."""

    def test_trigger_builder_called(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        trigger = result["triggers"][0]
        assert "trigger" in trigger  # trigger type key
        assert "entity_id" in trigger

    def test_condition_builder_called(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        # With bedroom area containing person + illuminance, should have conditions
        assert len(result["conditions"]) >= 1

    def test_action_builder_called(self):
        template = AutomationTemplate(_mock_entity_graph())
        result = template.build(_detection())
        action = result["actions"][0]
        assert "action" in action
        assert "target" in action
