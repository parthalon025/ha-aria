"""Tests for automation validator — 9-check validation suite."""

from unittest.mock import MagicMock

import pytest

from aria.automation.validator import validate_automation


@pytest.fixture
def entity_graph():
    """Mock EntityGraph with known entities."""
    graph = MagicMock()
    graph.has_entity = MagicMock(
        side_effect=lambda eid: (
            eid
            in {
                "binary_sensor.bedroom_motion",
                "light.bedroom",
                "light.kitchen",
                "switch.fan",
                "person.alice",
                "sensor.bedroom_illuminance",
                "lock.front_door",
                "alarm_control_panel.home",
                "cover.garage",
                "notify.mobile",
                "scene.evening",
            }
        )
    )
    return graph


@pytest.fixture
def valid_automation():
    """A minimal valid automation that passes all checks."""
    return {
        "id": "aria_bedroom_motion_abc12345",
        "alias": "Bedroom motion light",
        "description": "Turns on bedroom light on motion.",
        "triggers": [
            {
                "platform": "state",
                "entity_id": "binary_sensor.bedroom_motion",
                "to": "on",
            }
        ],
        "conditions": [],
        "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.bedroom"}}],
        "mode": "single",
    }


class TestCheck1YamlParseable:
    """Check 1: YAML parseable."""

    def test_valid_dict_passes(self, valid_automation, entity_graph):
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert valid
        assert not any("yaml" in e.lower() or "parseable" in e.lower() for e in errors)

    def test_non_dict_fails(self, entity_graph):
        valid, errors = validate_automation("not a dict", entity_graph, set())
        assert not valid
        assert any("dict" in e.lower() or "type" in e.lower() for e in errors)


class TestCheck2RequiredFields:
    """Check 2: Required fields present."""

    def test_all_required_present(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_missing_id(self, valid_automation, entity_graph):
        del valid_automation["id"]
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("id" in e.lower() for e in errors)

    def test_missing_alias(self, valid_automation, entity_graph):
        del valid_automation["alias"]
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("alias" in e.lower() for e in errors)

    def test_missing_triggers(self, valid_automation, entity_graph):
        del valid_automation["triggers"]
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("trigger" in e.lower() for e in errors)

    def test_empty_triggers(self, valid_automation, entity_graph):
        valid_automation["triggers"] = []
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("trigger" in e.lower() for e in errors)

    def test_missing_actions(self, valid_automation, entity_graph):
        del valid_automation["actions"]
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("action" in e.lower() for e in errors)

    def test_empty_actions(self, valid_automation, entity_graph):
        valid_automation["actions"] = []
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("action" in e.lower() for e in errors)


class TestCheck3StateQuoting:
    """Check 3: Boolean-like state values must be strings, not booleans."""

    def test_string_on_passes(self, valid_automation, entity_graph):
        valid_automation["triggers"][0]["to"] = "on"
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_boolean_true_in_trigger_fails(self, valid_automation, entity_graph):
        valid_automation["triggers"][0]["to"] = True
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("boolean" in e.lower() or "quot" in e.lower() or "state" in e.lower() for e in errors)

    def test_boolean_false_in_condition_fails(self, valid_automation, entity_graph):
        valid_automation["conditions"] = [
            {"condition": "state", "entity_id": "binary_sensor.bedroom_motion", "state": False}
        ]
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid

    def test_nested_boolean_in_action_fails(self, valid_automation, entity_graph):
        valid_automation["actions"][0]["data"] = {"state": True}
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid

    def test_numeric_value_passes(self, valid_automation, entity_graph):
        """Numeric values (brightness, temperature) are fine as non-string."""
        valid_automation["actions"][0]["data"] = {"brightness": 255}
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid


class TestCheck4EntitiesExist:
    """Check 4: All referenced entities must exist in EntityGraph."""

    def test_known_entities_pass(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_unknown_trigger_entity_fails(self, valid_automation, entity_graph):
        valid_automation["triggers"][0]["entity_id"] = "binary_sensor.nonexistent"
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("nonexistent" in e for e in errors)

    def test_unknown_action_entity_fails(self, valid_automation, entity_graph):
        valid_automation["actions"][0]["target"]["entity_id"] = "light.nonexistent"
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("nonexistent" in e for e in errors)

    def test_action_entity_list(self, valid_automation, entity_graph):
        """Entity list in target should be checked."""
        valid_automation["actions"][0]["target"] = {"entity_id": ["light.bedroom", "light.nonexistent"]}
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid

    def test_area_id_target_skips_entity_check(self, valid_automation, entity_graph):
        """Area targeting doesn't require entity existence check."""
        valid_automation["actions"][0]["target"] = {"area_id": "bedroom"}
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid


class TestCheck5ServicesValid:
    """Check 5: Service names must follow domain.service format."""

    def test_valid_service(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_invalid_service_format(self, valid_automation, entity_graph):
        valid_automation["actions"][0]["action"] = "invalid_service"
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("service" in e.lower() for e in errors)

    def test_known_services_pass(self, entity_graph):
        """Common HA services should all pass."""
        for service in ["light.turn_on", "switch.turn_off", "climate.set_temperature"]:
            auto = {
                "id": "test_id",
                "alias": "Test",
                "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
                "conditions": [],
                "actions": [{"action": service, "target": {"entity_id": "light.bedroom"}}],
                "mode": "single",
            }
            valid, errors = validate_automation(auto, entity_graph, set())
            assert valid, f"Service {service} should be valid: {errors}"


class TestCheck6NoCircularTrigger:
    """Check 6: Action entity must not be the same as trigger entity."""

    def test_no_circular_passes(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_circular_trigger_action_fails(self, valid_automation, entity_graph):
        """Trigger entity appearing in actions is circular."""
        valid_automation["actions"][0]["target"]["entity_id"] = "binary_sensor.bedroom_motion"
        valid, errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid
        assert any("circular" in e.lower() for e in errors)

    def test_circular_in_entity_list(self, valid_automation, entity_graph):
        """Trigger entity in an action entity list is circular."""
        valid_automation["actions"][0]["target"]["entity_id"] = [
            "light.bedroom",
            "binary_sensor.bedroom_motion",
        ]
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert not valid


class TestCheck7NoDuplicateId:
    """Check 7: ID must not collide with existing automations."""

    def test_unique_id_passes(self, valid_automation, entity_graph):
        existing = {"aria_other_123", "aria_something_456"}
        valid, _errors = validate_automation(valid_automation, entity_graph, existing)
        assert valid

    def test_duplicate_id_fails(self, valid_automation, entity_graph):
        existing = {"aria_bedroom_motion_abc12345", "aria_other_123"}
        valid, errors = validate_automation(valid_automation, entity_graph, existing)
        assert not valid
        assert any("duplicate" in e.lower() for e in errors)


class TestCheck8ModeAppropriate:
    """Check 8: Mode matches action domain semantics."""

    def test_single_mode_for_light_passes(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_notify_needs_queued(self, entity_graph):
        auto = {
            "id": "test_notify_1",
            "alias": "Notify test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "notify.mobile_app", "target": {"entity_id": "notify.mobile"}}],
            "mode": "single",  # Wrong — should be queued
        }
        valid, errors = validate_automation(auto, entity_graph, set())
        assert not valid
        assert any("mode" in e.lower() or "queued" in e.lower() for e in errors)

    def test_notify_with_queued_passes(self, entity_graph):
        auto = {
            "id": "test_notify_2",
            "alias": "Notify test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "notify.mobile_app", "target": {"entity_id": "notify.mobile"}}],
            "mode": "queued",
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert valid

    def test_scene_needs_parallel(self, entity_graph):
        auto = {
            "id": "test_scene_1",
            "alias": "Scene test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "scene.turn_on", "target": {"entity_id": "scene.evening"}}],
            "mode": "single",  # Wrong — should be parallel
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert not valid

    def test_scene_with_parallel_passes(self, entity_graph):
        auto = {
            "id": "test_scene_2",
            "alias": "Scene test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "scene.turn_on", "target": {"entity_id": "scene.evening"}}],
            "mode": "parallel",
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert valid


class TestCheck9RestrictedDomain:
    """Check 9: Restricted domains (lock, alarm, cover) need approval flag."""

    def test_light_domain_no_restriction(self, valid_automation, entity_graph):
        valid, _errors = validate_automation(valid_automation, entity_graph, set())
        assert valid

    def test_lock_without_approval_fails(self, entity_graph):
        auto = {
            "id": "test_lock_1",
            "alias": "Lock test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "lock.lock", "target": {"entity_id": "lock.front_door"}}],
            "mode": "single",
        }
        valid, errors = validate_automation(auto, entity_graph, set())
        assert not valid
        assert any("restricted" in e.lower() or "approval" in e.lower() for e in errors)

    def test_lock_with_approval_flag_passes(self, entity_graph):
        auto = {
            "id": "test_lock_2",
            "alias": "Lock test",
            "description": "ARIA_REQUIRES_APPROVAL — Auto-lock front door.",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "lock.lock", "target": {"entity_id": "lock.front_door"}}],
            "mode": "single",
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert valid

    def test_cover_without_approval_fails(self, entity_graph):
        auto = {
            "id": "test_cover_1",
            "alias": "Cover test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "cover.open_cover", "target": {"entity_id": "cover.garage"}}],
            "mode": "single",
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert not valid

    def test_alarm_without_approval_fails(self, entity_graph):
        auto = {
            "id": "test_alarm_1",
            "alias": "Alarm test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.bedroom_motion", "to": "on"}],
            "conditions": [],
            "actions": [
                {
                    "action": "alarm_control_panel.alarm_arm_away",
                    "target": {"entity_id": "alarm_control_panel.home"},
                }
            ],
            "mode": "single",
        }
        valid, _errors = validate_automation(auto, entity_graph, set())
        assert not valid


class TestEntityGraphNoneGuard:
    """Issue #214: entity_graph=None must not raise AttributeError."""

    def test_entity_graph_none_returns_valid_empty_errors(self):
        """validate_automation with entity_graph=None returns (True, []) for valid automation."""
        auto = {
            "id": "test_none_graph",
            "alias": "Test None Graph",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.bedroom"}}],
            "mode": "single",
        }
        # Should not raise AttributeError — must return empty errors (entity check skipped)
        valid, errors = validate_automation(auto, None, set())
        assert valid
        assert errors == []

    def test_entity_graph_none_logs_warning(self, caplog):
        """validate_automation with entity_graph=None emits a warning."""
        import logging

        auto = {
            "id": "test_none_graph_log",
            "alias": "Test",
            "triggers": [{"platform": "state", "entity_id": "binary_sensor.motion", "to": "on"}],
            "conditions": [],
            "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.bedroom"}}],
            "mode": "single",
        }
        with caplog.at_level(logging.WARNING, logger="aria.automation.validator"):
            validate_automation(auto, None, set())
        assert any("entity_graph" in r.message.lower() for r in caplog.records)


class TestMultipleErrors:
    """Test that validator collects all errors, not just the first."""

    def test_multiple_errors_collected(self, entity_graph):
        auto = {
            # Missing id, alias, triggers empty, actions empty
            "triggers": [],
            "conditions": [],
            "actions": [],
            "mode": "single",
        }
        valid, errors = validate_automation(auto, entity_graph, set())
        assert not valid
        assert len(errors) >= 3  # At least id, alias, triggers, actions missing
