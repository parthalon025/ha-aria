"""Tests for LLM automation suggestion parsing and validation."""

import json
import unittest

from aria.engine.llm.automation_suggestions import (
    parse_automation_suggestions, _validate_yaml_structure,
    _format_co_occurrences,
)


class TestYAMLValidation(unittest.TestCase):
    def test_valid_yaml_has_trigger_and_action(self):
        yaml = "alias: Test automation\ntrigger:\n  - platform: state\naction:\n  - service: light.turn_on"
        self.assertTrue(_validate_yaml_structure(yaml))

    def test_missing_trigger_fails(self):
        yaml = "action:\n  - service: light.turn_on"
        self.assertFalse(_validate_yaml_structure(yaml))

    def test_missing_action_fails(self):
        yaml = "trigger:\n  - platform: state"
        self.assertFalse(_validate_yaml_structure(yaml))

    def test_empty_string_fails(self):
        self.assertFalse(_validate_yaml_structure(""))


class TestParseAutomationSuggestions(unittest.TestCase):
    def test_parse_valid_suggestion(self):
        response = json.dumps([{
            "description": "Motion triggers hallway light",
            "trigger_entity": "binary_sensor.motion_hallway",
            "action_entity": "light.hallway",
            "confidence": "high",
            "yaml": "alias: Test\ntrigger:\n  - platform: state\naction:\n  - service: light.turn_on",
        }])
        result = parse_automation_suggestions(response)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"], "high")

    def test_parse_with_think_tags(self):
        response = """<think>Let me analyze the patterns...</think>

Based on the data:
[{"description": "Auto light", "yaml": "trigger:\\n  - platform: state\\naction:\\n  - service: light.turn_on"}]"""
        result = parse_automation_suggestions(response)
        self.assertEqual(len(result), 1)

    def test_parse_invalid_json(self):
        result = parse_automation_suggestions("[invalid json}")
        self.assertEqual(result, [])

    def test_parse_empty_response(self):
        result = parse_automation_suggestions("")
        self.assertEqual(result, [])

    def test_filters_invalid_yaml(self):
        response = json.dumps([
            {
                "description": "Valid",
                "yaml": "trigger:\n  - platform: state\naction:\n  - service: light.turn_on",
            },
            {
                "description": "Invalid — no trigger or action",
                "yaml": "just some text",
            },
        ])
        result = parse_automation_suggestions(response)
        self.assertEqual(len(result), 1)

    def test_max_3_suggestions(self):
        suggestions = [{
            "description": f"Suggestion {i}",
            "yaml": "trigger:\n  - platform: state\naction:\n  - service: light.turn_on",
        } for i in range(5)]
        result = parse_automation_suggestions(json.dumps(suggestions))
        self.assertEqual(len(result), 3)

    def test_missing_required_fields_filtered(self):
        response = json.dumps([
            {"description": "Has desc but no yaml"},
            {"yaml": "trigger:\naction:\n"},  # Missing description
            {"description": "Valid", "yaml": "trigger:\n  test\naction:\n  test"},
        ])
        result = parse_automation_suggestions(response)
        self.assertEqual(len(result), 1)


class TestFormatCoOccurrences(unittest.TestCase):
    def test_formats_pairs(self):
        entity_corrs = {
            "top_co_occurrences": [
                {
                    "entity_a": "binary_sensor.motion",
                    "entity_b": "light.hallway",
                    "count": 20,
                    "conditional_prob_a_given_b": 0.9,
                    "conditional_prob_b_given_a": 0.8,
                    "typical_hour": 19,
                    "strength": "very_strong",
                }
            ]
        }
        result = _format_co_occurrences(entity_corrs)
        self.assertIn("binary_sensor.motion", result)
        self.assertIn("light.hallway", result)
        self.assertIn("20x", result)
        self.assertIn("90%", result)

    def test_empty_returns_message(self):
        result = _format_co_occurrences({})
        self.assertIn("No entity correlation", result)


if __name__ == "__main__":
    unittest.main()
