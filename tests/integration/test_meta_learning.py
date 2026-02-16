"""Tier 2: Ollama meta-learning validation tests.

Tests parse_suggestions() with recorded/sample LLM responses.
No live Ollama required for non-ollama-marked tests.
"""

import json

import pytest

from aria.engine.llm.meta_learning import parse_suggestions

# Recorded Ollama response for CI replay — matches real deepseek-r1 output format.
# Uses the actual schema: action, target, reason, expected_impact, confidence.
SAMPLE_META_RESPONSE = """<think>
Looking at the model performance, power_watts has R2=0.45 and lights_on has R2=0.32.
The feature importance shows hour_sin and people_home_count are the top features.
I should suggest enabling interaction features to capture the relationship.
</think>

```json
[
  {
    "action": "enable_feature",
    "target": "people_home_x_hour_sin",
    "reason": "people_home_count and hour_sin both top features. Interaction captures occupancy-time pattern.",
    "expected_impact": "power_watts MAE -5-10%",
    "confidence": "medium"
  },
  {
    "action": "enable_feature",
    "target": "daylight_x_lights",
    "reason": "lights_on R2=0.32 is the weakest metric. daylight and lights correlated, no interaction feature.",
    "expected_impact": "lights_on R2 +0.05",
    "confidence": "low"
  }
]
```"""


class TestParseMetaLearningOutput:
    """Meta-learning output should parse correctly from recorded responses."""

    def test_parse_valid_response(self):
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        assert len(suggestions) == 2
        assert suggestions[0]["action"] == "enable_feature"
        assert suggestions[0]["target"] == "people_home_x_hour_sin"

    def test_parse_empty_response(self):
        suggestions = parse_suggestions("")
        assert suggestions == []

    def test_parse_response_with_no_json(self):
        suggestions = parse_suggestions("No changes needed at this time.")
        assert suggestions == []

    def test_parse_response_with_think_block_only(self):
        response = "<think>\nI analyzed the data but found no improvements.\n</think>\n\nNo suggestions."
        suggestions = parse_suggestions(response)
        assert suggestions == []

    def test_suggestions_have_required_fields(self):
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        for s in suggestions:
            assert "action" in s, "Each suggestion must have an 'action' field"
            assert "target" in s, "Each suggestion must have a 'target' field"

    def test_suggestions_have_reasoning(self):
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        for s in suggestions:
            assert "reason" in s
            assert len(s["reason"]) > 10

    def test_suggestions_capped_at_max(self):
        """parse_suggestions caps at MAX_META_CHANGES_PER_WEEK (3)."""
        many_suggestions = json.dumps(
            [{"action": "enable_feature", "target": f"feat_{i}", "reason": "test"} for i in range(10)]
        )
        suggestions = parse_suggestions(many_suggestions)
        assert len(suggestions) <= 3

    def test_invalid_suggestions_filtered_out(self):
        """Suggestions missing required fields are dropped.

        Note: parse_suggestions slices to MAX_META_CHANGES_PER_WEEK (3) first,
        then filters for valid entries. So we put valid items within the first 3.
        """
        response = json.dumps(
            [
                {"action": "enable_feature", "target": "valid_one"},
                {"action": "disable_feature", "target": "valid_two"},
                {"reason": "missing action and target"},
            ]
        )
        suggestions = parse_suggestions(response)
        assert len(suggestions) == 2
        assert suggestions[0]["target"] == "valid_one"
        assert suggestions[1]["target"] == "valid_two"

    def test_missing_action_or_target_dropped(self):
        """Entries with only action or only target are excluded."""
        response = json.dumps(
            [
                {"action": "enable_feature"},  # missing target
                {"target": "missing_action"},  # missing action
                {"action": "enable_feature", "target": "valid"},
            ]
        )
        suggestions = parse_suggestions(response)
        assert len(suggestions) == 1
        assert suggestions[0]["target"] == "valid"


class TestMetaLearningOutputValid:
    """Meta-learning suggestions should reference valid interaction features."""

    def test_suggested_targets_exist_in_config(self):
        from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG

        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        interaction_features = DEFAULT_FEATURE_CONFIG.get("interaction_features", {})
        for s in suggestions:
            if s["action"] in ("enable_feature", "disable_feature"):
                target = s["target"]
                assert target in interaction_features, f"Target '{target}' not found in interaction_features"


class TestParseMetaLearningFromFixture:
    """Parse suggestions from the saved fixture file."""

    def test_parse_fixture_response(self):
        import pathlib

        fixture_path = (
            pathlib.Path(__file__).parent.parent / "fixtures" / "ollama_responses" / "meta_learning_sample.json"
        )
        with open(fixture_path) as f:
            data = json.load(f)

        suggestions = parse_suggestions(data["response"])
        assert len(suggestions) >= 1
        assert suggestions[0]["action"] == "enable_feature"
        assert suggestions[0]["target"] == "people_home_x_hour_sin"


@pytest.mark.ollama
class TestMetaLearningLive:
    """Tests that require a running Ollama instance. Skip in CI."""

    def test_meta_learning_placeholder(self):
        """Placeholder — requires Ollama integration wiring."""
        pytest.skip("Requires live Ollama instance — run with -m ollama")
