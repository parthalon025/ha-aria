"""Tests for LLM integration: think tag stripping, meta-learning parsing/guardrails."""

import copy
import json
import unittest
import tempfile
import shutil

from aria.engine.config import OllamaConfig, PathConfig
from aria.engine.llm.client import strip_think_tags
from aria.engine.llm.meta_learning import (
    parse_suggestions, apply_suggestion_to_config, validate_suggestion,
)
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.storage.data_store import DataStore


HAS_SKLEARN = True
try:
    import numpy  # noqa: F401
except ImportError:
    HAS_SKLEARN = False


class TestDeepseekIntegration(unittest.TestCase):
    def test_strip_think_tags(self):
        text = "<think>Let me think about this...</think>\n\nHere is the answer."
        self.assertEqual(strip_think_tags(text), "Here is the answer.")

    def test_strip_think_tags_multiline(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>\n\nResult here."
        self.assertEqual(strip_think_tags(text), "Result here.")

    def test_strip_think_tags_no_tags(self):
        text = "Just a normal response with no thinking."
        self.assertEqual(strip_think_tags(text), text)

    def test_strip_think_tags_empty(self):
        self.assertEqual(strip_think_tags(""), "")

    def test_ollama_model_is_deepseek(self):
        self.assertEqual(OllamaConfig().model, "deepseek-r1:8b")


class TestMetaLearning(unittest.TestCase):
    def test_parse_suggestions_valid_json(self):
        response = '''Here are my suggestions:
[{"action": "enable_feature", "target": "is_weekend_x_temp", "reason": "weekend power off by 15%", "expected_impact": "power_watts MAE -5%", "confidence": "medium"}]'''
        suggestions = parse_suggestions(response)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["action"], "enable_feature")
        self.assertEqual(suggestions[0]["target"], "is_weekend_x_temp")

    def test_parse_suggestions_with_think_block(self):
        response = '''<think>Let me analyze the accuracy data...
I see that weekend predictions are off by 15%...</think>

Based on my analysis:
[{"action": "enable_feature", "target": "is_weekend_x_temp", "reason": "test", "expected_impact": "test", "confidence": "high"}]'''
        suggestions = parse_suggestions(response)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["action"], "enable_feature")

    def test_parse_suggestions_empty_response(self):
        self.assertEqual(parse_suggestions(""), [])
        self.assertEqual(parse_suggestions("No suggestions."), [])

    def test_parse_suggestions_invalid_json(self):
        self.assertEqual(parse_suggestions("[invalid json}"), [])

    def test_parse_suggestions_max_3(self):
        response = json.dumps([
            {"action": "enable_feature", "target": "a"},
            {"action": "enable_feature", "target": "b"},
            {"action": "enable_feature", "target": "c"},
            {"action": "enable_feature", "target": "d"},
        ])
        suggestions = parse_suggestions(response)
        self.assertEqual(len(suggestions), 3)

    def test_parse_suggestions_filters_invalid(self):
        response = json.dumps([
            {"action": "enable_feature", "target": "a"},
            {"no_action": True},
            {"action": "disable_feature"},
        ])
        suggestions = parse_suggestions(response)
        self.assertEqual(len(suggestions), 1)

    def test_apply_suggestion_enable_interaction(self):
        config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        suggestion = {"action": "enable_feature", "target": "is_weekend_x_temp"}
        result = apply_suggestion_to_config(suggestion, config)
        self.assertTrue(result)
        self.assertTrue(config["interaction_features"]["is_weekend_x_temp"])

    def test_apply_suggestion_disable_interaction(self):
        config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        config["interaction_features"]["is_weekend_x_temp"] = True
        suggestion = {"action": "disable_feature", "target": "is_weekend_x_temp"}
        result = apply_suggestion_to_config(suggestion, config)
        self.assertTrue(result)
        self.assertFalse(config["interaction_features"]["is_weekend_x_temp"])

    def test_apply_suggestion_unknown_target(self):
        config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        suggestion = {"action": "enable_feature", "target": "nonexistent_feature"}
        result = apply_suggestion_to_config(suggestion, config)
        self.assertFalse(result)

    def test_validate_suggestion_insufficient_data(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        config = DEFAULT_FEATURE_CONFIG
        snapshots = []
        suggestion = {"action": "enable_feature", "target": "is_weekend_x_temp"}
        improvement, _ = validate_suggestion(suggestion, snapshots, config)
        self.assertIsNone(improvement)

    def test_load_save_applied_suggestions(self):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=__import__("pathlib").Path(tmpdir))
            paths.ensure_dirs()
            store = DataStore(paths)
            history = store.load_applied_suggestions()
            self.assertEqual(history["total_applied"], 0)
            history["applied"].append({"date": "2026-02-10", "suggestion": {"action": "test"}, "improvement": 3.5})
            history["total_applied"] = 1
            store.save_applied_suggestions(history)
            reloaded = store.load_applied_suggestions()
            self.assertEqual(reloaded["total_applied"], 1)
            self.assertEqual(len(reloaded["applied"]), 1)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
