"""Tests for entity co-occurrence correlation analysis."""

import unittest

from aria.engine.analysis.entity_correlations import (
    _is_trackable,
    compute_co_occurrences,
    compute_hourly_patterns,
    summarize_entity_correlations,
)


def _make_entries(pairs):
    """Build logbook entries from (entity_id, timestamp_str) pairs."""
    return [{"entity_id": eid, "when": ts} for eid, ts in pairs]


class TestEntityTracking(unittest.TestCase):
    def test_light_is_trackable(self):
        self.assertTrue(_is_trackable("light.living_room"))

    def test_sensor_is_not_trackable(self):
        self.assertFalse(_is_trackable("sensor.temperature"))

    def test_clock_sensor_excluded(self):
        self.assertFalse(_is_trackable("sensor.time_utc"))

    def test_switch_is_trackable(self):
        self.assertTrue(_is_trackable("switch.garage"))

    def test_binary_sensor_is_trackable(self):
        self.assertTrue(_is_trackable("binary_sensor.motion_hallway"))


class TestCoOccurrences(unittest.TestCase):
    def test_finds_co_occurring_pair(self):
        entries = _make_entries(
            [
                # Pattern: motion triggers light, repeated 5 times
                ("binary_sensor.motion_hallway", "2026-02-10T18:00:00+00:00"),
                ("light.hallway", "2026-02-10T18:00:30+00:00"),
                ("binary_sensor.motion_hallway", "2026-02-10T19:00:00+00:00"),
                ("light.hallway", "2026-02-10T19:00:30+00:00"),
                ("binary_sensor.motion_hallway", "2026-02-10T20:00:00+00:00"),
                ("light.hallway", "2026-02-10T20:00:30+00:00"),
                ("binary_sensor.motion_hallway", "2026-02-10T21:00:00+00:00"),
                ("light.hallway", "2026-02-10T21:00:30+00:00"),
                ("binary_sensor.motion_hallway", "2026-02-10T22:00:00+00:00"),
                ("light.hallway", "2026-02-10T22:00:30+00:00"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertGreater(len(result), 0)
        pair = result[0]
        self.assertIn("binary_sensor.motion_hallway", [pair["entity_a"], pair["entity_b"]])
        self.assertIn("light.hallway", [pair["entity_a"], pair["entity_b"]])
        self.assertEqual(pair["count"], 5)

    def test_no_co_occurrence_outside_window(self):
        entries = _make_entries(
            [
                ("light.living_room", "2026-02-10T10:00:00+00:00"),
                ("light.bedroom", "2026-02-10T12:00:00+00:00"),  # 2 hours later
                ("light.living_room", "2026-02-10T14:00:00+00:00"),
                ("light.bedroom", "2026-02-10T16:00:00+00:00"),
                ("light.living_room", "2026-02-10T18:00:00+00:00"),
                ("light.bedroom", "2026-02-10T20:00:00+00:00"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertEqual(len(result), 0)

    def test_conditional_probability(self):
        entries = _make_entries(
            [
                # motion_hallway fires 10 times, light follows 8 of those
                ("binary_sensor.motion_hallway", "2026-02-10T18:00:00"),
                ("light.hallway", "2026-02-10T18:00:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T18:10:00"),
                ("light.hallway", "2026-02-10T18:10:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T18:20:00"),
                ("light.hallway", "2026-02-10T18:20:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T18:30:00"),
                ("light.hallway", "2026-02-10T18:30:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T18:40:00"),
                # No light this time
                ("binary_sensor.motion_hallway", "2026-02-10T18:50:00"),
                ("light.hallway", "2026-02-10T18:50:30"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertGreater(len(result), 0)
        pair = result[0]
        # Light followed motion 5 times; motion happened 6 times
        # So P(light|motion) ~ 5/6 ~ 0.833
        self.assertGreater(pair["conditional_prob_b_given_a"], 0.5)

    def test_minimum_co_occurrence_threshold(self):
        # Only 2 co-occurrences — below minimum of 3
        entries = _make_entries(
            [
                ("light.living_room", "2026-02-10T18:00:00"),
                ("switch.fan", "2026-02-10T18:00:30"),
                ("light.living_room", "2026-02-10T19:00:00"),
                ("switch.fan", "2026-02-10T19:00:30"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertEqual(len(result), 0)

    def test_too_few_entries_returns_empty(self):
        entries = _make_entries(
            [
                ("light.living_room", "2026-02-10T18:00:00"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertEqual(len(result), 0)

    def test_typical_hour_computed(self):
        entries = _make_entries(
            [
                ("binary_sensor.motion_hallway", "2026-02-10T19:00:00"),
                ("light.hallway", "2026-02-10T19:00:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T19:30:00"),
                ("light.hallway", "2026-02-10T19:30:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T19:45:00"),
                ("light.hallway", "2026-02-10T19:45:30"),
                # Extra events to clear the 10-event minimum
                ("binary_sensor.motion_hallway", "2026-02-10T20:00:00"),
                ("light.hallway", "2026-02-10T20:00:30"),
                ("binary_sensor.motion_hallway", "2026-02-10T20:15:00"),
                ("light.hallway", "2026-02-10T20:15:30"),
            ]
        )
        result = compute_co_occurrences(entries, window_minutes=5)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]["typical_hour"], 19)


class TestHourlyPatterns(unittest.TestCase):
    def test_computes_peak_hour(self):
        entries = _make_entries([("light.living_room", f"2026-02-10T{h:02d}:00:00") for h in [18, 18, 18, 19, 19, 20]])
        patterns = compute_hourly_patterns(entries)
        self.assertIn("light.living_room", patterns)
        self.assertEqual(patterns["light.living_room"]["peak_hour"], 18)
        self.assertEqual(patterns["light.living_room"]["total_events"], 6)

    def test_ignores_low_activity_entities(self):
        entries = _make_entries(
            [
                ("light.closet", "2026-02-10T10:00:00"),
                ("light.closet", "2026-02-10T11:00:00"),
            ]
        )
        patterns = compute_hourly_patterns(entries)
        self.assertNotIn("light.closet", patterns)  # < 5 events


class TestSummary(unittest.TestCase):
    def test_summarize_entity_correlations(self):
        co_occurrences = [
            {
                "entity_a": "binary_sensor.motion_hallway",
                "entity_b": "light.hallway",
                "count": 20,
                "conditional_prob_a_given_b": 0.9,
                "conditional_prob_b_given_a": 0.8,
                "typical_hour": 19,
                "strength": "very_strong",
            },
            {
                "entity_a": "light.kitchen",
                "entity_b": "light.living_room",
                "count": 8,
                "conditional_prob_a_given_b": 0.4,
                "conditional_prob_b_given_a": 0.35,
                "typical_hour": 18,
                "strength": "moderate",
            },
        ]
        hourly_patterns = {"light.hallway": {"total_events": 30, "peak_hour": 19}}

        summary = summarize_entity_correlations(co_occurrences, hourly_patterns)
        self.assertEqual(len(summary["top_co_occurrences"]), 2)
        self.assertEqual(len(summary["automation_worthy_pairs"]), 1)
        self.assertGreater(len(summary["most_correlated_entities"]), 0)
        self.assertEqual(summary["entities_with_patterns"], 1)


if __name__ == "__main__":
    unittest.main()
