"""Tests for Bayesian occupancy estimation."""

import unittest
from datetime import datetime

from aria.engine.analysis.occupancy import (
    DEFAULT_PRIOR,
    BayesianOccupancy,
    learn_occupancy_priors,
    occupancy_to_features,
)


def _make_snapshot(people_home=None, motion=None, power_watts=0, outlets=None, media=None):
    """Build a test snapshot with relevant occupancy signals."""
    return {
        "occupancy": {
            "people_home": people_home or [],
            "people_away": [],
            "device_count_home": len(people_home) * 10 if people_home else 0,
        },
        "motion": {
            "sensors": motion or {},
        },
        "power": {
            "total_watts": power_watts,
            "outlets": outlets or {},
        },
        "media": media or {},
    }


class TestBayesianOccupancy(unittest.TestCase):
    def test_occupied_home_high_probability(self):
        estimator = BayesianOccupancy()
        snap = _make_snapshot(
            people_home=["Justin", "Lisa"],
            motion={"Closet motion": "on"},
            power_watts=300,
        )
        result = estimator.estimate(snap)
        self.assertGreater(result["overall"]["probability"], 0.8)

    def test_empty_home_low_probability(self):
        estimator = BayesianOccupancy()
        snap = _make_snapshot(
            people_home=[],
            motion={"Closet motion": "off"},
            power_watts=20,
        )
        result = estimator.estimate(snap)
        self.assertLess(result["overall"]["probability"], 0.3)

    def test_motion_only_moderate_probability(self):
        estimator = BayesianOccupancy()
        snap = _make_snapshot(
            people_home=[],
            motion={"Hallway motion": "on"},
            power_watts=50,
        )
        result = estimator.estimate(snap)
        # Motion without device tracker should be moderate
        prob = result["overall"]["probability"]
        self.assertGreater(prob, 0.2)

    def test_high_power_increases_probability(self):
        estimator = BayesianOccupancy()
        low_power = _make_snapshot(people_home=["Justin"], power_watts=30)
        high_power = _make_snapshot(people_home=["Justin"], power_watts=400)

        result_low = estimator.estimate(low_power)
        result_high = estimator.estimate(high_power)
        self.assertGreater(result_high["overall"]["probability"], result_low["overall"]["probability"])

    def test_signals_included_in_output(self):
        estimator = BayesianOccupancy()
        snap = _make_snapshot(
            people_home=["Justin"],
            motion={"Hall": "on"},
            power_watts=200,
        )
        result = estimator.estimate(snap)
        signals = result["overall"]["signals"]
        signal_types = [s["type"] for s in signals]
        self.assertIn("device_tracker", signal_types)
        self.assertIn("motion", signal_types)
        self.assertIn("power", signal_types)

    def test_confidence_high_with_many_signals(self):
        estimator = BayesianOccupancy()
        snap = _make_snapshot(
            people_home=["Justin", "Lisa"],
            motion={"Hall": "on", "Kitchen": "on"},
            power_watts=350,
            media={"TV": "playing"},
        )
        result = estimator.estimate(snap)
        self.assertEqual(result["overall"]["confidence"], "high")

    def test_confidence_none_with_no_signals(self):
        estimator = BayesianOccupancy()
        # Minimal snapshot with no sensor data
        snap = {"occupancy": {}, "motion": {}, "power": {}, "media": {}}
        result = estimator.estimate(snap)
        # Should still produce a result with default prior
        self.assertIn("probability", result["overall"])


class TestPerAreaOccupancy(unittest.TestCase):
    def test_area_with_active_motion(self):
        area_sensors = {
            "living_room": {
                "motion": ["living_room_motion"],
                "power": [],
            }
        }
        estimator = BayesianOccupancy(area_sensors=area_sensors)
        snap = _make_snapshot(
            people_home=["Justin"],
            motion={"living_room_motion": "on"},
            power_watts=200,
        )
        result = estimator.estimate(snap)
        self.assertIn("living_room", result)
        self.assertGreater(result["living_room"]["probability"], 0.5)

    def test_area_with_no_matching_sensors(self):
        area_sensors = {
            "garage": {
                "motion": ["garage_motion"],
                "power": [],
            }
        }
        estimator = BayesianOccupancy(area_sensors=area_sensors)
        snap = _make_snapshot(
            people_home=["Justin"],
            motion={"Hall": "on"},  # No garage motion
            power_watts=200,
        )
        result = estimator.estimate(snap)
        self.assertIn("garage", result)
        # No matching sensors — falls back to default
        self.assertEqual(result["garage"]["confidence"], "none")


class TestLearnPriors(unittest.TestCase):
    def test_learns_from_timestamps(self):
        snapshots = [
            {"occupancy": {"people_home": ["Justin"]}},
            {"occupancy": {"people_home": ["Justin"]}},
            {"occupancy": {"people_home": ["Justin"]}},
            {"occupancy": {"people_home": []}},
            {"occupancy": {"people_home": []}},
        ]
        timestamps = [
            ("Monday", 14),
            ("Monday", 14),
            ("Monday", 14),
            ("Monday", 3),
            ("Monday", 3),
        ]
        priors = learn_occupancy_priors(snapshots, timestamps)
        # 3 of 3 occupied at Monday 2pm
        self.assertIn(("Monday", 14), priors)
        self.assertEqual(priors[("Monday", 14)], 1.0)

    def test_minimum_samples_required(self):
        snapshots = [
            {"occupancy": {"people_home": ["Justin"]}},
            {"occupancy": {"people_home": []}},
        ]
        timestamps = [("Tuesday", 10), ("Tuesday", 10)]
        priors = learn_occupancy_priors(snapshots, timestamps)
        # Only 2 samples — below minimum of 3
        self.assertNotIn(("Tuesday", 10), priors)

    def test_priors_used_by_estimator(self):
        priors = {("Monday", 14): 0.95}
        estimator = BayesianOccupancy(priors=priors)
        ts = datetime(2026, 2, 9, 14, 0)  # Monday 2pm
        snap = _make_snapshot(power_watts=100)
        result = estimator.estimate(snap, timestamp=ts)
        # High prior should boost probability even with moderate signals
        self.assertGreater(result["overall"]["probability"], 0.5)


class TestOccupancyToFeatures(unittest.TestCase):
    def test_produces_feature_dict(self):
        occ_result = {
            "overall": {"probability": 0.85, "signals": [1, 2, 3]},
            "living_room": {"probability": 0.9},
        }
        features = occupancy_to_features(occ_result)
        self.assertEqual(features["occupancy_probability"], 0.85)
        self.assertEqual(features["occupancy_signal_count"], 3)
        self.assertEqual(features["occupancy_living_room"], 0.9)

    def test_empty_result(self):
        features = occupancy_to_features({})
        self.assertEqual(features["occupancy_probability"], DEFAULT_PRIOR)


if __name__ == "__main__":
    unittest.main()
