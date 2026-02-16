"""Tests for features: time encoding, feature config, vector builder."""

import copy
import unittest

from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.config import HolidayConfig
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.features.time_features import (
    _time_to_minutes,
    build_time_features,
    cyclical_encode,
)
from aria.engine.features.vector_builder import (
    build_feature_vector,
    build_training_data,
    extract_target_values,
    get_feature_names,
)


class TestTimeFeatures(unittest.TestCase):
    def test_cyclical_encode_periodicity(self):
        s0, c0 = cyclical_encode(0, 24)
        s24, c24 = cyclical_encode(24, 24)
        self.assertAlmostEqual(s0, s24, places=4)
        self.assertAlmostEqual(c0, c24, places=4)

    def test_cyclical_encode_opposite(self):
        s0, c0 = cyclical_encode(0, 24)
        s12, c12 = cyclical_encode(12, 24)
        self.assertAlmostEqual(s0, 0.0, places=4)
        self.assertAlmostEqual(s12, 0.0, places=3)
        self.assertAlmostEqual(c0, 1.0, places=4)
        self.assertAlmostEqual(c12, -1.0, places=4)

    def test_build_time_features_structure(self):
        sun_data = {"sunrise": "06:42", "sunset": "17:58"}
        tf = build_time_features("2026-02-10T16:00:00", sun_data, "2026-02-10")
        required_keys = [
            "hour",
            "hour_sin",
            "hour_cos",
            "dow",
            "dow_sin",
            "dow_cos",
            "month",
            "month_sin",
            "month_cos",
            "day_of_year",
            "day_of_year_sin",
            "day_of_year_cos",
            "is_weekend",
            "is_holiday",
            "is_night",
            "is_work_hours",
            "minutes_since_midnight",
            "minutes_since_sunrise",
            "minutes_until_sunset",
            "daylight_remaining_pct",
            "week_of_year",
        ]
        for key in required_keys:
            self.assertIn(key, tf, f"Missing time feature: {key}")

    def test_time_features_values(self):
        sun_data = {"sunrise": "06:42", "sunset": "17:58"}
        tf = build_time_features("2026-02-10T16:00:00", sun_data, "2026-02-10")
        self.assertEqual(tf["hour"], 16)
        self.assertEqual(tf["dow"], 1)  # Tuesday
        self.assertEqual(tf["month"], 2)
        self.assertFalse(tf["is_weekend"])
        self.assertFalse(tf["is_night"])
        self.assertTrue(tf["is_work_hours"])
        self.assertEqual(tf["minutes_since_midnight"], 960)
        self.assertEqual(tf["minutes_since_sunrise"], 558)
        self.assertEqual(tf["minutes_until_sunset"], 118)

    def test_time_features_night(self):
        sun_data = {"sunrise": "06:42", "sunset": "17:58"}
        tf = build_time_features("2026-02-10T02:00:00", sun_data, "2026-02-10")
        self.assertTrue(tf["is_night"])
        self.assertFalse(tf["is_work_hours"])
        self.assertEqual(tf["daylight_remaining_pct"], 0)

    def test_time_features_weekend(self):
        sun_data = {"sunrise": "06:42", "sunset": "17:58"}
        tf = build_time_features("2026-02-14T10:00:00", sun_data, "2026-02-14")
        self.assertTrue(tf["is_weekend"])  # Saturday
        self.assertFalse(tf["is_work_hours"])

    def test_time_to_minutes(self):
        self.assertEqual(_time_to_minutes("06:42"), 402)
        self.assertEqual(_time_to_minutes("17:58"), 1078)
        self.assertEqual(_time_to_minutes("00:00"), 0)
        self.assertEqual(_time_to_minutes("23:59"), 1439)


class TestFeatureVector(unittest.TestCase):
    def test_load_default_config(self):
        config = DEFAULT_FEATURE_CONFIG
        self.assertIn("time_features", config)
        self.assertIn("weather_features", config)
        self.assertIn("home_features", config)
        self.assertIn("target_metrics", config)
        self.assertTrue(config["time_features"]["hour_sin_cos"])

    def test_get_feature_names(self):
        config = DEFAULT_FEATURE_CONFIG
        names = get_feature_names(config)
        self.assertIn("hour_sin", names)
        self.assertIn("hour_cos", names)
        self.assertIn("weather_temp_f", names)
        self.assertIn("people_home_count", names)
        self.assertIn("prev_snapshot_power", names)
        self.assertNotIn("is_weekend_x_temp", names)
        self.assertGreater(len(names), 20)

    def test_build_feature_vector(self):
        config = DEFAULT_FEATURE_CONFIG
        snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
        snapshot["time_features"] = build_time_features(
            "2026-02-10T16:00:00", {"sunrise": "06:42", "sunset": "17:58"}, "2026-02-10"
        )
        snapshot["weather"] = {"temp_f": 72, "humidity_pct": 60, "wind_mph": 8}
        snapshot["power"]["total_watts"] = 245.3
        snapshot["lights"]["on"] = 8
        snapshot["occupancy"]["people_home_count"] = 2
        snapshot["occupancy"]["device_count_home"] = 62
        snapshot["motion"] = {"sensors": {}, "active_count": 1}
        snapshot["media"] = {"active_players": [], "total_active": 0}
        snapshot["ev"] = {"TARS": {"battery_pct": 71, "is_charging": False}}

        fv = build_feature_vector(snapshot, config)
        self.assertEqual(fv["weather_temp_f"], 72)
        self.assertEqual(fv["people_home_count"], 2)
        self.assertEqual(fv["lights_on"], 8)
        self.assertEqual(fv["ev_battery_pct"], 71)
        self.assertEqual(fv["ev_is_charging"], 0)
        self.assertIn("hour_sin", fv)
        self.assertIn("is_weekend", fv)

    def test_build_feature_vector_with_lag(self):
        config = DEFAULT_FEATURE_CONFIG
        snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
        snapshot["time_features"] = build_time_features("2026-02-10T16:00:00", None, "2026-02-10")
        snapshot["media"] = {"total_active": 0}
        snapshot["motion"] = {"active_count": 0}
        snapshot["ev"] = {}

        prev = build_empty_snapshot("2026-02-10", HolidayConfig())
        prev["power"]["total_watts"] = 200.0
        prev["lights"]["on"] = 5

        fv = build_feature_vector(snapshot, config, prev_snapshot=prev)
        self.assertEqual(fv["prev_snapshot_power"], 200.0)
        self.assertEqual(fv["prev_snapshot_lights"], 5)

    def test_build_training_data(self):
        import importlib.util
        import pathlib

        _spec = importlib.util.spec_from_file_location(
            "engine_conftest",
            pathlib.Path(__file__).parent / "conftest.py",
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        make_synthetic_snapshots = _mod.make_synthetic_snapshots

        config = DEFAULT_FEATURE_CONFIG
        snapshots = make_synthetic_snapshots(10)

        names, X, targets = build_training_data(snapshots, config)
        self.assertEqual(len(X), 10)
        self.assertEqual(len(X[0]), len(names))
        self.assertEqual(len(targets["power_watts"]), 10)

    def test_extract_target_values(self):
        snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
        snapshot["power"]["total_watts"] = 156.5
        snapshot["lights"]["on"] = 8
        tv = extract_target_values(snapshot)
        self.assertEqual(tv["power_watts"], 156.5)
        self.assertEqual(tv["lights_on"], 8)

    def test_interaction_features_when_enabled(self):
        config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        config["interaction_features"]["is_weekend_x_temp"] = True

        snapshot = build_empty_snapshot("2026-02-14", HolidayConfig())  # Saturday
        snapshot["time_features"] = build_time_features("2026-02-14T12:00:00", None, "2026-02-14")
        snapshot["weather"] = {"temp_f": 80, "humidity_pct": 50, "wind_mph": 5}
        snapshot["media"] = {"total_active": 0}
        snapshot["motion"] = {"active_count": 0}
        snapshot["ev"] = {}

        fv = build_feature_vector(snapshot, config)
        self.assertIn("is_weekend_x_temp", fv)
        self.assertEqual(fv["is_weekend_x_temp"], 80)


class TestPresenceFeatures(unittest.TestCase):
    def _make_snapshot_with_presence(self):
        """Helper to create a snapshot with presence data."""
        snap = {
            "date": "2026-02-16",
            "time_features": {
                "hour_sin": 0.5,
                "hour_cos": 0.87,
                "dow_sin": 0.0,
                "dow_cos": 1.0,
                "month_sin": 0.5,
                "month_cos": 0.87,
                "day_of_year_sin": 0.3,
                "day_of_year_cos": 0.95,
                "is_weekend": False,
                "is_holiday": False,
                "is_night": False,
                "is_work_hours": True,
                "minutes_since_sunrise": 200,
                "minutes_until_sunset": 400,
                "daylight_remaining_pct": 0.67,
            },
            "weather": {"temp_f": 55, "humidity_pct": 60, "wind_mph": 5},
            "occupancy": {"people_home": ["Justin"], "device_count_home": 10, "people_home_count": 1},
            "lights": {"on": 3, "total_brightness": 500},
            "motion": {"active_count": 1},
            "media": {"total_active": 0},
            "ev": {"TARS": {"battery_pct": 75, "is_charging": False}},
            "entities": {"total": 3050, "unavailable": 50},
            "power": {"total_watts": 200.0},
            "presence": {
                "overall_probability": 0.92,
                "occupied_room_count": 3,
                "identified_person_count": 2,
                "camera_signal_count": 5,
            },
        }
        return snap

    def test_presence_features_extracted(self):
        """Feature vector includes presence data when available."""
        snap = self._make_snapshot_with_presence()
        features = build_feature_vector(snap)
        self.assertEqual(features["presence_probability"], 0.92)
        self.assertEqual(features["presence_occupied_rooms"], 3)
        self.assertEqual(features["presence_identified_persons"], 2)
        self.assertEqual(features["presence_camera_signals"], 5)

    def test_presence_features_default_zero(self):
        """Feature vector defaults presence to 0 when no presence data."""
        snap = self._make_snapshot_with_presence()
        del snap["presence"]
        features = build_feature_vector(snap)
        self.assertEqual(features.get("presence_probability"), 0)
        self.assertEqual(features.get("presence_occupied_rooms"), 0)
        self.assertEqual(features.get("presence_identified_persons"), 0)
        self.assertEqual(features.get("presence_camera_signals"), 0)

    def test_presence_features_in_feature_names(self):
        """Presence feature names appear in get_feature_names()."""
        names = get_feature_names()
        self.assertIn("presence_probability", names)
        self.assertIn("presence_occupied_rooms", names)
        self.assertIn("presence_identified_persons", names)
        self.assertIn("presence_camera_signals", names)

    def test_presence_features_disabled(self):
        """Disabled presence features should not appear in output."""
        config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)
        config["presence_features"]["presence_probability"] = False
        config["presence_features"]["presence_camera_signals"] = False
        snap = self._make_snapshot_with_presence()
        features = build_feature_vector(snap, config)
        self.assertNotIn("presence_probability", features)
        self.assertIn("presence_occupied_rooms", features)
        self.assertNotIn("presence_camera_signals", features)
        self.assertIn("presence_identified_persons", features)

    def test_presence_in_training_data(self):
        """build_training_data includes presence features in matrix."""
        snap = self._make_snapshot_with_presence()
        names, X, targets = build_training_data([snap])
        self.assertIn("presence_probability", names)
        idx = names.index("presence_probability")
        self.assertEqual(X[0][idx], 0.92)


class TestMRMRFeatureSelection(unittest.TestCase):
    def test_selects_fewer_features_than_input(self):
        """48 features, max_features=10 → exactly 10 returned."""
        import numpy as np

        from aria.engine.features.feature_selection import mrmr_select

        rng = np.random.RandomState(42)
        n_samples, n_features = 100, 48
        X = rng.randn(n_samples, n_features)
        y = X[:, 0] + 0.5 * X[:, 1] + rng.randn(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(n_features)]

        selected = mrmr_select(X, y, names, max_features=10)
        self.assertEqual(len(selected), 10)
        # All returned names must be from the original set
        for name in selected:
            self.assertIn(name, names)

    def test_selects_relevant_features(self):
        """Signal features (feat_0 weight=5, feat_1 weight=3) should appear in top 5."""
        import numpy as np

        from aria.engine.features.feature_selection import mrmr_select

        rng = np.random.RandomState(123)
        n_samples, n_features = 200, 20
        X = rng.randn(n_samples, n_features)
        y = 5.0 * X[:, 0] + 3.0 * X[:, 1] + rng.randn(n_samples) * 0.1
        names = [f"feat_{i}" for i in range(n_features)]

        selected = mrmr_select(X, y, names, max_features=5)
        self.assertIn("feat_0", selected)
        self.assertIn("feat_1", selected)

    def test_handles_fewer_features_than_max(self):
        """5 features with max_features=10 → returns all 5."""
        import numpy as np

        from aria.engine.features.feature_selection import mrmr_select

        rng = np.random.RandomState(99)
        n_samples, n_features = 50, 5
        X = rng.randn(n_samples, n_features)
        y = X[:, 0] + rng.randn(n_samples) * 0.5
        names = [f"feat_{i}" for i in range(n_features)]

        selected = mrmr_select(X, y, names, max_features=10)
        self.assertEqual(len(selected), 5)
        self.assertEqual(set(selected), set(names))


if __name__ == "__main__":
    unittest.main()
