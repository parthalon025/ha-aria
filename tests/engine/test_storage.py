"""Tests for storage: intraday save/load, aggregation."""

import os
import unittest
import tempfile
import shutil
from pathlib import Path

from aria.engine.config import HolidayConfig, PathConfig
from aria.engine.collectors.snapshot import (
    build_empty_snapshot, aggregate_intraday_to_daily,
)
from aria.engine.storage.data_store import DataStore


class TestIntradaySnapshot(unittest.TestCase):
    def test_save_and_load_intraday(self):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=Path(tmpdir))
            paths.ensure_dirs()
            store = DataStore(paths)

            snapshot = build_empty_snapshot("2026-02-10", HolidayConfig())
            snapshot["hour"] = 16
            snapshot["timestamp"] = "2026-02-10T16:00:00"
            snapshot["power"]["total_watts"] = 245.3

            path = store.save_intraday_snapshot(snapshot)
            self.assertTrue(os.path.isfile(path))
            self.assertIn("16.json", str(path))

            loaded = store.load_intraday_snapshots("2026-02-10")
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(loaded[0]["power"]["total_watts"], 245.3)
        finally:
            shutil.rmtree(tmpdir)


class TestIntradayAggregation(unittest.TestCase):
    def test_aggregate_intraday_curves(self):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=Path(tmpdir))
            paths.ensure_dirs()
            store = DataStore(paths)

            for hour, power, lights, people in [(0, 80, 0, 2), (8, 120, 3, 1), (16, 250, 8, 2), (20, 180, 12, 2)]:
                snap = build_empty_snapshot("2026-02-10", HolidayConfig())
                snap["hour"] = hour
                snap["power"]["total_watts"] = power
                snap["lights"]["on"] = lights
                snap["occupancy"]["people_home_count"] = people
                snap["motion"] = {"active_count": hour // 4}
                snap["batteries"] = {"lock.test": {"level": 100 - hour}}
                snap["ev"] = {"TARS": {"range_miles": 200 - hour}}
                store.save_intraday_snapshot(snap)

            result = aggregate_intraday_to_daily("2026-02-10", store)
            self.assertIsNotNone(result)

            # Curves
            self.assertEqual(result["intraday_curves"]["power_curve"], [80, 120, 250, 180])
            self.assertEqual(result["intraday_curves"]["lights_curve"], [0, 3, 8, 12])

            # Aggregates
            self.assertEqual(result["daily_aggregates"]["power_max"], 250)
            self.assertEqual(result["daily_aggregates"]["power_min"], 80)
            self.assertEqual(result["daily_aggregates"]["lights_max"], 12)
            self.assertEqual(result["daily_aggregates"]["peak_power_hour"], 16)

            # Battery drain
            self.assertIn("lock.test", result["batteries_snapshot"])
            self.assertEqual(result["batteries_snapshot"]["lock.test"]["level"], 80)
        finally:
            shutil.rmtree(tmpdir)

    def test_aggregate_no_data_returns_none(self):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=Path(tmpdir))
            paths.ensure_dirs()
            store = DataStore(paths)
            result = aggregate_intraday_to_daily("2099-01-01", store)
            self.assertIsNone(result)
        finally:
            shutil.rmtree(tmpdir)


class TestSequenceModelStorage:
    """Pytest-style tests for sequence model + anomaly persistence."""

    def test_save_load_sequence_model_roundtrip(self, store):
        model_data = {
            "transitions": {
                "light.office:on": {"light.atrium:on": 0.6, "lock.back_door:locked": 0.4},
                "lock.back_door:locked": {"light.office:off": 1.0},
            },
            "entity_counts": {"light.office:on": 10, "light.atrium:on": 6},
            "trained_at": "2026-02-12T16:00:00",
            "event_count": 500,
        }
        store.save_sequence_model(model_data)
        loaded = store.load_sequence_model()
        assert loaded == model_data

    def test_load_sequence_model_returns_none_when_missing(self, store):
        assert store.load_sequence_model() is None

    def test_save_load_sequence_anomalies_roundtrip(self, store):
        summary = {
            "anomalies": [
                {"sequence": ["light.office:on", "lock.back_door:unlocked"],
                 "probability": 0.02, "expected": "lock.back_door:locked"},
            ],
            "total_sequences": 150,
            "anomaly_count": 1,
            "detected_at": "2026-02-12T16:30:00",
        }
        store.save_sequence_anomalies(summary)
        loaded = store.load_sequence_anomalies()
        assert loaded == summary

    def test_load_sequence_anomalies_returns_none_when_missing(self, store):
        assert store.load_sequence_anomalies() is None


if __name__ == "__main__":
    unittest.main()
