"""Tests for ML models: training, anomaly detection, device failure, blending."""

import os
import unittest
import tempfile
import shutil

from ha_intelligence.features.feature_config import DEFAULT_FEATURE_CONFIG
from ha_intelligence.features.vector_builder import build_training_data
from ha_intelligence.models.training import (
    train_continuous_model, blend_predictions, count_days_of_data,
    predict_with_ml,
)
from ha_intelligence.models.isolation_forest import IsolationForestModel
from ha_intelligence.models.device_failure import (
    train_device_failure_model, predict_device_failures,
    detect_contextual_anomalies,
)
from ha_intelligence.config import HolidayConfig, PathConfig
from ha_intelligence.collectors.snapshot import build_empty_snapshot
from ha_intelligence.storage.data_store import DataStore

from conftest import make_synthetic_snapshots


HAS_SKLEARN = True
try:
    import numpy as np
except ImportError:
    HAS_SKLEARN = False


class TestSklearnTraining(unittest.TestCase):
    def test_train_continuous_model_synthetic(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = make_synthetic_snapshots(120)
            config = DEFAULT_FEATURE_CONFIG
            names, X, targets = build_training_data(snapshots, config)
            result = train_continuous_model(
                "power_watts", names, X, targets["power_watts"], tmpdir)
            self.assertNotIn("error", result)
            self.assertIn("r2", result)
            self.assertIn("mae", result)
            self.assertGreater(result["samples_train"], 0)
            self.assertGreater(result["samples_val"], 0)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "power_watts.pkl")))
            self.assertIn("feature_importance", result)
        finally:
            shutil.rmtree(tmpdir)

    def test_train_continuous_model_insufficient_data(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        result = train_continuous_model("test", ["a", "b"], [[1, 2]] * 10, [1] * 10, "/tmp")
        self.assertIn("error", result)
        self.assertIn("insufficient", result["error"])

    def test_train_anomaly_detector(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = make_synthetic_snapshots(120)
            config = DEFAULT_FEATURE_CONFIG
            names, X, _ = build_training_data(snapshots, config)
            iso_model = IsolationForestModel()
            result = iso_model.train(names, X, tmpdir)
            self.assertNotIn("error", result)
            self.assertEqual(result["samples"], 120)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "anomaly_detector.pkl")))
        finally:
            shutil.rmtree(tmpdir)

    def test_detect_contextual_anomalies(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = make_synthetic_snapshots(120)
            config = DEFAULT_FEATURE_CONFIG
            names, X, _ = build_training_data(snapshots, config)
            iso_model = IsolationForestModel()
            iso_model.train(names, X, tmpdir)
            result = detect_contextual_anomalies(X[25], tmpdir)
            self.assertIn("is_anomaly", result)
            self.assertIn("anomaly_score", result)
        finally:
            shutil.rmtree(tmpdir)

    def test_train_device_failure_model(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = []
            for i in range(30):
                snap = build_empty_snapshot(
                    f"2026-02-{(i % 28) + 1:02d}", HolidayConfig())
                snap["batteries"] = {"sensor.flaky": {"level": 30}}
                if i % 5 == 0:
                    snap["entities"]["unavailable_list"] = ["sensor.flaky"]
                else:
                    snap["entities"]["unavailable_list"] = []
                snapshots.append(snap)
            result = train_device_failure_model(snapshots, tmpdir)
            self.assertNotIn("error", result)
            self.assertGreater(result["samples"], 0)
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "device_failure.pkl")))
        finally:
            shutil.rmtree(tmpdir)

    def test_predict_device_failures(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = []
            for i in range(30):
                snap = build_empty_snapshot(
                    f"2026-02-{(i % 28) + 1:02d}", HolidayConfig())
                snap["batteries"] = {"sensor.flaky": {"level": 10}}
                if i % 3 == 0:
                    snap["entities"]["unavailable_list"] = ["sensor.flaky"]
                else:
                    snap["entities"]["unavailable_list"] = []
                snapshots.append(snap)
            train_device_failure_model(snapshots, tmpdir)
            preds = predict_device_failures(snapshots, tmpdir)
            self.assertIsInstance(preds, list)
        finally:
            shutil.rmtree(tmpdir)

    def test_predict_with_ml(self):
        if not HAS_SKLEARN:
            self.skipTest("sklearn not installed")
        tmpdir = tempfile.mkdtemp()
        try:
            snapshots = make_synthetic_snapshots(120)
            config = DEFAULT_FEATURE_CONFIG
            names, X, targets = build_training_data(snapshots, config)
            train_continuous_model(
                "power_watts", names, X, targets["power_watts"], tmpdir)

            test_snap = snapshots[-1]
            preds = predict_with_ml(test_snap, config, models_dir=tmpdir)
            self.assertIn("power_watts", preds)
            self.assertIsInstance(preds["power_watts"], float)
        finally:
            shutil.rmtree(tmpdir)

    def test_blend_predictions(self):
        # Under 14 days: pure statistical
        result = blend_predictions(100, 200, 10)
        self.assertEqual(result, 100)
        # At 14 days: 70/30 blend
        result = blend_predictions(100, 200, 30)
        self.assertAlmostEqual(result, 130.0)
        # At 60 days: 50/50
        result = blend_predictions(100, 200, 75)
        self.assertAlmostEqual(result, 150.0)
        # At 90+ days: 30/70
        result = blend_predictions(100, 200, 120)
        self.assertAlmostEqual(result, 170.0)

    def test_count_days_of_data(self, tmp_path=None):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=__import__("pathlib").Path(tmpdir))
            paths.ensure_dirs()
            for d in ["2026-02-05.json", "2026-02-06.json", "2026-02-07.json"]:
                with open(paths.daily_dir / d, "w") as f:
                    f.write("{}")
            self.assertEqual(count_days_of_data(paths), 3)
        finally:
            shutil.rmtree(tmpdir)

    def test_blend_predictions_none_ml(self):
        result = blend_predictions(150, None, 60)
        self.assertEqual(result, 150)


if __name__ == "__main__":
    unittest.main()
