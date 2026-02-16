"""Tests for NeuralProphet seasonal forecaster."""

import os
import unittest
from datetime import datetime, timedelta

import pytest

try:
    import neuralprophet  # noqa: F401

    HAS_NP = True
except ImportError:
    HAS_NP = False


def _make_daily_snapshots(n_days=30, base_power=200, weekly_pattern=True):
    """Generate synthetic daily snapshots for testing."""
    snapshots = []
    base_date = datetime(2026, 1, 1)
    for i in range(n_days):
        date = base_date + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")

        # Add weekly pattern: higher on weekends
        power = base_power
        if weekly_pattern and date.weekday() >= 5:
            power += 50  # Weekend boost

        snap = {
            "power": {"total_watts": power + (i % 7) * 5},
            "lights": {"on": 3 + (i % 4)},
            "occupancy": {"device_count_home": 10 + (i % 3)},
            "entities": {"unavailable": 2 + (i % 2)},
        }
        snapshots.append((date_str, snap))
    return snapshots


class TestNeuralProphetModule(unittest.TestCase):
    """Test module-level attributes are always importable."""

    def test_has_neural_prophet_flag_exists(self):
        from aria.engine.models.neural_prophet_forecaster import HAS_NEURAL_PROPHET

        self.assertIsInstance(HAS_NEURAL_PROPHET, bool)

    def test_metrics_list(self):
        from aria.engine.models.neural_prophet_forecaster import NEURALPROPHET_METRICS

        self.assertEqual(len(NEURALPROPHET_METRICS), 4)
        self.assertIn("power_watts", NEURALPROPHET_METRICS)

    def test_extract_metric(self):
        from aria.engine.models.neural_prophet_forecaster import NeuralProphetForecaster

        snap = {"power": {"total_watts": 250.5}}
        result = NeuralProphetForecaster._extract_metric(snap, "power_watts")
        self.assertEqual(result, 250.5)

    def test_extract_missing_metric(self):
        from aria.engine.models.neural_prophet_forecaster import NeuralProphetForecaster

        snap = {"power": {}}
        result = NeuralProphetForecaster._extract_metric(snap, "power_watts")
        self.assertIsNone(result)


@pytest.mark.skipif(not HAS_NP, reason="neuralprophet not installed")
class TestNeuralProphetForecaster(unittest.TestCase):
    """Test NeuralProphet training and prediction (requires neuralprophet)."""

    def test_train_produces_model_files(self):
        from aria.engine.models.neural_prophet_forecaster import NeuralProphetForecaster

        forecaster = NeuralProphetForecaster()
        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-neuralprophet-train"
        os.makedirs(model_dir, exist_ok=True)

        result = forecaster.train("power_watts", snapshots, model_dir)
        self.assertNotIn("error", result)
        self.assertEqual(result["metric"], "power_watts")
        self.assertEqual(result["backend"], "neuralprophet")
        self.assertIn("mae", result)
        self.assertIn("mape", result)
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "neuralprophet_power_watts.pkl")))

    def test_predict_returns_forecast(self):
        from aria.engine.models.neural_prophet_forecaster import NeuralProphetForecaster

        forecaster = NeuralProphetForecaster()
        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-neuralprophet-predict"
        os.makedirs(model_dir, exist_ok=True)

        forecaster.train("power_watts", snapshots, model_dir)
        result = forecaster.predict("power_watts", model_dir, horizon_days=1)
        self.assertIsNotNone(result)
        self.assertIn("predicted", result)
        self.assertGreater(result["predicted"], 0)
        self.assertLess(result["predicted"], 1000)

    def test_insufficient_data_returns_error(self):
        from aria.engine.models.neural_prophet_forecaster import NeuralProphetForecaster

        forecaster = NeuralProphetForecaster()
        snapshots = _make_daily_snapshots(n_days=5)
        result = forecaster.train("power_watts", snapshots, "/tmp/test-np-insufficient")
        self.assertIn("error", result)
        self.assertIn("insufficient", result["error"])

    def test_all_four_metrics_trainable(self):
        from aria.engine.models.neural_prophet_forecaster import (
            NEURALPROPHET_METRICS,
            train_neuralprophet_models,
        )

        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-neuralprophet-all"
        os.makedirs(model_dir, exist_ok=True)

        results = train_neuralprophet_models(snapshots, model_dir)
        self.assertNotIn("error", results)
        for metric in NEURALPROPHET_METRICS:
            self.assertIn(metric, results)
            self.assertNotIn("error", results[metric])


if __name__ == "__main__":
    unittest.main()
