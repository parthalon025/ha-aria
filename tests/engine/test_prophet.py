"""Tests for Prophet seasonal forecaster."""

import os
import unittest
from datetime import datetime, timedelta

try:
    import prophet  # noqa: F401

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


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


class TestProphetMetricExtraction(unittest.TestCase):
    """Test metric extraction from snapshot dicts."""

    def test_extract_power_watts(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        snap = {"power": {"total_watts": 250.5}}
        result = ProphetForecaster._extract_metric(snap, "power_watts")
        self.assertEqual(result, 250.5)

    def test_extract_lights_on(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        snap = {"lights": {"on": 5}}
        result = ProphetForecaster._extract_metric(snap, "lights_on")
        self.assertEqual(result, 5.0)

    def test_extract_missing_key(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        snap = {"power": {}}
        result = ProphetForecaster._extract_metric(snap, "power_watts")
        self.assertIsNone(result)

    def test_extract_unknown_metric(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        snap = {"power": {"total_watts": 100}}
        result = ProphetForecaster._extract_metric(snap, "not_a_metric")
        self.assertIsNone(result)


@unittest.skipUnless(HAS_PROPHET, "prophet not installed")
class TestProphetForecaster(unittest.TestCase):
    """Test Prophet training and prediction with real Prophet."""

    def test_train_insufficient_data(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        forecaster = ProphetForecaster()
        # Only 5 days — below 14-day minimum
        snapshots = _make_daily_snapshots(n_days=5)
        result = forecaster.train("power_watts", snapshots, "/tmp/test-prophet")
        self.assertIn("error", result)
        self.assertIn("insufficient", result["error"])

    def test_train_produces_results(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        forecaster = ProphetForecaster()
        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-prophet-train"
        os.makedirs(model_dir, exist_ok=True)

        result = forecaster.train("power_watts", snapshots, model_dir)
        self.assertNotIn("error", result)
        self.assertEqual(result["metric"], "power_watts")
        self.assertIn("mae", result)
        self.assertIn("mape", result)
        self.assertEqual(result["training_days"], 30)
        self.assertIn("components", result)

        # Model file should exist
        self.assertTrue(os.path.isfile(os.path.join(model_dir, "prophet_power_watts.pkl")))

    def test_predict_returns_forecast(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        forecaster = ProphetForecaster()
        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-prophet-predict"
        os.makedirs(model_dir, exist_ok=True)

        # Train first
        forecaster.train("power_watts", snapshots, model_dir)

        # Predict
        result = forecaster.predict("power_watts", model_dir, horizon_days=1)
        self.assertIsNotNone(result)
        self.assertIn("predicted", result)
        self.assertIn("lower", result)
        self.assertIn("upper", result)
        self.assertIn("trend", result)
        # Prediction should be in reasonable range
        self.assertGreater(result["predicted"], 0)
        self.assertLess(result["predicted"], 1000)

    def test_predict_missing_model_returns_none(self):
        from aria.engine.models.prophet_forecaster import ProphetForecaster

        forecaster = ProphetForecaster()
        result = forecaster.predict("power_watts", "/tmp/nonexistent-dir")
        self.assertIsNone(result)


@unittest.skipUnless(HAS_PROPHET, "prophet not installed")
class TestProphetConvenienceFunctions(unittest.TestCase):
    """Test the module-level convenience functions."""

    def test_train_prophet_models(self):
        from aria.engine.models.prophet_forecaster import train_prophet_models

        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-prophet-all"
        os.makedirs(model_dir, exist_ok=True)

        results = train_prophet_models(snapshots, model_dir)
        self.assertNotIn("error", results)
        # Should have results for each metric
        self.assertIn("power_watts", results)
        self.assertIn("lights_on", results)

    def test_predict_with_prophet(self):
        from aria.engine.models.prophet_forecaster import (
            predict_with_prophet,
            train_prophet_models,
        )

        snapshots = _make_daily_snapshots(n_days=30)
        model_dir = "/tmp/test-prophet-roundtrip"
        os.makedirs(model_dir, exist_ok=True)

        # Train
        train_prophet_models(snapshots, model_dir)

        # Predict
        predictions = predict_with_prophet(model_dir)
        self.assertIsInstance(predictions, dict)
        self.assertIn("power_watts", predictions)
        self.assertIsInstance(predictions["power_watts"], float)


class TestProphetRegistered(unittest.TestCase):
    """Verify Prophet is registered in the ModelRegistry."""

    def test_prophet_in_registry(self):
        # Import to trigger registration
        import aria.engine.models.prophet_forecaster  # noqa: F401
        from aria.engine.models.registry import ModelRegistry

        self.assertIn("prophet", ModelRegistry.available())


if __name__ == "__main__":
    unittest.main()
