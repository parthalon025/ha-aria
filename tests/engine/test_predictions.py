"""Tests for predictions: generation, scoring, ML blending."""

import shutil
import tempfile
import unittest
from pathlib import Path

from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.config import HolidayConfig, PathConfig
from aria.engine.predictions.predictor import generate_predictions
from aria.engine.predictions.scoring import (
    accuracy_trend,
    score_all_predictions,
    score_prediction,
)


class TestPredictions(unittest.TestCase):
    def test_predict_uses_baseline_mean(self):
        baselines = {
            "Wednesday": {
                "sample_count": 4,
                "power_watts": {"mean": 160, "stddev": 15},
                "lights_on": {"mean": 35, "stddev": 5},
                "devices_home": {"mean": 55, "stddev": 8},
                "unavailable": {"mean": 905, "stddev": 10},
                "useful_events": {"mean": 2400, "stddev": 200},
            }
        }
        predictions = generate_predictions("2026-02-11", baselines, correlations=[], weather_forecast=None)
        self.assertEqual(predictions["target_date"], "2026-02-11")
        self.assertAlmostEqual(predictions["power_watts"]["predicted"], 160, delta=20)
        self.assertIn("confidence", predictions["power_watts"])

    def test_predict_adjusts_for_weather(self):
        baselines = {
            "Wednesday": {
                "sample_count": 4,
                "power_watts": {"mean": 160, "stddev": 15},
                "lights_on": {"mean": 35, "stddev": 5},
                "devices_home": {"mean": 55, "stddev": 8},
                "unavailable": {"mean": 905, "stddev": 10},
                "useful_events": {"mean": 2400, "stddev": 200},
            }
        }
        hot_corrs = [{"x": "weather_temp", "y": "power_watts", "r": 0.85}]
        pred_hot = generate_predictions("2026-02-11", baselines, hot_corrs, {"temp_f": 95})
        pred_normal = generate_predictions("2026-02-11", baselines, [], None)

        self.assertGreater(pred_hot["power_watts"]["predicted"], pred_normal["power_watts"]["predicted"])


class TestSelfReinforcement(unittest.TestCase):
    def test_score_prediction_perfect(self):
        prediction = {"power_watts": {"predicted": 150, "baseline_mean": 150, "baseline_stddev": 10}}
        actual = {"power": {"total_watts": 150}}
        score = score_prediction("power_watts", prediction, actual)
        self.assertEqual(score["accuracy"], 100)

    def test_score_prediction_off_by_one_sigma(self):
        prediction = {"power_watts": {"predicted": 150, "baseline_mean": 150, "baseline_stddev": 10}}
        actual = {"power": {"total_watts": 160}}
        score = score_prediction("power_watts", prediction, actual)
        self.assertGreater(score["accuracy"], 50)
        self.assertLess(score["accuracy"], 100)

    def test_accuracy_history_tracks_trend(self):
        history = {
            "scores": [
                {"date": "2026-02-05", "overall": 70},
                {"date": "2026-02-06", "overall": 75},
                {"date": "2026-02-07", "overall": 80},
                {"date": "2026-02-08", "overall": 82},
                {"date": "2026-02-09", "overall": 85},
            ]
        }
        trend = accuracy_trend(history)
        self.assertEqual(trend, "improving")


class TestMLEnhancedPredictions(unittest.TestCase):
    def test_generate_predictions_with_ml(self):
        tmpdir = tempfile.mkdtemp()
        try:
            paths = PathConfig(data_dir=Path(tmpdir))
            paths.ensure_dirs()
            # Create 30 fake daily files
            for d in range(30):
                with open(paths.daily_dir / f"2026-01-{d + 1:02d}.json", "w") as f:
                    f.write("{}")

            baselines = {
                "Wednesday": {
                    "sample_count": 4,
                    "power_watts": {"mean": 160, "stddev": 15},
                    "lights_on": {"mean": 35, "stddev": 5},
                    "devices_home": {"mean": 55, "stddev": 8},
                    "unavailable": {"mean": 905, "stddev": 10},
                    "useful_events": {"mean": 2400, "stddev": 200},
                }
            }
            ml_preds = {"power_watts": 180.0, "lights_on": 40.0}
            predictions = generate_predictions("2026-02-11", baselines, ml_predictions=ml_preds, paths=paths)
            self.assertAlmostEqual(predictions["power_watts"]["predicted"], 166.0, delta=1)
            self.assertEqual(predictions["prediction_method"], "blended")
            self.assertAlmostEqual(predictions["lights_on"]["predicted"], 36.5, delta=1)
        finally:
            shutil.rmtree(tmpdir)

    def test_generate_predictions_without_ml_stays_statistical(self):
        baselines = {
            "Wednesday": {
                "sample_count": 4,
                "power_watts": {"mean": 160, "stddev": 15},
                "lights_on": {"mean": 35, "stddev": 5},
                "devices_home": {"mean": 55, "stddev": 8},
                "unavailable": {"mean": 905, "stddev": 10},
                "useful_events": {"mean": 2400, "stddev": 200},
            }
        }
        predictions = generate_predictions("2026-02-11", baselines)
        self.assertEqual(predictions["prediction_method"], "statistical")
        self.assertAlmostEqual(predictions["power_watts"]["predicted"], 160.0, delta=1)

    def test_generate_predictions_includes_device_failures(self):
        baselines = {
            "Wednesday": {
                "sample_count": 1,
                "power_watts": {"mean": 100, "stddev": 10},
                "lights_on": {"mean": 10, "stddev": 2},
                "devices_home": {"mean": 50, "stddev": 5},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        failures = [{"entity_id": "sensor.flaky", "failure_probability": 0.85, "risk": "high"}]
        predictions = generate_predictions("2026-02-11", baselines, device_failures=failures)
        self.assertEqual(len(predictions["device_failures"]), 1)
        self.assertEqual(predictions["device_failures"][0]["entity_id"], "sensor.flaky")

    def test_generate_predictions_includes_contextual_anomalies(self):
        baselines = {
            "Wednesday": {
                "sample_count": 1,
                "power_watts": {"mean": 100, "stddev": 10},
                "lights_on": {"mean": 10, "stddev": 2},
                "devices_home": {"mean": 50, "stddev": 5},
                "unavailable": {"mean": 900, "stddev": 20},
                "useful_events": {"mean": 2500, "stddev": 300},
            }
        }
        ctx = {"is_anomaly": True, "anomaly_score": -0.35, "severity": "high"}
        predictions = generate_predictions("2026-02-11", baselines, contextual_anomalies=ctx)
        self.assertTrue(predictions["contextual_anomalies"]["is_anomaly"])
        self.assertEqual(predictions["contextual_anomalies"]["severity"], "high")

    def test_score_all_tracks_prediction_method(self):
        predictions = {
            "target_date": "2026-02-11",
            "prediction_method": "blended",
            "days_of_data": 30,
            "power_watts": {"predicted": 150, "baseline_mean": 150, "baseline_stddev": 10},
            "lights_on": {"predicted": 30, "baseline_mean": 30, "baseline_stddev": 5},
            "devices_home": {"predicted": 50, "baseline_mean": 50, "baseline_stddev": 10},
            "unavailable": {"predicted": 900, "baseline_mean": 900, "baseline_stddev": 20},
            "useful_events": {"predicted": 2500, "baseline_mean": 2500, "baseline_stddev": 300},
        }
        actual = build_empty_snapshot("2026-02-11", HolidayConfig())
        actual["power"]["total_watts"] = 155
        actual["lights"]["on"] = 32
        actual["occupancy"]["device_count_home"] = 52
        actual["entities"]["unavailable"] = 905
        actual["logbook_summary"] = {"useful_events": 2400}
        result = score_all_predictions(predictions, actual)
        self.assertEqual(result["prediction_method"], "blended")
        self.assertEqual(result["days_of_data"], 30)
        self.assertGreater(result["overall"], 0)


if __name__ == "__main__":
    unittest.main()
