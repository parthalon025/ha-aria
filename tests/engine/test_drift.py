"""Tests for drift detection: rolling MAE monitoring and retrain triggers."""

import unittest

import pytest

pytest.importorskip("river")

from aria.engine.analysis.drift import ADWINDetector, DriftDetector


class TestDriftDetector(unittest.TestCase):
    def _make_history(self, scores):
        """Build accuracy_history from a list of (date, overall, power_error) tuples."""
        return {
            "scores": [
                {
                    "date": date,
                    "overall": overall,
                    "metrics": {
                        "power_watts": {"error": power_error, "accuracy": overall},
                        "lights_on": {"error": power_error * 0.1, "accuracy": overall},
                    },
                }
                for date, overall, power_error in scores
            ]
        }

    def test_insufficient_data_no_retrain(self):
        detector = DriftDetector(min_samples=5)
        history = self._make_history(
            [
                ("2026-02-10", 80, 10),
                ("2026-02-11", 82, 8),
            ]
        )
        result = detector.check(history)
        self.assertFalse(result["needs_retrain"])
        self.assertIn("insufficient", result["reason"])

    def test_stable_errors_no_retrain(self):
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0)
        # All errors around 10 — no drift
        history = self._make_history([(f"2026-02-{d:02d}", 85, 10 + (d % 3)) for d in range(5, 13)])
        result = detector.check(history)
        self.assertFalse(result["needs_retrain"])
        self.assertIn("no drift", result["reason"])
        self.assertIn("power_watts", result["rolling_mae"])

    def test_spike_triggers_retrain(self):
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0)
        # Stable errors around 10, then a massive spike
        scores = [(f"2026-02-{d:02d}", 85, 10) for d in range(5, 12)]
        scores.append(("2026-02-12", 40, 50))  # Big spike
        history = self._make_history(scores)
        result = detector.check(history)
        self.assertTrue(result["needs_retrain"])
        self.assertIn("drift detected", result["reason"])
        self.assertIn("power_watts", result.get("drifted_metrics", []))

    def test_overall_accuracy_drop_triggers_retrain(self):
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0)
        # Gradual accuracy decline (errors stay within threshold but overall drops)
        scores = [
            ("2026-02-05", 85, 10),
            ("2026-02-06", 83, 11),
            ("2026-02-07", 82, 12),
            ("2026-02-08", 80, 13),
            ("2026-02-09", 60, 14),
            ("2026-02-10", 55, 15),
            ("2026-02-11", 50, 16),
        ]
        history = self._make_history(scores)
        result = detector.check(history)
        self.assertTrue(result["needs_retrain"])
        self.assertIn("accuracy dropped", result["reason"])

    def test_should_skip_retrain_when_stable_and_accurate(self):
        detector = DriftDetector(min_samples=5)
        history = self._make_history([(f"2026-02-{d:02d}", 85, 10) for d in range(5, 13)])
        self.assertTrue(detector.should_skip_scheduled_retrain(history))

    def test_should_not_skip_retrain_when_drift_detected(self):
        detector = DriftDetector(min_samples=5)
        scores = [(f"2026-02-{d:02d}", 85, 10) for d in range(5, 12)]
        scores.append(("2026-02-12", 40, 50))
        history = self._make_history(scores)
        self.assertFalse(detector.should_skip_scheduled_retrain(history))

    def test_should_not_skip_retrain_when_accuracy_low(self):
        detector = DriftDetector(min_samples=5)
        history = self._make_history([(f"2026-02-{d:02d}", 60, 20) for d in range(5, 13)])
        self.assertFalse(detector.should_skip_scheduled_retrain(history))

    def test_custom_threshold_multiplier(self):
        # With very strict threshold (1.5x), smaller spike triggers retrain
        detector = DriftDetector(window_days=7, threshold_multiplier=1.5)
        scores = [(f"2026-02-{d:02d}", 85, 10) for d in range(5, 12)]
        scores.append(("2026-02-12", 70, 20))  # 2x median, triggers at 1.5x
        history = self._make_history(scores)
        result = detector.check(history)
        self.assertTrue(result["needs_retrain"])


class TestADWINDetector(unittest.TestCase):
    def test_detects_sudden_drift(self):
        """Stable period then sudden shift should detect drift."""
        detector = ADWINDetector(delta=0.002)
        stable = [0.05] * 100
        shifted = [0.95] * 50
        result = detector.check_series("error_rate", stable + shifted)
        self.assertTrue(result["drift_detected"])
        self.assertIsNotNone(result["drift_point"])
        self.assertEqual(result["observations"], 150)

    def test_no_false_alarm_on_stable(self):
        """100 identical values should produce zero alarms."""
        detector = ADWINDetector(delta=0.002)
        values = [0.5] * 100
        result = detector.check_series("stable_metric", values)
        self.assertFalse(result["drift_detected"])
        self.assertIsNone(result["drift_point"])

    def test_per_metric_independence(self):
        """Two different metrics should be tracked independently."""
        detector = ADWINDetector(delta=0.002)
        detector.update("metric_a", 0.1)
        detector.update("metric_a", 0.2)
        detector.update("metric_b", 0.9)
        detector.update("metric_b", 0.8)
        stats = detector.get_stats()
        self.assertIn("metric_a", stats)
        self.assertIn("metric_b", stats)
        # Each should have independent state
        self.assertNotEqual(stats["metric_a"], stats["metric_b"])


class TestEnsembleDrift(unittest.TestCase):
    def _make_history(self, scores):
        """Build accuracy_history from a list of (date, overall, power_error) tuples."""
        return {
            "scores": [
                {
                    "date": date,
                    "overall": overall,
                    "metrics": {
                        "power_watts": {"error": power_error, "accuracy": overall},
                        "lights_on": {"error": power_error * 0.1, "accuracy": overall},
                    },
                }
                for date, overall, power_error in scores
            ]
        }

    def test_ensemble_includes_adwin_results(self):
        """DriftDetector with use_adwin=True should include 'adwin' key in result."""
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0, use_adwin=True)
        # Stable errors — enough data points for analysis
        history = self._make_history([(f"2026-02-{d:02d}", 85, 10 + (d % 3)) for d in range(5, 13)])
        result = detector.check(history)
        self.assertIn("adwin", result)
        # ADWIN results should be a dict with per-metric entries
        self.assertIsInstance(result["adwin"], dict)


if __name__ == "__main__":
    unittest.main()
