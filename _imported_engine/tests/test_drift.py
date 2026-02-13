"""Tests for drift detection: rolling MAE monitoring and retrain triggers."""

import unittest

from ha_intelligence.analysis.drift import DriftDetector


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
        history = self._make_history([
            ("2026-02-10", 80, 10),
            ("2026-02-11", 82, 8),
        ])
        result = detector.check(history)
        self.assertFalse(result["needs_retrain"])
        self.assertIn("insufficient", result["reason"])

    def test_stable_errors_no_retrain(self):
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0)
        # All errors around 10 — no drift
        history = self._make_history([
            (f"2026-02-{d:02d}", 85, 10 + (d % 3))
            for d in range(5, 13)
        ])
        result = detector.check(history)
        self.assertFalse(result["needs_retrain"])
        self.assertIn("no drift", result["reason"])
        self.assertIn("power_watts", result["rolling_mae"])

    def test_spike_triggers_retrain(self):
        detector = DriftDetector(window_days=7, threshold_multiplier=2.0)
        # Stable errors around 10, then a massive spike
        scores = [
            (f"2026-02-{d:02d}", 85, 10)
            for d in range(5, 12)
        ]
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
        history = self._make_history([
            (f"2026-02-{d:02d}", 85, 10)
            for d in range(5, 13)
        ])
        self.assertTrue(detector.should_skip_scheduled_retrain(history))

    def test_should_not_skip_retrain_when_drift_detected(self):
        detector = DriftDetector(min_samples=5)
        scores = [
            (f"2026-02-{d:02d}", 85, 10)
            for d in range(5, 12)
        ]
        scores.append(("2026-02-12", 40, 50))
        history = self._make_history(scores)
        self.assertFalse(detector.should_skip_scheduled_retrain(history))

    def test_should_not_skip_retrain_when_accuracy_low(self):
        detector = DriftDetector(min_samples=5)
        history = self._make_history([
            (f"2026-02-{d:02d}", 60, 20)
            for d in range(5, 13)
        ])
        self.assertFalse(detector.should_skip_scheduled_retrain(history))

    def test_custom_threshold_multiplier(self):
        # With very strict threshold (1.5x), smaller spike triggers retrain
        detector = DriftDetector(window_days=7, threshold_multiplier=1.5)
        scores = [
            (f"2026-02-{d:02d}", 85, 10)
            for d in range(5, 12)
        ]
        scores.append(("2026-02-12", 70, 20))  # 2x median, triggers at 1.5x
        history = self._make_history(scores)
        result = detector.check(history)
        self.assertTrue(result["needs_retrain"])


if __name__ == "__main__":
    unittest.main()
