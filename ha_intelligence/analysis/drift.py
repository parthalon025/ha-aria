"""Concept drift detection — monitor prediction error and trigger retraining.

Tracks rolling MAE over a configurable window. Fires retrain signal when
current error exceeds threshold (default: 2x rolling median).
"""

import statistics
from datetime import datetime, timedelta


class DriftDetector:
    """Monitor rolling prediction error and decide when to retrain.

    Args:
        window_days: Number of days in the rolling window (default 7).
        threshold_multiplier: Retrain when MAE > multiplier * median_MAE (default 2.0).
        min_samples: Minimum scored days before drift detection is meaningful.
    """

    def __init__(self, window_days: int = 7, threshold_multiplier: float = 2.0,
                 min_samples: int = 5):
        self.window_days = window_days
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples

    def check(self, accuracy_history: dict) -> dict:
        """Analyze accuracy history and determine if retraining is needed.

        Args:
            accuracy_history: Dict with "scores" list, each entry having
                "date", "overall", and "metrics" (with per-metric error/accuracy).

        Returns:
            Dict with:
                - needs_retrain: bool
                - reason: str (human-readable explanation)
                - rolling_mae: dict of per-metric rolling MAE
                - current_mae: dict of per-metric latest MAE
                - threshold: dict of per-metric retrain thresholds
                - days_analyzed: int
        """
        scores = accuracy_history.get("scores", [])
        if len(scores) < self.min_samples:
            return {
                "needs_retrain": False,
                "reason": f"insufficient data ({len(scores)} days, need {self.min_samples})",
                "days_analyzed": len(scores),
            }

        # Use the most recent window_days entries
        window = scores[-self.window_days:]
        latest = scores[-1]

        # Extract per-metric absolute errors from the window
        metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
        rolling_mae = {}
        current_mae = {}
        thresholds = {}
        drift_detected = []

        for metric in metrics:
            errors = []
            for entry in window:
                metric_data = entry.get("metrics", {}).get(metric, {})
                error = metric_data.get("error")
                if error is not None:
                    errors.append(abs(error))

            if len(errors) < 3:
                continue

            median_error = statistics.median(errors)
            rolling_mae[metric] = round(median_error, 2)

            # Current error is the latest entry's error
            latest_error = latest.get("metrics", {}).get(metric, {}).get("error")
            if latest_error is not None:
                current_mae[metric] = round(abs(latest_error), 2)
                threshold = round(median_error * self.threshold_multiplier, 2)
                thresholds[metric] = threshold

                if abs(latest_error) > threshold and median_error > 0:
                    drift_detected.append(metric)

        if drift_detected:
            return {
                "needs_retrain": True,
                "reason": f"drift detected in {', '.join(drift_detected)} "
                          f"(error > {self.threshold_multiplier}x rolling median)",
                "drifted_metrics": drift_detected,
                "rolling_mae": rolling_mae,
                "current_mae": current_mae,
                "threshold": thresholds,
                "days_analyzed": len(window),
            }

        # Also check overall accuracy degradation
        if len(window) >= 3:
            recent_overall = [s["overall"] for s in window[-3:]]
            earlier_overall = [s["overall"] for s in window[:-3]] if len(window) > 3 else recent_overall
            if earlier_overall and statistics.mean(recent_overall) < statistics.mean(earlier_overall) - 10:
                return {
                    "needs_retrain": True,
                    "reason": f"overall accuracy dropped >10% "
                              f"({statistics.mean(recent_overall):.0f}% vs "
                              f"{statistics.mean(earlier_overall):.0f}%)",
                    "rolling_mae": rolling_mae,
                    "current_mae": current_mae,
                    "threshold": thresholds,
                    "days_analyzed": len(window),
                }

        return {
            "needs_retrain": False,
            "reason": "no drift detected, error within normal range",
            "rolling_mae": rolling_mae,
            "current_mae": current_mae,
            "threshold": thresholds,
            "days_analyzed": len(window),
        }

    def should_skip_scheduled_retrain(self, accuracy_history: dict) -> bool:
        """Return True if scheduled retrain can be skipped (error is stable/low).

        Use this to replace fixed Monday retrains — if no drift detected and
        accuracy is above 75%, skip the retrain to save compute.
        """
        result = self.check(accuracy_history)
        if result.get("needs_retrain"):
            return False

        scores = accuracy_history.get("scores", [])
        if len(scores) < self.min_samples:
            return False  # Not enough data to skip confidently

        recent = [s["overall"] for s in scores[-3:]]
        return statistics.mean(recent) >= 75
