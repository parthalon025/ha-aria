"""Concept drift detection — monitor prediction error and trigger retraining.

Provides two detection strategies:
1. DriftDetector: Fixed-window rolling MAE with threshold (original)
2. PageHinkleyDetector: Adaptive sequential change detection (research-backed)

The Page-Hinkley test is a sequential analysis technique for detecting changes
in the mean of a signal. Unlike fixed-window approaches, it adapts to the data
and detects both gradual and sudden shifts with theoretical guarantees.

Reference: Page (1954), Hinkley (1971). Validated for IoT concept drift in
"Lightweight Concept Drift for IoT" (IEEE IoT Magazine 2021).
"""

import statistics


class PageHinkleyDetector:
    """Page-Hinkley test for adaptive drift detection.

    Monitors a cumulative sum of deviations from a running mean.
    Signals drift when the cumulative sum exceeds a threshold (lambda_).

    This is a pure-Python implementation — no external dependencies.

    Args:
        delta_: Minimum magnitude of change to detect. Higher values
            reduce false positives but miss smaller drifts. Default 0.005.
        lambda_: Detection threshold. Higher = fewer false alarms,
            slower detection. Default 50.
        alpha_: Forgetting factor for running mean (0-1). Lower values
            give more weight to recent observations. Default 0.9999.
    """

    def __init__(self, delta_: float = 0.005, lambda_: float = 50.0,
                 alpha_: float = 0.9999):
        self.delta_ = delta_
        self.lambda_ = lambda_
        self.alpha_ = alpha_
        self.reset()

    def reset(self):
        """Reset detector state."""
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._cumulative_sum = 0.0
        self._min_cumulative_sum = 0.0

    def update(self, value: float) -> bool:
        """Feed a new observation and return whether drift is detected.

        Args:
            value: New error/metric observation.

        Returns:
            True if drift is detected at this step.
        """
        self._n += 1

        # Update running mean with forgetting factor
        if self._n == 1:
            self._mean = value
        else:
            self._mean = self.alpha_ * self._mean + (1 - self.alpha_) * value

        # Update cumulative sum
        self._sum += value - self._mean - self.delta_
        self._cumulative_sum = self._sum

        # Track minimum cumulative sum
        if self._cumulative_sum < self._min_cumulative_sum:
            self._min_cumulative_sum = self._cumulative_sum

        # Drift detected when gap between cumulative sum and its minimum exceeds threshold
        return (self._cumulative_sum - self._min_cumulative_sum) > self.lambda_

    @property
    def drift_score(self) -> float:
        """Current gap between cumulative sum and minimum (0 = no drift signal)."""
        return max(0.0, self._cumulative_sum - self._min_cumulative_sum)

    def check_series(self, values: list) -> dict:
        """Run Page-Hinkley test over a series of values.

        Args:
            values: List of error/metric observations.

        Returns:
            Dict with drift_detected, drift_point (index), drift_score.
        """
        self.reset()
        for i, v in enumerate(values):
            if self.update(v):
                return {
                    "drift_detected": True,
                    "drift_point": i,
                    "drift_score": round(self.drift_score, 4),
                    "observations": len(values),
                }
        return {
            "drift_detected": False,
            "drift_point": None,
            "drift_score": round(self.drift_score, 4),
            "observations": len(values),
        }


class DriftDetector:
    """Monitor rolling prediction error and decide when to retrain.

    Uses both fixed-window median threshold (original) and Page-Hinkley
    adaptive detection. Either detector triggering counts as drift.

    Args:
        window_days: Number of days in the rolling window (default 7).
        threshold_multiplier: Retrain when MAE > multiplier * median_MAE (default 2.0).
        min_samples: Minimum scored days before drift detection is meaningful.
        use_page_hinkley: Whether to also run Page-Hinkley detection (default True).
    """

    def __init__(self, window_days: int = 7, threshold_multiplier: float = 2.0,
                 min_samples: int = 5, use_page_hinkley: bool = True):
        self.window_days = window_days
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples
        self.use_page_hinkley = use_page_hinkley

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
                - page_hinkley: dict of per-metric PH results (if enabled)
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
        ph_results = {}

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

            # Run Page-Hinkley on the full error series (not just window)
            if self.use_page_hinkley and len(errors) >= self.min_samples:
                ph = PageHinkleyDetector()
                ph_result = ph.check_series(errors)
                ph_results[metric] = ph_result
                if ph_result["drift_detected"] and metric not in drift_detected:
                    drift_detected.append(metric)

        if drift_detected:
            method = "threshold + page-hinkley" if self.use_page_hinkley else "threshold"
            return {
                "needs_retrain": True,
                "reason": f"drift detected in {', '.join(drift_detected)} "
                          f"(method: {method})",
                "drifted_metrics": drift_detected,
                "rolling_mae": rolling_mae,
                "current_mae": current_mae,
                "threshold": thresholds,
                "days_analyzed": len(window),
                "page_hinkley": ph_results if self.use_page_hinkley else {},
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
                    "page_hinkley": ph_results if self.use_page_hinkley else {},
                }

        return {
            "needs_retrain": False,
            "reason": "no drift detected, error within normal range",
            "rolling_mae": rolling_mae,
            "current_mae": current_mae,
            "threshold": thresholds,
            "days_analyzed": len(window),
            "page_hinkley": ph_results if self.use_page_hinkley else {},
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
