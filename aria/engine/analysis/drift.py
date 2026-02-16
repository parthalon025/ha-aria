"""Concept drift detection — monitor prediction error and trigger retraining.

Provides three detection strategies:
1. DriftDetector: Fixed-window rolling MAE with threshold (original)
2. PageHinkleyDetector: Adaptive sequential change detection (research-backed)
3. ADWINDetector: Adaptive windowing (ADWIN) from river.drift (research-backed)

The Page-Hinkley test is a sequential analysis technique for detecting changes
in the mean of a signal. Unlike fixed-window approaches, it adapts to the data
and detects both gradual and sudden shifts with theoretical guarantees.

ADWIN (ADaptive WINdowing) maintains a variable-length window of recent items
and detects distribution changes by comparing sub-windows. It automatically
adjusts window size — shrinking when change is detected, growing during
stable periods. Complements Page-Hinkley by using a fundamentally different
detection mechanism (statistical sub-window comparison vs cumulative sum).

References:
- Page (1954), Hinkley (1971). Validated for IoT concept drift in
  "Lightweight Concept Drift for IoT" (IEEE IoT Magazine 2021).
- Bifet & Gavalda (2007). "Learning from Time-Changing Data with Adaptive
  Windowing." SIAM International Conference on Data Mining.
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

    def __init__(self, delta_: float = 0.005, lambda_: float = 50.0, alpha_: float = 0.9999):
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
        self._min_cumulative_sum = min(self._min_cumulative_sum, self._cumulative_sum)

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


try:
    from river.drift import ADWIN as _RiverADWIN

    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class ADWINDetector:
    """ADWIN (Adaptive Windowing) drift detector using river.drift.

    Maintains per-metric ADWIN instances that automatically adjust their
    window size based on observed distribution changes. Detects both
    gradual and sudden drift with low false-positive rates.

    Requires the `river` package. Falls back gracefully (no detection)
    if river is not installed.

    Args:
        delta: Confidence parameter for ADWIN. Lower values = fewer
            false positives but slower detection. Default 0.002.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._detectors: dict = {}

    def _get_detector(self, metric_name: str):
        """Get or create an ADWIN detector for the given metric."""
        if metric_name not in self._detectors:
            if not HAS_RIVER:
                return None
            self._detectors[metric_name] = _RiverADWIN(delta=self.delta)
        return self._detectors[metric_name]

    def update(self, metric_name: str, value: float) -> bool:
        """Feed a new observation for a metric and return whether drift is detected.

        Args:
            metric_name: Name of the metric being tracked.
            value: New error/metric observation.

        Returns:
            True if drift is detected at this step.
        """
        detector = self._get_detector(metric_name)
        if detector is None:
            return False
        detector.update(value)
        return detector.drift_detected

    def check_series(self, metric_name: str, values: list) -> dict:
        """Run ADWIN over a series of values for a given metric.

        Creates a fresh ADWIN instance (does not affect persistent state).
        This is intentional: the batch drift check runs over historical
        prediction errors accumulated since last check.  For streaming
        detection, use update() with persistent detectors via get_stats().

        Args:
            metric_name: Name of the metric.
            values: List of error/metric observations.

        Returns:
            Dict with drift_detected, drift_point (index), observations.
        """
        if not HAS_RIVER:
            return {
                "drift_detected": False,
                "drift_point": None,
                "observations": len(values),
            }

        adwin = _RiverADWIN(delta=self.delta)
        for i, v in enumerate(values):
            adwin.update(v)
            if adwin.drift_detected:
                return {
                    "drift_detected": True,
                    "drift_point": i,
                    "observations": len(values),
                }
        return {
            "drift_detected": False,
            "drift_point": None,
            "observations": len(values),
        }

    def get_stats(self) -> dict:
        """Return per-metric statistics from tracked ADWIN detectors.

        Returns:
            Dict mapping metric_name to {width, total, drift_detected}.
        """
        stats = {}
        for name, detector in self._detectors.items():
            stats[name] = {
                "width": int(detector.width),
                "total": detector.total,
                "drift_detected": detector.drift_detected,
            }
        return stats


class DriftDetector:
    """Monitor rolling prediction error and decide when to retrain.

    Uses both fixed-window median threshold (original) and Page-Hinkley
    adaptive detection. Either detector triggering counts as drift.
    Optionally runs ADWIN ensemble detection alongside for confirmation.

    Args:
        window_days: Number of days in the rolling window (default 7).
        threshold_multiplier: Retrain when MAE > multiplier * median_MAE (default 2.0).
        min_samples: Minimum scored days before drift detection is meaningful.
        use_page_hinkley: Whether to also run Page-Hinkley detection (default True).
        use_adwin: Whether to also run ADWIN detection (default True).
        adwin_delta: ADWIN confidence parameter (default 0.002).
    """

    def __init__(  # noqa: PLR0913 — drift detector configuration requires all threshold parameters
        self,
        window_days: int = 7,
        threshold_multiplier: float = 2.0,
        min_samples: int = 5,
        use_page_hinkley: bool = True,
        use_adwin: bool = True,
        adwin_delta: float = 0.002,
    ):
        self.window_days = window_days
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples
        self.use_page_hinkley = use_page_hinkley
        self.use_adwin = use_adwin
        self.adwin_delta = adwin_delta

    def _analyze_metric_errors(self, window, latest):
        """Extract per-metric error analysis from the scoring window.

        Returns (rolling_mae, current_mae, thresholds, drift_detected, ph_results, metric_errors).
        """
        metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
        rolling_mae = {}
        current_mae = {}
        thresholds = {}
        drift_detected = []
        ph_results = {}
        metric_errors = {}

        for metric in metrics:
            errors = []
            for entry in window:
                error = entry.get("metrics", {}).get(metric, {}).get("error")
                if error is not None:
                    errors.append(abs(error))

            if len(errors) < 3:
                continue

            metric_errors[metric] = errors
            median_error = statistics.median(errors)
            rolling_mae[metric] = round(median_error, 2)

            latest_error = latest.get("metrics", {}).get(metric, {}).get("error")
            if latest_error is not None:
                current_mae[metric] = round(abs(latest_error), 2)
                threshold = round(median_error * self.threshold_multiplier, 2)
                thresholds[metric] = threshold
                if abs(latest_error) > threshold and median_error > 0:
                    drift_detected.append(metric)

            if self.use_page_hinkley and len(errors) >= self.min_samples:
                ph = PageHinkleyDetector()
                ph_result = ph.check_series(errors)
                ph_results[metric] = ph_result
                if ph_result["drift_detected"] and metric not in drift_detected:
                    drift_detected.append(metric)

        return rolling_mae, current_mae, thresholds, drift_detected, ph_results, metric_errors

    def _run_adwin_analysis(self, metric_errors, drift_detected):
        """Run ADWIN drift detection on collected metric errors."""
        adwin_results = {}
        if not (self.use_adwin and HAS_RIVER):
            return adwin_results

        adwin = ADWINDetector(delta=self.adwin_delta)
        for metric, errors in metric_errors.items():
            if len(errors) >= self.min_samples:
                adwin_result = adwin.check_series(metric, errors)
                adwin_results[metric] = adwin_result
                if adwin_result["drift_detected"] and metric not in drift_detected:
                    drift_detected.append(metric)

        return adwin_results

    def _build_result(  # noqa: PLR0913 — internal builder aggregating all check outputs
        self, needs_retrain, reason, rolling_mae, current_mae,
        thresholds, window, ph_results, adwin_results,
    ):
        """Build a standardized check result dict."""
        return {
            "needs_retrain": needs_retrain,
            "reason": reason,
            "rolling_mae": rolling_mae,
            "current_mae": current_mae,
            "threshold": thresholds,
            "days_analyzed": len(window),
            "page_hinkley": ph_results if self.use_page_hinkley else {},
            "adwin": adwin_results if self.use_adwin else {},
        }

    def check(self, accuracy_history: dict) -> dict:
        """Analyze accuracy history and determine if retraining is needed.

        Args:
            accuracy_history: Dict with "scores" list, each entry having
                "date", "overall", and "metrics" (with per-metric error/accuracy).

        Returns:
            Dict with needs_retrain, reason, rolling_mae, current_mae,
            threshold, days_analyzed, page_hinkley, adwin.
        """
        scores = accuracy_history.get("scores", [])
        if len(scores) < self.min_samples:
            return {
                "needs_retrain": False,
                "reason": f"insufficient data ({len(scores)} days, need {self.min_samples})",
                "days_analyzed": len(scores),
            }

        window = scores[-self.window_days :]
        latest = scores[-1]

        rolling_mae, current_mae, thresholds, drift_detected, ph_results, metric_errors = (
            self._analyze_metric_errors(window, latest)
        )
        adwin_results = self._run_adwin_analysis(metric_errors, drift_detected)

        if drift_detected:
            methods = ["threshold"]
            if self.use_page_hinkley:
                methods.append("page-hinkley")
            if self.use_adwin:
                methods.append("adwin")
            method = " + ".join(methods)
            result = self._build_result(
                True, f"drift detected in {', '.join(drift_detected)} (method: {method})",
                rolling_mae, current_mae, thresholds, window, ph_results, adwin_results,
            )
            result["drifted_metrics"] = drift_detected
            return result

        # Check overall accuracy degradation
        if len(window) >= 3:
            recent_overall = [s["overall"] for s in window[-3:]]
            earlier_overall = [s["overall"] for s in window[:-3]] if len(window) > 3 else recent_overall
            if earlier_overall and statistics.mean(recent_overall) < statistics.mean(earlier_overall) - 10:
                return self._build_result(
                    True,
                    f"overall accuracy dropped >10% "
                    f"({statistics.mean(recent_overall):.0f}% vs "
                    f"{statistics.mean(earlier_overall):.0f}%)",
                    rolling_mae, current_mae, thresholds, window, ph_results, adwin_results,
                )

        return self._build_result(
            False, "no drift detected, error within normal range",
            rolling_mae, current_mae, thresholds, window, ph_results, adwin_results,
        )

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
