"""Prediction generation — blends statistical baselines with ML model output.

Extracted from monolith generate_predictions (lines 1578-1648) and
blend_predictions (lines 1173-1186).
"""

import os
from datetime import datetime

from ha_intelligence.config import PathConfig


def blend_predictions(stat_pred, ml_pred, days_of_data):
    """Blend statistical and ML predictions based on data maturity."""
    if days_of_data < 14 or ml_pred is None:
        return stat_pred

    if days_of_data < 60:
        ml_weight = 0.3
    elif days_of_data < 90:
        ml_weight = 0.5
    else:
        ml_weight = 0.7

    stat_weight = 1.0 - ml_weight
    return round(stat_pred * stat_weight + ml_pred * ml_weight, 1)


def count_days_of_data(paths=None):
    """Count how many days of snapshot data exist.

    Args:
        paths: PathConfig instance. Uses default PathConfig if None.
    """
    if paths is None:
        paths = PathConfig()
    daily_dir = paths.daily_dir
    if not daily_dir.is_dir():
        return 0
    return len([f for f in os.listdir(daily_dir) if f.endswith(".json")])


def generate_predictions(target_date, baselines, correlations=None, weather_forecast=None,
                         ml_predictions=None, device_failures=None, contextual_anomalies=None,
                         paths=None):
    """Generate predictions for a target date with optional ML blending.

    Args:
        target_date: Date string in YYYY-MM-DD format.
        baselines: Dict of day-of-week baselines (e.g. baselines["Monday"]["power_watts"]).
        correlations: Optional list of correlation dicts with x, y, r keys.
        weather_forecast: Optional dict with temp_f key.
        ml_predictions: Optional dict of metric -> ML predicted value.
        device_failures: Optional list of predicted device failures.
        contextual_anomalies: Optional contextual anomaly results.
        paths: Optional PathConfig for locating snapshot data directory.
    """
    dt = datetime.strptime(target_date, "%Y-%m-%d")
    dow = dt.strftime("%A")
    baseline = baselines.get(dow, {})
    days = count_days_of_data(paths)

    predictions = {
        "target_date": target_date,
        "day_of_week": dow,
        "generated_at": datetime.now().isoformat(),
        "prediction_method": "blended" if ml_predictions else "statistical",
        "days_of_data": days,
    }

    metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]

    for metric in metrics:
        bl = baseline.get(metric, {})
        mean = bl.get("mean", 0)
        stddev = bl.get("stddev", 0)
        sample_count = baseline.get("sample_count", 0)

        stat_predicted = mean
        adjustments = []

        if weather_forecast and correlations:
            temp = weather_forecast.get("temp_f")
            if temp is not None:
                for corr in correlations:
                    if corr["x"] == "weather_temp" and corr["y"] == metric:
                        temp_deviation = (temp - 72) / 30
                        adjustment = stat_predicted * temp_deviation * abs(corr["r"]) * 0.2
                        stat_predicted += adjustment
                        adjustments.append(f"weather({temp}°F): {'+' if adjustment > 0 else ''}{adjustment:.0f}")

        # Blend with ML prediction if available
        ml_val = ml_predictions.get(metric) if ml_predictions else None
        if ml_val is not None:
            predicted = blend_predictions(stat_predicted, ml_val, days)
            adjustments.append(f"ml_blend(weight={0.3 if days < 60 else (0.5 if days < 90 else 0.7)}): {ml_val:.1f}")
        else:
            predicted = stat_predicted

        if sample_count >= 7:
            confidence = "high"
        elif sample_count >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        predictions[metric] = {
            "predicted": round(predicted, 1),
            "baseline_mean": mean,
            "baseline_stddev": stddev,
            "confidence": confidence,
            "adjustments": adjustments,
        }

    # Attach device failure predictions
    if device_failures:
        predictions["device_failures"] = device_failures

    # Attach contextual anomaly result
    if contextual_anomalies:
        predictions["contextual_anomalies"] = contextual_anomalies

    return predictions
