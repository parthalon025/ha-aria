"""Prediction scoring and accuracy tracking.

Extracted from monolith lines 1662-1733: METRIC_TO_ACTUAL, score_prediction,
score_all_predictions, accuracy_trend.
"""

import statistics

# Mapping from prediction metric to snapshot accessor
METRIC_TO_ACTUAL = {
    "power_watts": lambda s: s["power"]["total_watts"],
    "lights_on": lambda s: s["lights"]["on"],
    "devices_home": lambda s: s["occupancy"]["device_count_home"],
    "unavailable": lambda s: s["entities"]["unavailable"],
    "useful_events": lambda s: s["logbook_summary"].get("useful_events", 0),
}


def score_prediction(metric, predictions, actual_snapshot):
    """Score a single prediction against actual data.

    Accuracy = max(0, 100 - |error| / stddev * 25).
    """
    pred_data = predictions.get(metric, {})
    predicted = pred_data.get("predicted", 0)
    stddev = pred_data.get("baseline_stddev", 1) or 1

    actual_fn = METRIC_TO_ACTUAL.get(metric)
    if actual_fn is None:
        return {"accuracy": 0, "error": None}
    actual = actual_fn(actual_snapshot)

    error = abs(predicted - actual)
    sigma_error = error / stddev if stddev > 0 else error
    accuracy = max(0, round(100 - sigma_error * 25))

    return {
        "accuracy": accuracy,
        "predicted": predicted,
        "actual": actual,
        "error": round(error, 1),
        "sigma_error": round(sigma_error, 2),
    }


def score_all_predictions(predictions, actual_snapshot):
    """Score all predictions and return overall accuracy with method tracking."""
    metrics = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
    scores = {}
    accuracies = []
    for metric in metrics:
        result = score_prediction(metric, predictions, actual_snapshot)
        scores[metric] = result
        if result["accuracy"] is not None:
            accuracies.append(result["accuracy"])

    overall = round(statistics.mean(accuracies)) if accuracies else 0
    return {
        "date": predictions.get("target_date", ""),
        "overall": overall,
        "prediction_method": predictions.get("prediction_method", "statistical"),
        "days_of_data": predictions.get("days_of_data", 0),
        "metrics": scores,
    }


def accuracy_trend(history):
    """Determine if accuracy is improving, degrading, or stable."""
    scores = history.get("scores", [])
    if len(scores) < 3:
        return "insufficient_data"
    recent = [s["overall"] for s in scores[-3:]]
    earlier = [s["overall"] for s in scores[-6:-3]] if len(scores) >= 6 else [s["overall"] for s in scores[:3]]
    recent_avg = statistics.mean(recent)
    earlier_avg = statistics.mean(earlier)
    if recent_avg > earlier_avg + 3:
        return "improving"
    elif recent_avg < earlier_avg - 3:
        return "degrading"
    return "stable"
