"""Z-score anomaly detection against day-of-week baselines."""

ANOMALY_THRESHOLD = 2.0  # z-score above which we flag anomaly


def detect_anomalies(snapshot, baselines):
    """Detect z-score anomalies vs day-of-week baseline."""
    dow = snapshot.get("day_of_week", "Unknown")
    baseline = baselines.get(dow, {})
    if not baseline:
        return []

    current_values = {
        "power_watts": snapshot["power"]["total_watts"],
        "lights_on": snapshot["lights"]["on"],
        "devices_home": snapshot["occupancy"]["device_count_home"],
        "unavailable": snapshot["entities"]["unavailable"],
        "useful_events": snapshot["logbook_summary"].get("useful_events", 0),
    }

    anomalies = []
    for metric, current in current_values.items():
        bl = baseline.get(metric, {})
        mean = bl.get("mean")
        stddev = bl.get("stddev")
        if mean is None or stddev is None or stddev == 0:
            continue
        z = abs(current - mean) / stddev
        if z > ANOMALY_THRESHOLD:
            direction = "above" if current > mean else "below"
            anomalies.append({
                "metric": metric,
                "current": current,
                "mean": mean,
                "stddev": stddev,
                "z_score": round(z, 2),
                "direction": direction,
                "description": f"{metric} is {z:.1f}σ {direction} normal ({current} vs {mean:.0f}±{stddev:.0f})",
            })

    return anomalies
