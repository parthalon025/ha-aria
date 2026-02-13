"""Compute per-day-of-week baselines from historical snapshots."""

import statistics


def compute_baselines(snapshots):
    """Compute per-day-of-week baselines from historical snapshots."""
    by_day = {}
    for snap in snapshots:
        dow = snap.get("day_of_week", "Unknown")
        by_day.setdefault(dow, []).append(snap)

    baselines = {}
    for dow, snaps in by_day.items():
        metrics = {
            "power_watts": [s["power"]["total_watts"] for s in snaps],
            "lights_on": [s["lights"]["on"] for s in snaps],
            "lights_off": [s["lights"]["off"] for s in snaps],
            "devices_home": [s["occupancy"]["device_count_home"] for s in snaps],
            "unavailable": [s["entities"]["unavailable"] for s in snaps],
            "useful_events": [s["logbook_summary"].get("useful_events", 0) for s in snaps],
        }

        baseline = {"sample_count": len(snaps)}
        for metric_name, values in metrics.items():
            if len(values) >= 2:
                baseline[metric_name] = {
                    "mean": statistics.mean(values),
                    "stddev": statistics.stdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            elif len(values) == 1:
                baseline[metric_name] = {
                    "mean": values[0],
                    "stddev": 0,
                    "min": values[0],
                    "max": values[0],
                }
        baselines[dow] = baseline

    return baselines
