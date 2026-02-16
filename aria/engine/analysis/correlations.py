"""Pearson correlation and cross-metric correlation discovery."""

import math


def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two sequences."""
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def cross_correlate(snapshots, min_r=0.5):
    """Find significant correlations between all tracked metrics."""
    if len(snapshots) < 5:
        return []

    series = {}
    for snap in snapshots:
        series.setdefault("weather_temp", []).append(snap.get("weather", {}).get("temp_f") or 0)
        series.setdefault("calendar_count", []).append(len(snap.get("calendar_events", [])))
        series.setdefault("is_weekend", []).append(1 if snap.get("is_weekend") else 0)
        series.setdefault("power_watts", []).append(snap["power"]["total_watts"])
        series.setdefault("lights_on", []).append(snap["lights"]["on"])
        series.setdefault("devices_home", []).append(snap["occupancy"]["device_count_home"])
        series.setdefault("unavailable", []).append(snap["entities"]["unavailable"])
        series.setdefault("useful_events", []).append(snap["logbook_summary"].get("useful_events", 0))
        ev = snap.get("ev", {}).get("TARS", {})
        series.setdefault("ev_battery", []).append(ev.get("battery_pct", 0))
        series.setdefault("ev_power", []).append(ev.get("charger_power_kw", 0))

    keys = list(series.keys())
    results = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = pearson_r(series[keys[i]], series[keys[j]])
            if abs(r) >= min_r:
                strength = "strong" if abs(r) >= 0.8 else "moderate"
                direction = "positive" if r > 0 else "negative"
                results.append(
                    {
                        "x": keys[i],
                        "y": keys[j],
                        "r": round(r, 3),
                        "strength": strength,
                        "direction": direction,
                        "description": f"{keys[i]} ↔ {keys[j]}: r={r:.2f} ({strength} {direction})",
                    }
                )

    results.sort(key=lambda c: -abs(c["r"]))
    return results
