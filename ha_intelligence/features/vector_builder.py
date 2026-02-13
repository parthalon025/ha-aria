"""Feature vector construction from snapshots using feature config."""

from .feature_config import DEFAULT_FEATURE_CONFIG, load_feature_config


def get_feature_names(config=None):
    """Return ordered list of feature names from config."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG
    names = []
    tc = config.get("time_features", {})
    if tc.get("hour_sin_cos"):
        names.extend(["hour_sin", "hour_cos"])
    if tc.get("dow_sin_cos"):
        names.extend(["dow_sin", "dow_cos"])
    if tc.get("month_sin_cos"):
        names.extend(["month_sin", "month_cos"])
    if tc.get("day_of_year_sin_cos"):
        names.extend(["day_of_year_sin", "day_of_year_cos"])
    for simple in ["is_weekend", "is_holiday", "is_night", "is_work_hours",
                    "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct"]:
        if tc.get(simple):
            names.append(simple)

    for key in config.get("weather_features", {}):
        if config["weather_features"][key]:
            names.append(f"weather_{key}")

    for key in config.get("home_features", {}):
        if config["home_features"][key]:
            names.append(key)

    for key in config.get("lag_features", {}):
        if config["lag_features"][key]:
            names.append(key)

    for key in config.get("interaction_features", {}):
        if config["interaction_features"][key]:
            names.append(key)

    return names


def build_feature_vector(snapshot, config=None, prev_snapshot=None, rolling_stats=None):
    """Build a feature vector (dict) from a snapshot using the feature config.

    Returns dict of feature_name -> float value.
    """
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    features = {}
    tf = snapshot.get("time_features", {})
    tc = config.get("time_features", {})

    # Time features
    if tc.get("hour_sin_cos"):
        features["hour_sin"] = tf.get("hour_sin", 0)
        features["hour_cos"] = tf.get("hour_cos", 0)
    if tc.get("dow_sin_cos"):
        features["dow_sin"] = tf.get("dow_sin", 0)
        features["dow_cos"] = tf.get("dow_cos", 0)
    if tc.get("month_sin_cos"):
        features["month_sin"] = tf.get("month_sin", 0)
        features["month_cos"] = tf.get("month_cos", 0)
    if tc.get("day_of_year_sin_cos"):
        features["day_of_year_sin"] = tf.get("day_of_year_sin", 0)
        features["day_of_year_cos"] = tf.get("day_of_year_cos", 0)
    for simple in ["is_weekend", "is_holiday", "is_night", "is_work_hours",
                    "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct"]:
        if tc.get(simple):
            val = tf.get(simple, 0)
            features[simple] = 1 if val is True else (0 if val is False else (val or 0))

    # Weather features
    weather = snapshot.get("weather", {})
    for key, enabled in config.get("weather_features", {}).items():
        if enabled:
            features[f"weather_{key}"] = weather.get(key) or 0

    # Home state features
    hc = config.get("home_features", {})
    if hc.get("people_home_count"):
        features["people_home_count"] = snapshot.get("occupancy", {}).get("people_home_count",
            len(snapshot.get("occupancy", {}).get("people_home", [])))
    if hc.get("device_count_home"):
        features["device_count_home"] = snapshot.get("occupancy", {}).get("device_count_home", 0)
    if hc.get("lights_on"):
        features["lights_on"] = snapshot.get("lights", {}).get("on", 0)
    if hc.get("total_brightness"):
        features["total_brightness"] = snapshot.get("lights", {}).get("total_brightness", 0)
    if hc.get("motion_active_count"):
        features["motion_active_count"] = snapshot.get("motion", {}).get("active_count", 0)
    if hc.get("active_media_players"):
        features["active_media_players"] = snapshot.get("media", {}).get("total_active", 0)
    if hc.get("ev_battery_pct"):
        features["ev_battery_pct"] = snapshot.get("ev", {}).get("TARS", {}).get("battery_pct", 0)
    if hc.get("ev_is_charging"):
        features["ev_is_charging"] = 1 if snapshot.get("ev", {}).get("TARS", {}).get("is_charging") else 0

    # Lag features
    lc = config.get("lag_features", {})
    if lc.get("prev_snapshot_power") and prev_snapshot:
        features["prev_snapshot_power"] = prev_snapshot.get("power", {}).get("total_watts", 0)
    elif lc.get("prev_snapshot_power"):
        features["prev_snapshot_power"] = 0
    if lc.get("prev_snapshot_lights") and prev_snapshot:
        features["prev_snapshot_lights"] = prev_snapshot.get("lights", {}).get("on", 0)
    elif lc.get("prev_snapshot_lights"):
        features["prev_snapshot_lights"] = 0
    if lc.get("prev_snapshot_occupancy") and prev_snapshot:
        features["prev_snapshot_occupancy"] = prev_snapshot.get("occupancy", {}).get("device_count_home", 0)
    elif lc.get("prev_snapshot_occupancy"):
        features["prev_snapshot_occupancy"] = 0
    if lc.get("rolling_7d_power_mean"):
        features["rolling_7d_power_mean"] = (rolling_stats or {}).get("power_mean_7d", 0)
    if lc.get("rolling_7d_lights_mean"):
        features["rolling_7d_lights_mean"] = (rolling_stats or {}).get("lights_mean_7d", 0)

    # Interaction features
    ic = config.get("interaction_features", {})
    if ic.get("is_weekend_x_temp"):
        features["is_weekend_x_temp"] = features.get("is_weekend", 0) * features.get("weather_temp_f", 0)
    if ic.get("people_home_x_hour_sin"):
        features["people_home_x_hour_sin"] = features.get("people_home_count", 0) * features.get("hour_sin", 0)
    if ic.get("daylight_x_lights"):
        features["daylight_x_lights"] = features.get("daylight_remaining_pct", 0) * features.get("lights_on", 0)

    return features


def extract_target_values(snapshot):
    """Extract target metric values from a snapshot for training."""
    return {
        "power_watts": snapshot.get("power", {}).get("total_watts", 0),
        "lights_on": snapshot.get("lights", {}).get("on", 0),
        "devices_home": snapshot.get("occupancy", {}).get("device_count_home", 0),
        "unavailable": snapshot.get("entities", {}).get("unavailable", 0),
        "useful_events": snapshot.get("logbook_summary", {}).get("useful_events", 0),
    }


def build_training_data(snapshots, config=None):
    """Build feature matrix and target arrays from a list of snapshots.

    Returns (feature_names, X_list_of_dicts, targets_dict_of_lists).
    """
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    feature_names = get_feature_names(config)
    X = []
    targets = {m: [] for m in config.get("target_metrics", DEFAULT_FEATURE_CONFIG["target_metrics"])}

    for i, snap in enumerate(snapshots):
        prev = snapshots[i - 1] if i > 0 else None
        # Simple rolling stats
        rolling = {}
        if i >= 7:
            recent = snapshots[max(0, i - 7):i]
            rolling["power_mean_7d"] = sum(s.get("power", {}).get("total_watts", 0) for s in recent) / len(recent)
            rolling["lights_mean_7d"] = sum(s.get("lights", {}).get("on", 0) for s in recent) / len(recent)

        fv = build_feature_vector(snap, config, prev, rolling)
        # Convert to ordered list matching feature_names
        row = [fv.get(name, 0) for name in feature_names]
        X.append(row)

        tv = extract_target_values(snap)
        for metric in targets:
            targets[metric].append(tv.get(metric, 0))

    return feature_names, X, targets
