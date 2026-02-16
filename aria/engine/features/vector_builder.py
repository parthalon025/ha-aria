"""Feature vector construction from snapshots using feature config."""

from .feature_config import DEFAULT_FEATURE_CONFIG


def _get_time_feature_names(tc):
    """Extract enabled time feature names from time_features config."""
    names = []
    sin_cos_pairs = [
        ("hour_sin_cos", ["hour_sin", "hour_cos"]),
        ("dow_sin_cos", ["dow_sin", "dow_cos"]),
        ("month_sin_cos", ["month_sin", "month_cos"]),
        ("day_of_year_sin_cos", ["day_of_year_sin", "day_of_year_cos"]),
    ]
    for key, pair_names in sin_cos_pairs:
        if tc.get(key):
            names.extend(pair_names)

    simple_features = [
        "is_weekend", "is_holiday", "is_night", "is_work_hours",
        "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct",
    ]
    for simple in simple_features:
        if tc.get(simple):
            names.append(simple)
    return names


def _get_enabled_keys(config, section, prefix=""):
    """Get enabled feature names from a config section."""
    return [f"{prefix}{key}" for key, enabled in config.get(section, {}).items() if enabled]


def get_feature_names(config=None):
    """Return ordered list of feature names from config."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG
    names = _get_time_feature_names(config.get("time_features", {}))
    names.extend(_get_enabled_keys(config, "weather_features", prefix="weather_"))
    names.extend(_get_enabled_keys(config, "home_features"))
    names.extend(_get_enabled_keys(config, "lag_features"))
    names.extend(_get_enabled_keys(config, "interaction_features"))
    names.extend(_get_enabled_keys(config, "presence_features"))
    return names


def _build_time_features(features, snapshot, tc):
    """Extract time features from snapshot into features dict."""
    tf = snapshot.get("time_features", {})
    sin_cos_pairs = [
        ("hour_sin_cos", ["hour_sin", "hour_cos"]),
        ("dow_sin_cos", ["dow_sin", "dow_cos"]),
        ("month_sin_cos", ["month_sin", "month_cos"]),
        ("day_of_year_sin_cos", ["day_of_year_sin", "day_of_year_cos"]),
    ]
    for key, pair_names in sin_cos_pairs:
        if tc.get(key):
            for name in pair_names:
                features[name] = tf.get(name, 0)

    simple_features = [
        "is_weekend", "is_holiday", "is_night", "is_work_hours",
        "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct",
    ]
    for simple in simple_features:
        if tc.get(simple):
            val = tf.get(simple, 0)
            features[simple] = 1 if val is True else (0 if val is False else (val or 0))


def _build_home_features(features, snapshot, hc):
    """Extract home state features from snapshot into features dict."""
    if hc.get("people_home_count"):
        features["people_home_count"] = snapshot.get("occupancy", {}).get(
            "people_home_count", len(snapshot.get("occupancy", {}).get("people_home", []))
        )
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


def _build_lag_features(features, prev_snapshot, rolling_stats, lc):
    """Extract lag features from previous snapshot and rolling stats."""
    lag_mappings = [
        ("prev_snapshot_power", lambda ps: ps.get("power", {}).get("total_watts", 0)),
        ("prev_snapshot_lights", lambda ps: ps.get("lights", {}).get("on", 0)),
        ("prev_snapshot_occupancy", lambda ps: ps.get("occupancy", {}).get("device_count_home", 0)),
    ]
    for name, extractor in lag_mappings:
        if lc.get(name):
            features[name] = extractor(prev_snapshot) if prev_snapshot else 0

    rolling = rolling_stats or {}
    if lc.get("rolling_7d_power_mean"):
        features["rolling_7d_power_mean"] = rolling.get("power_mean_7d", 0)
    if lc.get("rolling_7d_lights_mean"):
        features["rolling_7d_lights_mean"] = rolling.get("lights_mean_7d", 0)


def build_feature_vector(snapshot, config=None, prev_snapshot=None, rolling_stats=None):
    """Build a feature vector (dict) from a snapshot using the feature config.

    Returns dict of feature_name -> float value.
    """
    if config is None:
        config = DEFAULT_FEATURE_CONFIG

    features = {}

    # Time features
    _build_time_features(features, snapshot, config.get("time_features", {}))

    # Weather features
    weather = snapshot.get("weather", {})
    for key, enabled in config.get("weather_features", {}).items():
        if enabled:
            features[f"weather_{key}"] = weather.get(key) or 0

    # Home state features
    _build_home_features(features, snapshot, config.get("home_features", {}))

    # Lag features
    _build_lag_features(features, prev_snapshot, rolling_stats, config.get("lag_features", {}))

    # Interaction features
    _build_interaction_features(features, config.get("interaction_features", {}))

    # Presence features (from real-time presence module cache)
    _build_presence_features(features, snapshot, config.get("presence_features", {}))

    return features


def _build_interaction_features(features, ic):
    """Add interaction (cross-term) features to the feature vector."""
    if ic.get("is_weekend_x_temp"):
        features["is_weekend_x_temp"] = features.get("is_weekend", 0) * features.get("weather_temp_f", 0)
    if ic.get("people_home_x_hour_sin"):
        features["people_home_x_hour_sin"] = features.get("people_home_count", 0) * features.get("hour_sin", 0)
    if ic.get("daylight_x_lights"):
        features["daylight_x_lights"] = features.get("daylight_remaining_pct", 0) * features.get("lights_on", 0)


def _build_presence_features(features, snapshot, pc):
    """Add presence-based features to the feature vector."""
    presence = snapshot.get("presence", {})
    if pc.get("presence_probability"):
        features["presence_probability"] = presence.get("overall_probability", 0)
    if pc.get("presence_occupied_rooms"):
        features["presence_occupied_rooms"] = presence.get("occupied_room_count", 0)
    if pc.get("presence_identified_persons"):
        features["presence_identified_persons"] = presence.get("identified_person_count", 0)
    if pc.get("presence_camera_signals"):
        features["presence_camera_signals"] = presence.get("camera_signal_count", 0)


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
            recent = snapshots[max(0, i - 7) : i]
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
