"""Default feature configuration for ML pipeline.

Provides the canonical default config, load/save via DataStore, and validation.
"""

import copy
from datetime import datetime

DEFAULT_FEATURE_CONFIG = {
    "version": 1,
    "last_modified": "",
    "modified_by": "initial",
    "time_features": {
        "hour_sin_cos": True,
        "dow_sin_cos": True,
        "month_sin_cos": True,
        "day_of_year_sin_cos": True,
        "is_weekend": True,
        "is_holiday": True,
        "is_night": True,
        "is_work_hours": True,
        "minutes_since_sunrise": True,
        "minutes_until_sunset": True,
        "daylight_remaining_pct": True,
    },
    "weather_features": {
        "temp_f": True,
        "humidity_pct": True,
        "wind_mph": True,
    },
    "home_features": {
        "people_home_count": True,
        "device_count_home": True,
        "lights_on": True,
        "total_brightness": True,
        "motion_active_count": True,
        "active_media_players": True,
        "ev_battery_pct": True,
        "ev_is_charging": True,
    },
    "lag_features": {
        "prev_snapshot_power": True,
        "prev_snapshot_lights": True,
        "prev_snapshot_occupancy": True,
        "rolling_7d_power_mean": True,
        "rolling_7d_lights_mean": True,
    },
    "interaction_features": {
        "is_weekend_x_temp": False,
        "people_home_x_hour_sin": False,
        "daylight_x_lights": False,
    },
    "presence_features": {
        "presence_probability": True,
        "presence_occupied_rooms": True,
        "presence_identified_persons": True,
        "presence_camera_signals": True,
    },
    "target_metrics": [
        "power_watts",
        "lights_on",
        "devices_home",
        "unavailable",
        "useful_events",
    ],
}

# Required top-level sections in a valid feature config
_REQUIRED_SECTIONS = {
    "time_features",
    "weather_features",
    "home_features",
    "lag_features",
    "interaction_features",
    "presence_features",
    "target_metrics",
}


def validate_feature_config(config):
    """Validate a feature config dict. Returns list of error strings (empty = valid)."""
    errors = []
    if not isinstance(config, dict):
        return ["Feature config must be a dict"]

    for section in _REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate feature sections contain only bool values
    for section in [
        "time_features",
        "weather_features",
        "home_features",
        "lag_features",
        "interaction_features",
        "presence_features",
    ]:
        sub = config.get(section, {})
        if not isinstance(sub, dict):
            errors.append(f"Section '{section}' must be a dict")
            continue
        for key, val in sub.items():
            if not isinstance(val, bool):
                errors.append(f"{section}.{key} must be bool, got {type(val).__name__}")

    # target_metrics must be a list of strings
    tm = config.get("target_metrics", [])
    if not isinstance(tm, list):
        errors.append("target_metrics must be a list")
    elif not all(isinstance(m, str) for m in tm):
        errors.append("target_metrics must contain only strings")

    return errors


def load_feature_config(store=None):
    """Load feature config, falling back to default if missing.

    Args:
        store: Optional DataStore instance for file-based config.
               If None, returns a copy of DEFAULT_FEATURE_CONFIG.
    """
    if store is not None:
        saved = store.load_feature_config()
        if saved is not None:
            return saved
    return copy.deepcopy(DEFAULT_FEATURE_CONFIG)


def save_feature_config(config, store):
    """Save feature config via DataStore with timestamp update.

    Args:
        config: Feature config dict to save.
        store: DataStore instance (required — I/O is the storage layer's job).
    """
    config["last_modified"] = datetime.now().isoformat()
    store.save_feature_config(config)
