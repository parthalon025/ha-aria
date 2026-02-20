"""Shared constants used across ARIA layers.

These constants are referenced by both engine and hub modules.
Centralizing them here eliminates cross-layer import coupling
(e.g., hub importing directly from engine internals).
"""

# Trajectory classification labels — used by:
#   - aria.engine.sequence.SequenceClassifier (heuristic + DTW labeling)
#   - aria.modules.ml_engine (trajectory encoding for feature vectors)
TRAJECTORY_CLASSES: list[str] = ["stable", "ramping_up", "winding_down", "anomalous_transition"]

# Default feature configuration — canonical defaults for ML pipeline.
# Used by:
#   - aria.engine.features.feature_config (load/save/validate)
#   - aria.engine.features.vector_builder (feature vector construction)
#   - aria.engine.llm.meta_learning (feature config adjustments)
#   - aria.engine.models.reference_model (clean reference training)
#   - aria.modules.ml_engine (hub-side feature names)
DEFAULT_FEATURE_CONFIG: dict = {
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
    "pattern_features": {
        "trajectory_class": True,
    },
    "event_features": {
        "event_count": True,
        "light_transitions": True,
        "motion_events": True,
        "unique_entities_active": True,
        "domain_entropy": True,
    },
    "target_metrics": [
        "power_watts",
        "lights_on",
        "devices_home",
        "unavailable",
        "useful_events",
    ],
}
