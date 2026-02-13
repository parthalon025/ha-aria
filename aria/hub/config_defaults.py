"""Config defaults registry — single source of truth for all tunable parameters.

Each parameter is defined with its key, default value, type, constraints, and
UI metadata. On startup, seed_config_defaults() inserts any missing keys using
INSERT OR IGNORE, preserving user overrides.
"""

from typing import Any, Dict, List


CONFIG_DEFAULTS: List[Dict[str, Any]] = [
    # ── Activity Monitor ──────────────────────────────────────────────
    {
        "key": "activity.daily_snapshot_cap",
        "default_value": "20",
        "value_type": "number",
        "label": "Daily Snapshot Cap",
        "description": "Maximum adaptive snapshots triggered per day.",
        "category": "Activity Monitor",
        "min_value": 5,
        "max_value": 100,
        "step": 1,
    },
    {
        "key": "activity.snapshot_cooldown_s",
        "default_value": "1800",
        "value_type": "number",
        "label": "Snapshot Cooldown (s)",
        "description": "Minimum seconds between adaptive snapshots.",
        "category": "Activity Monitor",
        "min_value": 300,
        "max_value": 7200,
        "step": 60,
    },
    {
        "key": "activity.flush_interval_s",
        "default_value": "900",
        "value_type": "number",
        "label": "Flush Interval (s)",
        "description": "How often buffered events are flushed to cache windows.",
        "category": "Activity Monitor",
        "min_value": 60,
        "max_value": 3600,
        "step": 60,
    },
    {
        "key": "activity.max_window_age_h",
        "default_value": "24",
        "value_type": "number",
        "label": "Max Window Age (h)",
        "description": "Rolling window retention in hours.",
        "category": "Activity Monitor",
        "min_value": 6,
        "max_value": 168,
        "step": 1,
    },
    # ── Feature Engineering ───────────────────────────────────────────
    {
        "key": "features.decay_half_life_days",
        "default_value": "7",
        "value_type": "number",
        "label": "Decay Half-Life (days)",
        "description": "Exponential decay half-life for training sample recency weighting.",
        "category": "Feature Engineering",
        "min_value": 1,
        "max_value": 30,
        "step": 1,
    },
    {
        "key": "features.weekday_alignment_bonus",
        "default_value": "1.5",
        "value_type": "number",
        "label": "Weekday Alignment Bonus",
        "description": "Multiplier for training samples from the same day of week.",
        "category": "Feature Engineering",
        "min_value": 1.0,
        "max_value": 3.0,
        "step": 0.1,
    },
    # ── Shadow Engine ─────────────────────────────────────────────────
    {
        "key": "shadow.min_confidence",
        "default_value": "0.3",
        "value_type": "number",
        "label": "Min Confidence",
        "description": "Predictions below this confidence are not stored.",
        "category": "Shadow Engine",
        "min_value": 0.05,
        "max_value": 0.9,
        "step": 0.05,
    },
    {
        "key": "shadow.default_window_seconds",
        "default_value": "600",
        "value_type": "number",
        "label": "Default Window (s)",
        "description": "Evaluation window for predictions in seconds.",
        "category": "Shadow Engine",
        "min_value": 60,
        "max_value": 3600,
        "step": 30,
    },
    {
        "key": "shadow.resolution_interval_s",
        "default_value": "60",
        "value_type": "number",
        "label": "Resolution Interval (s)",
        "description": "How often expired prediction windows are resolved.",
        "category": "Shadow Engine",
        "min_value": 10,
        "max_value": 300,
        "step": 10,
    },
    {
        "key": "shadow.prediction_cooldown_s",
        "default_value": "30",
        "value_type": "number",
        "label": "Prediction Cooldown (s)",
        "description": "Minimum seconds between prediction attempts (debounce).",
        "category": "Shadow Engine",
        "min_value": 5,
        "max_value": 300,
        "step": 5,
    },
    # ── Data Quality ──────────────────────────────────────────────────
    {
        "key": "curation.auto_exclude_domains",
        "default_value": "update,tts,stt,scene,button,number,select,input_boolean,input_number,input_select,input_text,input_datetime,counter,script,zone,sun,weather,conversation,event,automation,camera,image,remote",
        "value_type": "string",
        "label": "Auto-Exclude Domains",
        "description": "Comma-separated domains automatically excluded from curation.",
        "category": "Data Quality",
    },
    {
        "key": "curation.noise_event_threshold",
        "default_value": "1000",
        "value_type": "number",
        "label": "Noise Event Threshold",
        "description": "Daily event count above which an entity is considered noise (if low state variety).",
        "category": "Data Quality",
        "min_value": 100,
        "max_value": 10000,
        "step": 100,
    },
    {
        "key": "curation.stale_days_threshold",
        "default_value": "30",
        "value_type": "number",
        "label": "Stale Days Threshold",
        "description": "Entities with no state changes in this many days are auto-excluded.",
        "category": "Data Quality",
        "min_value": 7,
        "max_value": 90,
        "step": 1,
    },
    {
        "key": "curation.vehicle_patterns",
        "default_value": "tesla,luda,tessy,vehicle,car_",
        "value_type": "string",
        "label": "Vehicle Patterns",
        "description": "Comma-separated patterns to match vehicle-related entity names.",
        "category": "Data Quality",
    },
]


async def seed_config_defaults(cache) -> int:
    """Seed all config defaults into the database.

    Uses INSERT OR IGNORE so existing user overrides are preserved.

    Args:
        cache: CacheManager instance (must be initialized).

    Returns:
        Number of new parameters inserted.
    """
    inserted = 0
    for param in CONFIG_DEFAULTS:
        was_inserted = await cache.upsert_config_default(
            key=param["key"],
            default_value=param["default_value"],
            value_type=param["value_type"],
            label=param.get("label", ""),
            description=param.get("description", ""),
            category=param.get("category", ""),
            min_value=param.get("min_value"),
            max_value=param.get("max_value"),
            options=param.get("options"),
            step=param.get("step"),
        )
        if was_inserted:
            inserted += 1
    return inserted
