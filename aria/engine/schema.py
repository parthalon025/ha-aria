"""Shared engine-hub schema definitions and validators.

This module defines the canonical snapshot schema, ensuring that the engine
and hub read/write the same structure. All schema changes MUST be made here.
"""

from typing import Any

# Required top-level keys in snapshot JSON
REQUIRED_SNAPSHOT_KEYS = {
    "date",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "weather",
    "entities",
    "power",
    "occupancy",
    "lights",
    "logbook_summary",
}

# Required nested keys within snapshot sections
REQUIRED_NESTED_KEYS = {
    "power": {"total_watts"},
    "occupancy": {"people_home", "people_away"},
    "lights": {"on", "off"},
    "logbook_summary": {"total_events", "useful_events", "by_domain"},
    "entities": {"total", "by_domain"},
}


def validate_snapshot_schema(snapshot: dict[str, Any]) -> list[str]:
    """Validate snapshot against required schema.

    Args:
        snapshot: Snapshot dictionary to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check top-level keys
    missing_keys = REQUIRED_SNAPSHOT_KEYS - set(snapshot.keys())
    if missing_keys:
        errors.append(f"Missing top-level keys: {missing_keys}")

    # Check nested keys
    for section, required_nested in REQUIRED_NESTED_KEYS.items():
        if section not in snapshot:
            continue
        section_data = snapshot[section]
        if not isinstance(section_data, dict):
            errors.append(f"Section '{section}' is not a dict: {type(section_data)}")
            continue
        missing_nested = required_nested - set(section_data.keys())
        if missing_nested:
            errors.append(f"Missing keys in '{section}': {missing_nested}")

    return errors
