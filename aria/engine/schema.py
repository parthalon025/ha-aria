"""Shared engine-hub schema definitions and validators.

This module defines the canonical snapshot schema, ensuring that the engine
and hub read/write the same structure. All schema changes MUST be made here.
"""

from typing import Any

# Required nested keys — only check structure within sections that are present
REQUIRED_NESTED_KEYS = {
    "power": {"total_watts"},
    "occupancy": {"device_count_home"},  # Relaxed: don't require people_home/people_away
    "lights": {"on"},  # Relaxed: don't require off count
    "logbook_summary": {"useful_events"},  # Relaxed: don't require total_events/by_domain
    "entities": {"unavailable"},  # Relaxed: don't require by_domain/total
}


def validate_snapshot_schema(snapshot: dict[str, Any]) -> list[str]:
    """Validate snapshot against required schema.

    This checks that:
    1. All major data sections that are present are dicts (not null/invalid types)
    2. Within present sections, critical keys exist

    Snapshots may omit entire sections (e.g., weather if unavailable), which is OK.
    Returns early on type mismatch or missing critical keys within present sections.

    Args:
        snapshot: Snapshot dictionary to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # For each section that's supposed to exist, validate its structure
    for section, required_nested in REQUIRED_NESTED_KEYS.items():
        if section not in snapshot:
            continue  # OK if section is missing entirely
        section_data = snapshot[section]
        if not isinstance(section_data, dict):
            errors.append(f"Section '{section}' is not a dict: {type(section_data)}")
            continue
        missing_nested = required_nested - set(section_data.keys())
        if missing_nested:
            errors.append(f"Missing keys in '{section}': {missing_nested}")

    return errors
