"""Condition builder — presence, illuminance, time, weekday, safety conditions.

Generates additive HA conditions based on detected patterns and area context.
Conditions are optional enrichments — an automation is valid without any.
"""

import logging

from aria.automation.models import DetectionResult
from aria.automation.trigger_builder import _quote_state

logger = logging.getLogger(__name__)

# Safety conditions injected by default per action service domain.
# Each entry: (condition_type, description) used to generate condition dicts.
SAFETY_CONDITIONS = {
    "light.turn_on": [
        ("presence", "person.* == home"),
        ("illuminance", "sensor.*_illuminance < 50"),
    ],
    "notify.*": [
        ("quiet_hours", "time NOT between 23:00-07:00"),
    ],
    "climate.*": [
        ("presence", "person.* == home"),
    ],
}

# Weekday mappings for day_type conditions
DAY_TYPE_WEEKDAYS = {
    "workday": ["mon", "tue", "wed", "thu", "fri"],
    "weekend": ["sat", "sun"],
}


def build_conditions(
    detection: DetectionResult,
    entity_graph: object,
    config: dict | None = None,
) -> list[dict]:
    """Build HA conditions from a DetectionResult and area context.

    Conditions are additive — each check appends if applicable:
    1. Weekday condition (from day_type)
    2. Presence condition (if person entity in area)
    3. Illuminance condition (if light action + illuminance sensor in area)

    Args:
        detection: The detection result.
        entity_graph: EntityGraph for area entity lookups.
        config: Optional config overrides.

    Returns:
        List of HA-compatible condition dicts.
    """
    conditions: list[dict] = []

    # 1. Weekday condition
    _add_weekday_condition(detection.day_type, conditions)

    # 2. Area-based conditions (presence, illuminance)
    area_entities = _get_area_entities(detection.area_id, entity_graph)
    _add_presence_condition(area_entities, conditions)

    if _is_light_action(detection.action_entities):
        _add_illuminance_condition(area_entities, conditions)

    return conditions


def _add_weekday_condition(day_type: str, conditions: list[dict]) -> None:
    """Add weekday condition if day_type maps to specific days."""
    weekdays = DAY_TYPE_WEEKDAYS.get(day_type)
    if weekdays:
        conditions.append(
            {
                "condition": "time",
                "weekday": weekdays,
            }
        )


def _add_presence_condition(area_entities: list[str], conditions: list[dict]) -> None:
    """Add presence condition if a person entity exists in the area."""
    for entity_id in area_entities:
        if entity_id.startswith("person."):
            conditions.append(
                {
                    "condition": "state",
                    "entity_id": entity_id,
                    "state": _quote_state("home"),
                }
            )
            break  # One person condition is enough


def _add_illuminance_condition(
    area_entities: list[str],
    conditions: list[dict],
    threshold: int = 50,
) -> None:
    """Add illuminance condition if an illuminance sensor exists in the area."""
    for entity_id in area_entities:
        if "illuminance" in entity_id and entity_id.startswith("sensor."):
            conditions.append(
                {
                    "condition": "numeric_state",
                    "entity_id": entity_id,
                    "below": threshold,
                }
            )
            break  # One illuminance condition is enough


def _is_light_action(action_entities: list[str]) -> bool:
    """Check if any action entity is a light."""
    return any(e.startswith("light.") for e in action_entities)


def _get_area_entities(area_id: str | None, entity_graph: object) -> list[str]:
    """Get entity IDs in an area from the entity graph.

    EntityGraph.entities_in_area returns list[dict] with 'entity_id' keys.
    This helper extracts the string IDs for pattern matching.
    """
    if not area_id:
        return []
    try:
        raw = entity_graph.entities_in_area(area_id)  # type: ignore[union-attr]
        result = []
        for item in raw:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                eid = item.get("entity_id", "")
                if eid:
                    result.append(eid)
        return result
    except Exception as exc:
        logger.warning("Failed to resolve entities in area %s: %s", area_id, exc)
        return []
