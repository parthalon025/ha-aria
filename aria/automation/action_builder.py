"""Action builder — service selection, area targeting, restricted domain check.

Generates HA action dicts from DetectionResult action entities and area context.
Prefers area targeting when all entities are in the same area.
"""

import logging

from aria.automation.models import DetectionResult

logger = logging.getLogger(__name__)

# Domains that require explicit approval before automation.
RESTRICTED_DOMAINS = frozenset({"lock", "alarm_control_panel", "cover"})

# Domain → (on_service, off_service) mapping.
# on_service is used when the detected state is positive (on/playing/locked/etc.).
DOMAIN_SERVICE_MAP = {
    "light": ("light.turn_on", "light.turn_off"),
    "switch": ("switch.turn_on", "switch.turn_off"),
    "fan": ("fan.turn_on", "fan.turn_off"),
    "media_player": ("media_player.media_play", "media_player.media_stop"),
    "climate": ("climate.set_hvac_mode", "climate.turn_off"),
    "cover": ("cover.open_cover", "cover.close_cover"),
    "lock": ("lock.lock", "lock.unlock"),
    "alarm_control_panel": ("alarm_control_panel.alarm_arm_away", "alarm_control_panel.alarm_disarm"),
    "input_boolean": ("input_boolean.turn_on", "input_boolean.turn_off"),
}

# States considered "off" or "negative" for service selection
OFF_STATES = {"off", "closed", "unlocked", "disarmed", "idle", "paused", "standby", "not_home"}


def build_actions(
    detection: DetectionResult,
    entity_graph: object,
) -> list[dict]:
    """Build HA action dicts from a DetectionResult.

    Groups action entities by domain and prefers area targeting when
    all entities of a domain are in the detection's area.

    Args:
        detection: The detection result with action entities and chain.
        entity_graph: EntityGraph for area entity lookups.

    Returns:
        List of HA-compatible action dicts.
    """
    actions: list[dict] = []

    # Build entity → state mapping from chain
    entity_states = _extract_entity_states(detection)

    # Group action entities by domain
    domain_groups = _group_by_domain(detection.action_entities)

    # Get area entities for area targeting decisions
    area_entities = _get_area_entities(detection.area_id, entity_graph)

    for domain, entities in domain_groups.items():
        # Determine service from first entity's state
        first_entity = entities[0]
        state = entity_states.get(first_entity, "on")
        service = _select_service(domain, state)

        # Build target — prefer area if available and entities are in it
        target = _build_target(entities, detection.area_id, area_entities)

        action: dict = {
            "action": service,
            "target": target,
        }

        # Flag restricted domains
        if domain in RESTRICTED_DOMAINS:
            action["_restricted"] = True

        actions.append(action)

    return actions


def _extract_entity_states(detection: DetectionResult) -> dict[str, str]:
    """Extract entity → state mapping from the detection chain."""
    states: dict[str, str] = {}
    for link in detection.entity_chain:
        states[link.entity_id] = link.state
    return states


def _group_by_domain(action_entities: list[str]) -> dict[str, list[str]]:
    """Group action entities by their HA domain prefix."""
    groups: dict[str, list[str]] = {}
    for entity_id in action_entities:
        domain = entity_id.split(".")[0]
        groups.setdefault(domain, []).append(entity_id)
    return groups


def _select_service(domain: str, state: str) -> str:
    """Select the appropriate HA service for a domain and state."""
    services = DOMAIN_SERVICE_MAP.get(domain)
    if not services:
        # Fallback: homeassistant.turn_on/turn_off
        if state.lower() in OFF_STATES:
            return "homeassistant.turn_off"
        return "homeassistant.turn_on"

    on_service, off_service = services
    if state.lower() in OFF_STATES:
        return off_service
    return on_service


def _build_target(
    entities: list[str],
    area_id: str | None,
    area_entities: list[str],
) -> dict:
    """Build action target — area if possible, entity list otherwise."""
    if area_id and _all_entities_in_area(entities, area_entities):
        return {"area_id": area_id}

    if len(entities) == 1:
        return {"entity_id": entities[0]}

    return {"entity_id": entities}


def _all_entities_in_area(entities: list[str], area_entities: list[str]) -> bool:
    """Check if all action entities are present in the area entity list."""
    area_set = set(area_entities)
    return all(e in area_set for e in entities)


def _get_area_entities(area_id: str | None, entity_graph: object) -> list[str]:
    """Get entity IDs in an area from the entity graph.

    EntityGraph.entities_in_area returns list[dict] with 'entity_id' keys.
    This helper extracts the string IDs.
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
