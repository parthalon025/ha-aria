"""Automation validator — 9-check validation suite for generated automations.

Each check is a separate method for independent testability. All checks
run sequentially, collecting errors rather than short-circuiting, so
callers see the full picture.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Required top-level keys for a valid HA automation.
REQUIRED_FIELDS = ("id", "alias", "triggers", "actions")

# Required non-empty list fields.
REQUIRED_LISTS = ("triggers", "actions")

# Restricted domains that need explicit approval.
RESTRICTED_DOMAINS = frozenset({"lock", "alarm_control_panel", "cover"})

# Mode requirements per action domain prefix.
DOMAIN_MODE_REQUIREMENTS: dict[str, str] = {
    "notify": "queued",
    "scene": "parallel",
}


def validate_automation(
    automation: Any,
    entity_graph: object,
    existing_ids: set[str],
) -> tuple[bool, list[str]]:
    """Run all 9 validation checks on a generated automation.

    Args:
        automation: The automation dict to validate.
        entity_graph: EntityGraph for entity existence checks.
        existing_ids: Set of existing automation IDs for collision check.

    Returns:
        Tuple of (valid: bool, errors: list[str]).
    """
    errors: list[str] = []

    # Check 1: Must be a dict (YAML-parseable proxy)
    errors.extend(_check_yaml_parseable(automation))
    if not isinstance(automation, dict):
        return False, errors

    # Check 2: Required fields
    errors.extend(_check_required_fields(automation))

    # Check 3: State values quoted (no booleans)
    errors.extend(_check_state_quoting(automation))

    # Check 4: Entities exist
    errors.extend(_check_entities_exist(automation, entity_graph))

    # Check 5: Services valid
    errors.extend(_check_services_valid(automation))

    # Check 6: No circular triggers
    errors.extend(_check_no_circular_trigger(automation))

    # Check 7: No duplicate IDs
    errors.extend(_check_no_duplicate_id(automation, existing_ids))

    # Check 8: Mode appropriate
    errors.extend(_check_mode_appropriate(automation))

    # Check 9: Restricted domain approval
    errors.extend(_check_restricted_domains(automation))

    return len(errors) == 0, errors


def _check_yaml_parseable(automation: Any) -> list[str]:
    """Check 1: Automation must be a dict (valid structure)."""
    if not isinstance(automation, dict):
        return [f"Automation must be a dict, got {type(automation).__name__}"]
    return []


def _check_required_fields(automation: dict) -> list[str]:
    """Check 2: Required fields must be present and non-empty."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in automation:
            errors.append(f"Missing required field: {field}")
        elif field in REQUIRED_LISTS:
            val = automation[field]
            if not isinstance(val, list) or len(val) == 0:
                errors.append(f"Field '{field}' must be a non-empty list")
    return errors


def _check_state_quoting(automation: dict) -> list[str]:
    """Check 3: Boolean-like values must be strings, not Python booleans.

    YAML parses on/off/true/false as booleans. HA expects string values.
    Walk the entire automation dict tree looking for bool instances in
    state-relevant positions.
    """
    errors: list[str] = []
    _walk_for_booleans(automation, "automation", errors)
    return errors


def _walk_for_booleans(obj: Any, path: str, errors: list[str]) -> None:
    """Recursively walk a dict/list tree checking for boolean values."""
    if isinstance(obj, bool):
        suggested = "on" if obj else "off"
        errors.append(f"Boolean value at {path} — must be quoted string ('{suggested}')")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _walk_for_booleans(value, f"{path}.{key}", errors)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk_for_booleans(item, f"{path}[{i}]", errors)


def _check_entities_exist(automation: dict, entity_graph: object) -> list[str]:
    """Check 4: All referenced entity_ids must exist in EntityGraph."""
    if entity_graph is None:
        logger.warning("entity_graph is None in validate — entity existence check skipped")
        return []

    errors = []
    entity_ids = _collect_entity_ids(automation)

    for entity_id in entity_ids:
        if not entity_graph.has_entity(entity_id):  # type: ignore[attr-defined]
            errors.append(f"Entity not found: {entity_id}")

    return errors


def _collect_entity_ids(automation: dict) -> list[str]:
    """Extract all entity_id references from triggers, conditions, and actions."""
    entity_ids: list[str] = []

    for trigger in automation.get("triggers", []):
        _extract_entity_id(trigger, entity_ids)

    for condition in automation.get("conditions", []):
        _extract_entity_id(condition, entity_ids)

    for action in automation.get("actions", []):
        target = action.get("target", {})
        if isinstance(target, dict):
            _extract_entity_id(target, entity_ids)

    return entity_ids


def _extract_entity_id(obj: dict, out: list[str]) -> None:
    """Extract entity_id from a dict, handling str or list values."""
    eid = obj.get("entity_id")
    if isinstance(eid, str):
        out.append(eid)
    elif isinstance(eid, list):
        out.extend(eid)


def _check_services_valid(automation: dict) -> list[str]:
    """Check 5: Service names must follow domain.service format."""
    errors = []
    for action in automation.get("actions", []):
        service = action.get("action") or action.get("service")
        if not service:
            continue
        if "." not in service:
            errors.append(f"Invalid service format: '{service}' (expected domain.service)")
    return errors


def _check_no_circular_trigger(automation: dict) -> list[str]:
    """Check 6: Trigger entities must not appear in action targets."""
    trigger_entities = set()
    for trigger in automation.get("triggers", []):
        eid = trigger.get("entity_id")
        if isinstance(eid, str):
            trigger_entities.add(eid)
        elif isinstance(eid, list):
            trigger_entities.update(eid)

    action_entities = set()
    for action in automation.get("actions", []):
        target = action.get("target", {})
        if isinstance(target, dict):
            eid = target.get("entity_id")
            if isinstance(eid, str):
                action_entities.add(eid)
            elif isinstance(eid, list):
                action_entities.update(eid)

    overlap = trigger_entities & action_entities
    if overlap:
        return [f"Circular trigger detected: {', '.join(sorted(overlap))} appears in both triggers and actions"]
    return []


def _check_no_duplicate_id(automation: dict, existing_ids: set[str]) -> list[str]:
    """Check 7: Automation ID must not collide with existing ones."""
    auto_id = automation.get("id", "")
    if auto_id in existing_ids:
        return [f"Duplicate automation ID: '{auto_id}'"]
    return []


def _check_mode_appropriate(automation: dict) -> list[str]:
    """Check 8: Mode must match action domain requirements."""
    mode = automation.get("mode", "single")
    errors = []

    for action in automation.get("actions", []):
        service = action.get("action") or action.get("service") or ""
        domain = service.split(".")[0] if "." in service else ""

        required_mode = DOMAIN_MODE_REQUIREMENTS.get(domain)
        if required_mode and mode != required_mode:
            errors.append(f"Action domain '{domain}' requires mode '{required_mode}', got '{mode}'")
            break  # One mode error is enough

    return errors


def _check_restricted_domains(automation: dict) -> list[str]:
    """Check 9: Restricted domains need ARIA_REQUIRES_APPROVAL in description."""
    restricted_found = []
    for action in automation.get("actions", []):
        service = action.get("action") or action.get("service") or ""
        domain = service.split(".")[0] if "." in service else ""
        if domain in RESTRICTED_DOMAINS:
            restricted_found.append(domain)

    if not restricted_found:
        return []

    description = automation.get("description", "")
    if "ARIA_REQUIRES_APPROVAL" in description:
        return []

    domains_str = ", ".join(sorted(set(restricted_found)))
    return [f"Restricted domain(s) [{domains_str}] require 'ARIA_REQUIRES_APPROVAL' in description for human oversight"]
