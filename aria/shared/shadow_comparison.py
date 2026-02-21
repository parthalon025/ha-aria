"""Shadow Comparison Engine — Compare candidate automations against existing HA automations.

Detects duplicates (exact/superset/subset), conflicts (opposite actions,
parameter differences), and gap fills (cross-area coverage) using
EntityGraph for area-to-entity resolution.
"""

import logging
from dataclasses import dataclass
from typing import Any

from aria.automation.models import ShadowResult
from aria.shared.entity_graph import EntityGraph

logger = logging.getLogger(__name__)

# Service pairs that are considered opposite actions
OPPOSITE_SERVICES: dict[str, str] = {
    "light.turn_on": "light.turn_off",
    "light.turn_off": "light.turn_on",
    "switch.turn_on": "switch.turn_off",
    "switch.turn_off": "switch.turn_on",
    "fan.turn_on": "fan.turn_off",
    "fan.turn_off": "fan.turn_on",
    "cover.open_cover": "cover.close_cover",
    "cover.close_cover": "cover.open_cover",
    "lock.lock": "lock.unlock",
    "lock.unlock": "lock.lock",
}

# Numeric parameter keys to check for parameter conflicts
NUMERIC_PARAMS = {"brightness_pct", "brightness", "color_temp", "temperature", "position"}

# Minimum absolute difference to flag as conflict (percentage or degrees)
PARAM_CONFLICT_THRESHOLD = 20


@dataclass
class _MatchState:
    """Accumulated match state across all existing automations."""

    best_match: dict[str, Any] | None = None
    best_score: float = 0.0
    conflict_match: dict[str, Any] | None = None
    conflict_reason: str = ""
    gap_source: dict[str, Any] | None = None


def compare_candidate(
    candidate: dict[str, Any],
    ha_automations: list[dict[str, Any]],
    entity_graph: EntityGraph | None,
) -> ShadowResult:
    """Compare a candidate automation against existing HA automations.

    Args:
        candidate: Generated HA automation dict.
        ha_automations: List of existing HA automation dicts from cache.
        entity_graph: EntityGraph for area-to-entity resolution (may be None).

    Returns:
        ShadowResult annotation.
    """
    if not ha_automations:
        return _new_result(candidate, "No existing HA automations to compare against.")

    cand_triggers = _extract_trigger_signature(candidate)
    cand_targets = _extract_target_entities(candidate, entity_graph)
    cand_services = _extract_services(candidate)
    cand_areas = _extract_areas(candidate, entity_graph)
    state = _MatchState()

    for existing in ha_automations:
        result = _compare_one(
            candidate,
            existing,
            entity_graph,
            cand_triggers,
            cand_targets,
            cand_services,
            cand_areas,
            state,
        )
        if result is not None:
            return result  # Early return for disabled automation match

    return _build_final_result(candidate, entity_graph, cand_targets, state)


def _compare_one(  # noqa: PLR0913
    candidate: dict[str, Any],
    existing: dict[str, Any],
    entity_graph: EntityGraph | None,
    cand_triggers: set[str],
    cand_targets: set[str],
    cand_services: set[str],
    cand_areas: set[str],
    state: _MatchState,
) -> ShadowResult | None:
    """Compare candidate against one existing automation. Updates state in-place.

    Returns a ShadowResult for early exit (disabled match) or None to continue.
    """
    ex_triggers = _extract_trigger_signature(existing)
    trigger_overlap = _trigger_similarity(cand_triggers, ex_triggers)

    if trigger_overlap < 0.5:
        return None

    if existing.get("enabled") is False:
        alias = existing.get("alias", existing.get("id", "unknown"))
        return _new_result(
            candidate,
            f"Improved version of disabled automation '{alias}'.",
            gap_source=existing.get("id"),
        )

    ex_targets = _extract_target_entities(existing, entity_graph)
    ex_services = _extract_services(existing)
    ex_areas = _extract_areas(existing, entity_graph)

    _check_conflicts(candidate, existing, cand_services, ex_services, state)
    dup_score = _compute_duplicate_score(
        trigger_overlap,
        cand_targets,
        ex_targets,
        cand_services,
        ex_services,
        cand_areas,
        ex_areas,
    )

    if dup_score > state.best_score:
        state.best_score = dup_score
        state.best_match = existing

    if trigger_overlap >= 0.8 and cand_areas and ex_areas and not cand_areas.intersection(ex_areas):
        state.gap_source = existing

    return None


def _check_conflicts(
    candidate: dict[str, Any],
    existing: dict[str, Any],
    cand_services: set[str],
    ex_services: set[str],
    state: _MatchState,
) -> None:
    """Check for opposite-action and parameter conflicts. Updates state in-place."""
    if state.conflict_match:
        return

    for cs in cand_services:
        opposite = OPPOSITE_SERVICES.get(cs)
        if opposite and opposite in ex_services:
            state.conflict_match = existing
            alias = existing.get("alias", "")
            state.conflict_reason = f"Opposite action: candidate uses {cs}, existing '{alias}' uses {opposite}."
            return

    param_conflict = _check_parameter_conflict(candidate, existing)
    if param_conflict:
        state.conflict_match = existing
        state.conflict_reason = param_conflict


def _compute_duplicate_score(  # noqa: PLR0913
    trigger_overlap: float,
    cand_targets: set[str],
    ex_targets: set[str],
    cand_services: set[str],
    ex_services: set[str],
    cand_areas: set[str],
    ex_areas: set[str],
) -> float:
    """Compute weighted duplicate score from trigger, target, and service overlaps."""
    if cand_targets and ex_targets:
        target_score = _set_overlap_score(cand_targets, ex_targets)
    elif cand_areas and ex_areas:
        target_score = _set_overlap_score(cand_areas, ex_areas)
    elif not cand_targets and not ex_targets and cand_areas == ex_areas:
        target_score = 1.0
    else:
        target_score = 0.0

    service_score = _set_overlap_score(cand_services, ex_services) if cand_services and ex_services else 0.0
    return trigger_overlap * 0.4 + target_score * 0.4 + service_score * 0.2


def _build_final_result(
    candidate: dict[str, Any],
    entity_graph: EntityGraph | None,
    cand_targets: set[str],
    state: _MatchState,
) -> ShadowResult:
    """Build the final ShadowResult from accumulated match state."""
    if state.conflict_match:
        return ShadowResult(
            candidate=candidate,
            status="conflict",
            duplicate_score=state.best_score,
            conflicting_automation=state.conflict_match.get("id"),
            gap_source_automation=None,
            reason=state.conflict_reason,
        )

    if state.best_score >= 0.8 and state.best_match:
        return _classify_duplicate(candidate, entity_graph, cand_targets, state)

    if state.gap_source:
        alias = state.gap_source.get("alias", "")
        return ShadowResult(
            candidate=candidate,
            status="gap_fill",
            duplicate_score=state.best_score,
            conflicting_automation=None,
            gap_source_automation=state.gap_source.get("id"),
            reason=f"Fills coverage gap — extends trigger from '{alias}' to new area.",
        )

    return _new_result(candidate, "No matching existing automation found.", score=state.best_score)


def _classify_duplicate(
    candidate: dict[str, Any],
    entity_graph: EntityGraph | None,
    cand_targets: set[str],
    state: _MatchState,
) -> ShadowResult:
    """Classify a high-score match as exact duplicate, subset, or superset."""
    best = state.best_match
    alias = best.get("alias", best.get("id", "unknown"))

    if cand_targets:
        ex_targets = _extract_target_entities(best, entity_graph)
        if ex_targets:
            if cand_targets < ex_targets:
                return ShadowResult(
                    candidate=candidate,
                    status="duplicate",
                    duplicate_score=state.best_score,
                    conflicting_automation=None,
                    gap_source_automation=None,
                    reason=f"Subset of existing automation '{alias}'.",
                )
            if cand_targets > ex_targets:
                return _new_result(
                    candidate,
                    f"Expands on existing automation '{alias}'.",
                    score=state.best_score,
                    gap_source=best.get("id"),
                )

    return ShadowResult(
        candidate=candidate,
        status="duplicate",
        duplicate_score=state.best_score,
        conflicting_automation=None,
        gap_source_automation=None,
        reason=f"Duplicate of existing automation '{alias}'.",
    )


def _new_result(
    candidate: dict[str, Any],
    reason: str,
    score: float = 0.0,
    gap_source: str | None = None,
) -> ShadowResult:
    """Build a 'new' ShadowResult."""
    return ShadowResult(
        candidate=candidate,
        status="new",
        duplicate_score=score,
        conflicting_automation=None,
        gap_source_automation=gap_source,
        reason=reason,
    )


# ============================================================================
# Extraction helpers
# ============================================================================


def _extract_trigger_signature(automation: dict[str, Any]) -> set[str]:
    """Extract a set of trigger signatures for comparison."""
    signatures = set()
    for trigger in automation.get("triggers") or automation.get("trigger", []):
        platform = trigger.get("platform", "")
        if platform == "state":
            entity_id = trigger.get("entity_id", "")
            to_state = trigger.get("to", "")
            signatures.add(f"state:{entity_id}:{to_state}")
        elif platform == "time":
            signatures.add(f"time:{trigger.get('at', '')}")
        elif platform == "numeric_state":
            entity_id = trigger.get("entity_id", "")
            signatures.add(f"numeric_state:{entity_id}:{trigger.get('above', '')}:{trigger.get('below', '')}")
        else:
            signatures.add(f"{platform}:{trigger.get('entity_id', '')}")
    return signatures


def _extract_target_entities(
    automation: dict[str, Any],
    entity_graph: EntityGraph | None,
) -> set[str]:
    """Extract the set of target entity IDs, resolving area_id via EntityGraph."""
    entities = set()
    for action in automation.get("actions") or automation.get("action", []):
        target = action.get("target", {})
        if not target:
            continue
        _collect_entity_ids(target, entities)
        _resolve_area_entities(target, action, entity_graph, entities)
    return entities


def _collect_entity_ids(target: dict[str, Any], entities: set[str]) -> None:
    """Add entity_id (str or list) from target to the set."""
    entity_id = target.get("entity_id")
    if isinstance(entity_id, str):
        entities.add(entity_id)
    elif isinstance(entity_id, list):
        entities.update(entity_id)


def _resolve_area_entities(
    target: dict[str, Any],
    action: dict[str, Any],
    entity_graph: EntityGraph | None,
    entities: set[str],
) -> None:
    """Resolve area_id to entity list via EntityGraph and add to set."""
    area_id = target.get("area_id")
    if not area_id or not entity_graph:
        return
    service = action.get("service", "") or action.get("action", "")
    domain = service.split(".")[0] if "." in service else ""
    for ent in entity_graph.entities_in_area(area_id):
        eid = ent.get("entity_id", "")
        if (domain and eid.startswith(f"{domain}.")) or not domain:
            entities.add(eid)


def _extract_services(automation: dict[str, Any]) -> set[str]:
    """Extract the set of service calls from an automation."""
    return {
        (action.get("service", "") or action.get("action", ""))
        for action in (automation.get("actions") or automation.get("action", []))
        if action.get("service") or action.get("action")
    }


def _extract_areas(
    automation: dict[str, Any],
    entity_graph: EntityGraph | None,
) -> set[str]:
    """Extract the set of target areas from an automation."""
    areas = set()
    for action in automation.get("actions") or automation.get("action", []):
        target = action.get("target", {})
        if not target:
            continue
        area_id = target.get("area_id")
        if area_id:
            areas.add(area_id)
        _resolve_entity_areas(target, entity_graph, areas)
    return areas


def _resolve_entity_areas(
    target: dict[str, Any],
    entity_graph: EntityGraph | None,
    areas: set[str],
) -> None:
    """Resolve entity_id → area via EntityGraph and add to areas set."""
    entity_id = target.get("entity_id")
    if not entity_id or not entity_graph:
        return
    ids = [entity_id] if isinstance(entity_id, str) else entity_id
    for eid in ids:
        area = entity_graph.get_area(eid)
        if area:
            areas.add(area)


def _trigger_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two trigger signature sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _set_overlap_score(set_a: set[str], set_b: set[str]) -> float:
    """Compute overlap score: |intersection| / max(|a|, |b|)."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    max_size = max(len(set_a), len(set_b))
    return len(intersection) / max_size if max_size > 0 else 0.0


def _check_parameter_conflict(
    candidate: dict[str, Any],
    existing: dict[str, Any],
) -> str | None:
    """Check if two automations have conflicting numeric parameters.

    Returns a conflict reason string or None if no conflict.
    """
    for cand_action in candidate.get("actions") or candidate.get("action", []):
        cand_service = cand_action.get("service", "") or cand_action.get("action", "")
        cand_data = cand_action.get("data", {})
        if not cand_data:
            continue
        result = _check_action_params(cand_service, cand_data, existing)
        if result:
            return result
    return None


def _check_action_params(
    cand_service: str,
    cand_data: dict[str, Any],
    existing: dict[str, Any],
) -> str | None:
    """Check one candidate action's params against all existing actions."""
    for ex_action in existing.get("actions") or existing.get("action", []):
        if (ex_action.get("service", "") or ex_action.get("action", "")) != cand_service:
            continue
        ex_data = ex_action.get("data", {})
        if not ex_data:
            continue
        for param in NUMERIC_PARAMS:
            cand_val = cand_data.get(param)
            ex_val = ex_data.get(param)
            if cand_val is not None and ex_val is not None:
                try:
                    diff = abs(float(cand_val) - float(ex_val))
                    if diff > PARAM_CONFLICT_THRESHOLD:
                        alias = existing.get("alias", "")
                        return (
                            f"Parameter conflict: {param} differs by {diff:.0f} "
                            f"(candidate={cand_val}, existing '{alias}' has {ex_val})."
                        )
                except (TypeError, ValueError):
                    continue
    return None
