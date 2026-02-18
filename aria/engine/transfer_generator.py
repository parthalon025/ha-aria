"""Cross-domain pattern transfer hypothesis generator.

Scans organic discovery capabilities and finds pairs with structural
similarity. Generates TransferCandidate objects for the transfer engine
hub module to test via shadow engine.

Tier 3+ only — called periodically by TransferEngineModule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aria.engine.transfer import (
    MIN_SIMILARITY,
    TransferCandidate,
    TransferType,
    compute_jaccard_similarity,
)

logger = logging.getLogger(__name__)


def _extract_domain_set(entity_ids: list[str], entities_cache: dict) -> set[str]:
    """Extract the set of entity domains for a capability's entities."""
    domains = set()
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        domain = entity.get("domain", "")
        if domain:
            domains.add(domain)
    return domains


def _extract_device_class_set(entity_ids: list[str], entities_cache: dict) -> set[str]:
    """Extract the set of device classes for a capability's entities."""
    classes = set()
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        dc = entity.get("device_class", "")
        if dc:
            classes.add(dc)
    return classes


def _extract_area(entity_ids: list[str], entities_cache: dict) -> str | None:
    """Find the dominant area for a capability's entities."""
    area_counts: dict[str, int] = {}
    for eid in entity_ids:
        entity = entities_cache.get(eid, {})
        area = entity.get("area_id", "")
        if area:
            area_counts[area] = area_counts.get(area, 0) + 1
    if not area_counts:
        return None
    return max(area_counts, key=area_counts.get)


def _compute_structural_similarity(
    cap_a_entities: list[str],
    cap_b_entities: list[str],
    entities_cache: dict,
) -> float:
    """Compute structural similarity between two capabilities.

    Blends domain Jaccard (weight 0.6) and device class Jaccard (weight 0.4).
    """
    domains_a = _extract_domain_set(cap_a_entities, entities_cache)
    domains_b = _extract_domain_set(cap_b_entities, entities_cache)
    classes_a = _extract_device_class_set(cap_a_entities, entities_cache)
    classes_b = _extract_device_class_set(cap_b_entities, entities_cache)

    domain_sim = compute_jaccard_similarity(domains_a, domains_b)
    class_sim = compute_jaccard_similarity(classes_a, classes_b)

    # Weight domains more heavily — device class is a refinement
    return domain_sim * 0.6 + class_sim * 0.4


@dataclass(frozen=True)
class _PairContext:
    """Bundles source/target data for transfer candidate creation."""

    source_name: str
    target_name: str
    source_entities: list[str]
    target_entities: list[str]
    similarity: float


def _try_routine_transfer(
    ctx: _PairContext,
    source_data: dict,
    target_data: dict,
) -> TransferCandidate | None:
    """Try to create a routine-to-routine transfer candidate."""
    source_peaks = source_data.get("temporal_pattern", {}).get("peak_hours", [])
    target_peaks = target_data.get("temporal_pattern", {}).get("peak_hours", [])

    if not source_peaks or not target_peaks:
        return None

    offset_minutes = int((sum(target_peaks) / len(target_peaks) - sum(source_peaks) / len(source_peaks)) * 60)

    if offset_minutes == 0:
        return None

    try:
        return TransferCandidate(
            source_capability=ctx.source_name,
            target_context=ctx.target_name,
            transfer_type=TransferType.ROUTINE_TO_ROUTINE,
            similarity_score=ctx.similarity,
            source_entities=ctx.source_entities,
            target_entities=ctx.target_entities,
            timing_offset_minutes=offset_minutes,
        )
    except ValueError:
        return None


def _try_room_transfer(ctx: _PairContext, entities_cache: dict) -> TransferCandidate | None:
    """Try to create a room-to-room transfer candidate."""
    source_area = _extract_area(ctx.source_entities, entities_cache)
    target_area = _extract_area(ctx.target_entities, entities_cache)

    if source_area and target_area and source_area == target_area:
        return None

    try:
        return TransferCandidate(
            source_capability=ctx.source_name,
            target_context=target_area or ctx.target_name,
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=ctx.similarity,
            source_entities=ctx.source_entities,
            target_entities=ctx.target_entities,
        )
    except ValueError:
        return None


def generate_transfer_candidates(
    capabilities: dict[str, dict[str, Any]],
    entities_cache: dict[str, dict[str, Any]],
    min_similarity: float = 0.6,
) -> list[TransferCandidate]:
    """Generate transfer candidates from organic capabilities.

    Rules:
    - Only promoted organic capabilities can be sources
    - Seed capabilities are never sources
    - Room-to-room: domain-layer caps in different areas with similar composition
    - Routine-to-routine: behavioral-layer caps with overlapping entity sets
      but different temporal patterns

    Args:
        capabilities: Capabilities cache dict (name -> data).
        entities_cache: Entities cache dict (entity_id -> entity data).
        min_similarity: Minimum structural similarity for candidate generation.

    Returns:
        List of TransferCandidate objects.
    """
    candidates: list[TransferCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Filter to promoted organic capabilities
    promoted = {
        name: data
        for name, data in capabilities.items()
        if data.get("source") != "seed" and data.get("status") == "promoted"
    }

    # All organic capabilities (promoted or candidate) as potential targets
    all_organic = {name: data for name, data in capabilities.items() if data.get("source") != "seed"}

    for source_name, source_data in promoted.items():
        source_entities = source_data.get("entities", [])
        source_layer = source_data.get("layer", "domain")

        for target_name, target_data in all_organic.items():
            if target_name == source_name:
                continue

            pair_key = (source_name, target_name)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            target_entities = target_data.get("entities", [])
            target_layer = target_data.get("layer", "domain")

            # Compute structural similarity
            similarity = _compute_structural_similarity(source_entities, target_entities, entities_cache)

            if similarity < max(min_similarity, MIN_SIMILARITY):
                continue

            # Determine transfer type and create candidate
            ctx = _PairContext(source_name, target_name, source_entities, target_entities, similarity)
            if source_layer == "behavioral" and target_layer == "behavioral":
                candidate = _try_routine_transfer(ctx, source_data, target_data)
            else:
                candidate = _try_room_transfer(ctx, entities_cache)

            if candidate is not None:
                candidates.append(candidate)

    logger.info(f"Generated {len(candidates)} transfer candidates from {len(promoted)} promoted capabilities")
    return candidates
