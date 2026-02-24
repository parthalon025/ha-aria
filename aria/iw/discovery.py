"""Discovery engine — converts pattern and gap analysis into behavioral state definitions.

Three-stage pipeline:
  Stage 1 (Gather): Call patterns.detect_patterns() + anomaly_gap.analyze_gaps()
  Stage 2 (Build):  Construct indicator chains → BehavioralStateDefinition
  Stage 3 (Dedup):  Merge overlapping definitions (>60% entity overlap)

Runs periodically (default: every 6 hours, configurable via iw.discovery_interval_hours).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from aria.automation.models import DetectionResult
from aria.iw.models import BehavioralStateDefinition, Indicator

logger = logging.getLogger(__name__)

# Minimum gap (in seconds) before trigger to generate a quiet_period precondition
_QUIET_PERIOD_THRESHOLD_SECONDS = 4 * 3600  # 4 hours

# Jaccard similarity threshold for merging two definitions
_MERGE_OVERLAP_THRESHOLD = 0.60


class DiscoveryEngine:
    """Batch discovery: patterns + gaps → BehavioralStateDefinitions."""

    def __init__(self, hub: Any) -> None:
        self.hub = hub

    async def discover(self) -> list[BehavioralStateDefinition]:
        """Run the three-stage discovery pipeline.

        Returns:
            New BehavioralStateDefinitions discovered this run.
        """
        # Stage 1: Gather sources
        patterns, gaps = await self._gather_sources()
        if not patterns and not gaps:
            return []

        # Read config threshold
        min_confidence = await self.hub.cache.get_config_value("iw.min_discovery_confidence", 0.60)

        # Stage 2: Build indicator chains
        definitions = self._build_from_patterns(patterns, min_confidence)
        definitions.extend(self._build_from_gaps(gaps, min_confidence))

        if not definitions:
            return []

        # Stage 3: Deduplicate
        return self._deduplicate(definitions)

    # ------------------------------------------------------------------
    # Stage 1: Gather
    # ------------------------------------------------------------------

    async def _gather_sources(
        self,
    ) -> tuple[list[dict[str, Any]], list[DetectionResult]]:
        """Collect raw data from patterns module and gap analyzer."""
        patterns: list[dict[str, Any]] = []
        gaps: list[DetectionResult] = []

        patterns_mod = self.hub.modules.get("patterns")
        if patterns_mod is not None:
            try:
                patterns = await patterns_mod.detect_patterns()
            except Exception:
                logger.exception("patterns.detect_patterns() failed")

        gap_mod = self.hub.modules.get("anomaly_gap")
        if gap_mod is not None:
            try:
                gaps = await gap_mod.analyze_gaps()
            except Exception:
                logger.exception("anomaly_gap.analyze_gaps() failed")

        return patterns, gaps

    # ------------------------------------------------------------------
    # Stage 2: Build indicator chains
    # ------------------------------------------------------------------

    def _build_from_patterns(
        self,
        patterns: list[dict[str, Any]],
        min_confidence: float,
    ) -> list[BehavioralStateDefinition]:
        """Convert pattern dicts into BehavioralStateDefinitions."""
        results: list[BehavioralStateDefinition] = []
        for p in patterns:
            if p.get("confidence", 0) < min_confidence:
                continue

            chain = p.get("entity_chain", [])
            if not chain:
                continue

            trigger_entity = p.get("trigger_entity") or chain[0].get("entity_id")
            area = p.get("area", "")
            day_type = p.get("day_type", "")

            # Build trigger indicator from first chain link
            first = chain[0]
            trigger = Indicator(
                entity_id=trigger_entity,
                role="trigger",
                mode="state_change",
                expected_state=first.get("state"),
                max_delay_seconds=0,
                confidence=p.get("confidence", 0.0),
            )

            # Build confirming indicators from remaining chain links
            confirming: list[Indicator] = []
            for link in chain[1:]:
                ind = Indicator(
                    entity_id=link["entity_id"],
                    role="confirming",
                    mode="state_change",
                    expected_state=link.get("state"),
                    max_delay_seconds=int(link.get("offset_seconds", 0)),
                    confidence=p.get("confidence", 0.0),
                )
                confirming.append(ind)

            # Build preconditions (quiet period if applicable)
            preconditions: list[Indicator] = []
            gap_seconds = p.get("gap_before_trigger_seconds")
            if gap_seconds is not None and gap_seconds >= _QUIET_PERIOD_THRESHOLD_SECONDS:
                preconditions.append(
                    Indicator(
                        entity_id=trigger_entity,
                        role="trigger",
                        mode="quiet_period",
                        quiet_seconds=int(gap_seconds),
                    )
                )

            # Compute deterministic ID
            areas = frozenset([area]) if area else frozenset()
            day_types = frozenset([day_type]) if day_type else frozenset()
            confirming_ids = sorted(ind.entity_id for ind in confirming)
            def_id = _deterministic_id(trigger_entity, confirming_ids, areas, day_types)

            # Compute duration from max offset in chain
            max_offset = max((link.get("offset_seconds", 0) for link in chain), default=0)
            duration_minutes = max(max_offset / 60.0, 1.0)

            defn = BehavioralStateDefinition(
                id=def_id,
                name=p.get("name") or p.get("llm_description") or f"Pattern {def_id[:8]}",
                trigger=trigger,
                trigger_preconditions=preconditions,
                confirming=confirming,
                deviations=[],
                areas=areas,
                day_types=day_types,
                person_attribution=None,
                typical_duration_minutes=duration_minutes,
                expected_outcomes=(),
            )
            results.append(defn)

        return results

    def _build_from_gaps(
        self,
        gaps: list[DetectionResult],
        min_confidence: float,
    ) -> list[BehavioralStateDefinition]:
        """Convert gap DetectionResults into BehavioralStateDefinitions."""
        results: list[BehavioralStateDefinition] = []
        for gap in gaps:
            if gap.confidence < min_confidence:
                continue

            if not gap.entity_chain:
                continue

            first_link = gap.entity_chain[0]
            trigger = Indicator(
                entity_id=first_link.entity_id,
                role="trigger",
                mode="state_change",
                expected_state=first_link.state,
                max_delay_seconds=0,
                confidence=gap.confidence,
            )

            confirming: list[Indicator] = []
            for link in gap.entity_chain[1:]:
                ind = Indicator(
                    entity_id=link.entity_id,
                    role="confirming",
                    mode="state_change",
                    expected_state=link.state,
                    max_delay_seconds=int(link.offset_seconds),
                    confidence=gap.confidence,
                )
                confirming.append(ind)

            areas = frozenset([gap.area_id]) if gap.area_id else frozenset()
            day_types = frozenset([gap.day_type]) if gap.day_type else frozenset()
            confirming_ids = sorted(ind.entity_id for ind in confirming)
            def_id = _deterministic_id(gap.trigger_entity, confirming_ids, areas, day_types)

            max_offset = max((link.offset_seconds for link in gap.entity_chain), default=0)
            duration_minutes = max(max_offset / 60.0, 1.0)

            defn = BehavioralStateDefinition(
                id=def_id,
                name=f"Gap: {gap.trigger_entity}",
                trigger=trigger,
                trigger_preconditions=[],
                confirming=confirming,
                deviations=[],
                areas=areas,
                day_types=day_types,
                person_attribution=None,
                typical_duration_minutes=duration_minutes,
                expected_outcomes=(),
            )
            results.append(defn)

        return results

    # ------------------------------------------------------------------
    # Stage 3: Deduplicate
    # ------------------------------------------------------------------

    def _deduplicate(self, definitions: list[BehavioralStateDefinition]) -> list[BehavioralStateDefinition]:
        """Merge definitions with >60% entity overlap. Higher confidence wins."""
        if len(definitions) <= 1:
            return definitions

        merged: list[BehavioralStateDefinition] = []
        used: set[int] = set()

        for i, d1 in enumerate(definitions):
            if i in used:
                continue

            best = d1
            for j in range(i + 1, len(definitions)):
                if j in used:
                    continue
                d2 = definitions[j]
                if _entity_overlap(best, d2) >= _MERGE_OVERLAP_THRESHOLD:
                    # Keep the higher-confidence definition
                    used.add(j)
                    if d2.trigger.confidence > best.trigger.confidence:
                        best = d2

            merged.append(best)

        return merged


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _deterministic_id(
    trigger_entity: str,
    confirming_entity_ids: list[str],
    areas: frozenset[str],
    day_types: frozenset[str],
) -> str:
    """Produce a stable hash ID from the core identity of a definition.

    Same trigger + sorted confirming + sorted areas + sorted day_types → same ID.
    """
    parts = [
        trigger_entity,
        "|".join(sorted(confirming_entity_ids)),
        "|".join(sorted(areas)),
        "|".join(sorted(day_types)),
    ]
    raw = "::".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _entity_overlap(
    d1: BehavioralStateDefinition,
    d2: BehavioralStateDefinition,
) -> float:
    """Jaccard similarity of the full entity sets (trigger + confirming)."""
    set1 = {d1.trigger.entity_id} | {ind.entity_id for ind in d1.confirming}
    set2 = {d2.trigger.entity_id} | {ind.entity_id for ind in d2.confirming}
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)
