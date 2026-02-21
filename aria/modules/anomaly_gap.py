"""Anomaly-Gap Analyzer — detects repetitive manual actions missing automations.

Mines EventStore for manual-only events (context_parent_id IS NULL),
identifies frequently repeated entity sequences and solo toggles,
and produces DetectionResult objects for the automation generator.

Phase 3 component. Uses simplified PrefixSpan for short sequences (2-5
entities) — full algorithm is overkill for home automation chains.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from aria.automation.models import ChainLink, DetectionResult
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)

# Default config values — must match keys in config_defaults.py
DEFAULT_WINDOW_MINUTES = 10
DEFAULT_MIN_OCCURRENCES = 15
DEFAULT_MAX_CHAIN_LENGTH = 5
DEFAULT_MIN_DAYS = 14
DEFAULT_MIN_CONSISTENCY = 0.6


class AnomalyGapAnalyzer(Module):
    """Detects repetitive manual actions that could be automated.

    Two detection modes:
    1. Sequence gaps — entity A then entity B within a time window, repeated N times
    2. Solo gaps — single entity toggled manually at similar times, repeated N times
    """

    def __init__(self, hub: IntelligenceHub):
        super().__init__("anomaly_gap", hub)
        self.window_minutes = DEFAULT_WINDOW_MINUTES
        self.min_observations = DEFAULT_MIN_OCCURRENCES
        self.max_chain_length = DEFAULT_MAX_CHAIN_LENGTH
        self.analysis_days = DEFAULT_MIN_DAYS
        self.min_confidence = DEFAULT_MIN_CONSISTENCY

    async def _load_config(self):
        """Load config values from hub cache.

        Keys must match config_defaults.py exactly (Lesson #45).
        """
        self.window_minutes = await self.hub.cache.get_config_value("gap.window_minutes", DEFAULT_WINDOW_MINUTES)
        self.min_observations = await self.hub.cache.get_config_value("gap.min_occurrences", DEFAULT_MIN_OCCURRENCES)
        self.max_chain_length = await self.hub.cache.get_config_value("gap.max_chain_length", DEFAULT_MAX_CHAIN_LENGTH)
        self.analysis_days = await self.hub.cache.get_config_value("gap.min_days", DEFAULT_MIN_DAYS)
        self.min_confidence = await self.hub.cache.get_config_value("gap.min_consistency", DEFAULT_MIN_CONSISTENCY)

    async def analyze_gaps(self) -> list[DetectionResult]:
        """Run full gap analysis pipeline.

        Returns:
            List of DetectionResult objects with source="gap".
        """
        await self._load_config()

        now = datetime.now(UTC)
        start = (now - timedelta(days=self.analysis_days)).isoformat()
        end = now.isoformat()

        events = await self.hub.event_store.query_manual_events(start, end)
        if not events:
            return []

        results = []

        # Mode 1: Sequence gaps (entity pairs and chains)
        sequence_results = self._mine_sequences(events)
        results.extend(sequence_results)

        # Mode 2: Solo toggle gaps (single entities)
        solo_results = self._mine_solo_toggles(events)
        results.extend(solo_results)

        self.logger.info(
            "Gap analysis complete: %d sequence gaps, %d solo gaps",
            len(sequence_results),
            len(solo_results),
        )
        return results

    def _mine_sequences(self, events: list[dict]) -> list[DetectionResult]:
        """Find frequently repeated entity sequences within time windows.

        Groups events into daily sessions, extracts ordered pairs within
        the configured window, and counts pair frequencies across days.
        """
        window_seconds = self.window_minutes * 60
        daily_sessions = self._group_by_day(events)
        pair_occurrences = self._count_pair_occurrences(daily_sessions, window_seconds)

        return self._build_sequence_results(pair_occurrences, events)

    def _group_by_day(self, events: list[dict]) -> dict[str, list[dict]]:
        """Group events by date (YYYY-MM-DD)."""
        daily: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            ts = event.get("timestamp", "")
            day = ts[:10]  # YYYY-MM-DD
            if day:
                daily[day].append(event)
        return daily

    def _count_pair_occurrences(
        self,
        daily_sessions: dict[str, list[dict]],
        window_seconds: float,
    ) -> dict[tuple[str, str], list[dict]]:
        """Count how many days each (trigger, action) pair occurs.

        Returns mapping of (trigger_entity, action_entity) → list of
        representative event dicts (one per day observed).
        """
        pair_days: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for _day, day_events in daily_sessions.items():
            seen_pairs: set[tuple[str, str]] = set()
            sorted_events = sorted(day_events, key=lambda e: e.get("timestamp", ""))

            for i, event_a in enumerate(sorted_events):
                ts_a = self._parse_ts(event_a.get("timestamp", ""))
                if ts_a is None:
                    continue
                entity_a = event_a.get("entity_id", "")

                for j in range(i + 1, min(i + 20, len(sorted_events))):
                    event_b = sorted_events[j]
                    ts_b = self._parse_ts(event_b.get("timestamp", ""))
                    if ts_b is None:
                        continue

                    delta = (ts_b - ts_a).total_seconds()
                    if delta > window_seconds:
                        break  # sorted — all subsequent are further away
                    if delta < 0:
                        continue

                    entity_b = event_b.get("entity_id", "")
                    if entity_a == entity_b:
                        continue  # skip self-pairs

                    pair = (entity_a, entity_b)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pair_days[pair].append(event_a)

        return pair_days

    def _build_sequence_results(
        self,
        pair_occurrences: dict[tuple[str, str], list[dict]],
        all_events: list[dict],
    ) -> list[DetectionResult]:
        """Convert frequent pairs into DetectionResult objects."""
        results = []

        for (trigger, action), day_events in pair_occurrences.items():
            count = len(day_events)
            if count < self.min_observations:
                continue

            # Compute confidence as ratio of days observed vs total days
            timestamps = [e.get("timestamp", "") for e in day_events]
            first_seen, last_seen = self._temporal_bounds(timestamps)
            total_days = self._days_in_range(first_seen, last_seen)
            confidence = count / max(total_days, 1)

            if confidence < self.min_confidence:
                continue

            area_id = self._resolve_area(trigger, day_events)
            recency_weight = self._compute_recency(last_seen)

            results.append(
                DetectionResult(
                    source="gap",
                    trigger_entity=trigger,
                    action_entities=[action],
                    entity_chain=[
                        ChainLink(entity_id=trigger, state="on", offset_seconds=0),
                        ChainLink(entity_id=action, state="on", offset_seconds=60),
                    ],
                    area_id=area_id,
                    confidence=confidence,
                    recency_weight=recency_weight,
                    observation_count=count,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    day_type="all",  # gap analyzer doesn't segment by day type
                    combined_score=0.0,
                )
            )

        return results

    def _mine_solo_toggles(self, events: list[dict]) -> list[DetectionResult]:
        """Find single entities repeatedly toggled manually.

        Groups by entity_id, counts distinct days, flags entities
        toggled on at least min_observations distinct days.
        """
        entity_days: dict[str, set[str]] = defaultdict(set)
        entity_events: dict[str, list[dict]] = defaultdict(list)

        for event in events:
            entity_id = event.get("entity_id", "")
            new_state = event.get("new_state", "")
            ts = event.get("timestamp", "")
            day = ts[:10]

            # Only count "positive" state transitions
            if new_state in {"on", "open", "unlocked", "home", "playing", "True", "true", "detected"}:
                entity_days[entity_id].add(day)
                entity_events[entity_id].append(event)

        results = []
        for entity_id, days in entity_days.items():
            count = len(days)
            if count < self.min_observations:
                continue

            all_ts = [e.get("timestamp", "") for e in entity_events[entity_id]]
            first_seen, last_seen = self._temporal_bounds(all_ts)
            total_days = self._days_in_range(first_seen, last_seen)
            confidence = count / max(total_days, 1)

            if confidence < self.min_confidence:
                continue

            area_id = self._resolve_area(entity_id, entity_events[entity_id])
            recency_weight = self._compute_recency(last_seen)

            results.append(
                DetectionResult(
                    source="gap",
                    trigger_entity=entity_id,
                    action_entities=[entity_id],
                    entity_chain=[
                        ChainLink(entity_id=entity_id, state="on", offset_seconds=0),
                    ],
                    area_id=area_id,
                    confidence=confidence,
                    recency_weight=recency_weight,
                    observation_count=count,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    day_type="all",
                    combined_score=0.0,
                )
            )

        return results

    # ── Cross-reference HA automations ──────────────────────────────────

    def filter_covered_gaps(
        self,
        gaps: list[DetectionResult],
        ha_automations: list[dict],
    ) -> list[DetectionResult]:
        """Remove gaps already covered by existing HA automations.

        A gap is considered covered if an HA automation:
        - For sequence gaps (chain > 1): has the same trigger entity AND
          at least one matching action entity
        - For solo gaps (chain == 1): has the trigger entity as an action
          target (someone already automated turning it on/off)

        Args:
            gaps: DetectionResult objects from analyze_gaps().
            ha_automations: List of HA automation dicts (from REST API).

        Returns:
            Filtered list with covered gaps removed.
        """
        if not ha_automations:
            return gaps

        # Build lookup: entity_id → set of action entity_ids per automation
        automation_coverage = self._build_automation_coverage(ha_automations)

        filtered = []
        for gap in gaps:
            if self._is_gap_covered(gap, automation_coverage):
                self.logger.debug(
                    "Gap filtered (covered by HA): trigger=%s actions=%s",
                    gap.trigger_entity,
                    gap.action_entities,
                )
                continue
            filtered.append(gap)

        self.logger.info(
            "Cross-reference: %d gaps → %d after filtering (%d covered)",
            len(gaps),
            len(filtered),
            len(gaps) - len(filtered),
        )
        return filtered

    def _build_automation_coverage(
        self,
        ha_automations: list[dict],
    ) -> dict[str, set[str]]:
        """Build trigger→actions mapping from HA automations.

        Returns:
            Dict mapping trigger entity_id → set of action entity_ids.
            Also maps action entity_ids → {"__action_target__"} to flag
            entities that are action targets of any automation.
        """
        coverage: dict[str, set[str]] = defaultdict(set)

        for auto in ha_automations:
            trigger_entities = self._extract_trigger_entities(auto)
            action_entities = self._extract_action_entities(auto)

            for trigger in trigger_entities:
                coverage[trigger].update(action_entities)

            # Also mark action entities as covered targets
            for action in action_entities:
                coverage[action].add("__action_target__")

        return coverage

    def _is_gap_covered(
        self,
        gap: DetectionResult,
        coverage: dict[str, set[str]],
    ) -> bool:
        """Check if a gap is covered by existing automation coverage."""
        if len(gap.entity_chain) > 1:
            # Sequence gap: need trigger match AND action overlap
            trigger_actions = coverage.get(gap.trigger_entity, set())
            if not trigger_actions:
                return False
            return any(a in trigger_actions for a in gap.action_entities)
        else:
            # Solo gap: covered if entity is an action target of any automation
            return "__action_target__" in coverage.get(gap.trigger_entity, set())

    @staticmethod
    def _extract_trigger_entities(automation: dict) -> list[str]:
        """Extract entity_ids from automation triggers."""
        entities = []
        triggers = automation.get("trigger", automation.get("triggers", []))
        if isinstance(triggers, dict):
            triggers = [triggers]
        for trigger in triggers:
            entity_id = trigger.get("entity_id")
            if entity_id:
                if isinstance(entity_id, list):
                    entities.extend(entity_id)
                else:
                    entities.append(entity_id)
        return entities

    @staticmethod
    def _extract_action_entities(automation: dict) -> list[str]:
        """Extract entity_ids from automation actions."""
        entities = []
        actions = automation.get("action", automation.get("actions", []))
        if isinstance(actions, dict):
            actions = [actions]
        for action in actions:
            # Check target.entity_id
            target = action.get("target", {})
            if isinstance(target, dict):
                entity_id = target.get("entity_id")
                if entity_id:
                    if isinstance(entity_id, list):
                        entities.extend(entity_id)
                    else:
                        entities.append(entity_id)
            # Check data.entity_id (legacy format)
            data = action.get("data", {})
            if isinstance(data, dict):
                entity_id = data.get("entity_id")
                if entity_id:
                    if isinstance(entity_id, list):
                        entities.extend(entity_id)
                    else:
                        entities.append(entity_id)
        return entities

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_area(self, entity_id: str, events: list[dict]) -> str | None:
        """Resolve area — prefer event data, fall back to entity graph."""
        for event in events:
            area = event.get("area_id")
            if area:
                return area
        try:
            return self.hub.entity_graph.get_area(entity_id)
        except Exception:
            self.logger.debug("Failed to resolve area for %s via entity graph", entity_id)
            return None

    def _compute_recency(self, last_seen: str) -> float:
        """Exponential decay recency weight. 1.0 = today, decays over 30 days."""
        try:
            last = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            days_ago = (now - last).total_seconds() / 86400
            # Exponential decay: half-life of 15 days
            return max(0.01, 0.5 ** (days_ago / 15))
        except (ValueError, TypeError):
            return 0.5

    def _temporal_bounds(self, timestamps: list[str]) -> tuple[str, str]:
        """Return (first_seen, last_seen) from a list of ISO timestamps."""
        valid = sorted(t for t in timestamps if t)
        if not valid:
            return ("", "")
        return (valid[0], valid[-1])

    def _days_in_range(self, first_seen: str, last_seen: str) -> int:
        """Count calendar days between first and last seen."""
        try:
            first = datetime.fromisoformat(first_seen[:10])
            last = datetime.fromisoformat(last_seen[:10])
            return max((last - first).days + 1, 1)
        except (ValueError, TypeError):
            return 1

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime | None:
        """Parse ISO 8601 timestamp string to datetime."""
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError):
            return None
