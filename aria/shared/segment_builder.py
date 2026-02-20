"""Segment builder — generates ML feature segments from EventStore events.

Reads raw state_changed events from EventStore and produces fixed-interval
feature dicts suitable for ML training and prediction. Each segment
summarizes activity within a time window (default 15 min).
"""

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore


class SegmentBuilder:
    """Build feature segments from EventStore event windows.

    Each segment captures:
    - event_count: total events in the window
    - light_transitions: on↔off changes for light.* entities
    - motion_events: binary_sensor events with device_class=motion
    - unique_entities_active: distinct entity_ids that fired
    - domain_entropy: Shannon entropy over domain distribution
    - per_area_activity: {area_id: count} for events with area_id set
    - per_domain_counts: {domain: count}
    """

    def __init__(self, event_store: EventStore, entity_graph: EntityGraph):
        self.event_store = event_store
        self.entity_graph = entity_graph

    async def build_segment(self, start: str, end: str) -> dict:
        """Build a single feature segment for the [start, end) window."""
        events = await self.event_store.query_events(start, end)
        return self._compute_features(events, start, end)

    async def build_segments(self, start: str, end: str, interval_minutes: int = 15) -> list[dict]:
        """Build consecutive segments covering [start, end)."""
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        segments = []
        current = start_dt
        while current < end_dt:
            window_end = min(current + timedelta(minutes=interval_minutes), end_dt)
            segment = await self.build_segment(current.isoformat(), window_end.isoformat())
            segments.append(segment)
            current = window_end
        return segments

    def _compute_features(self, events: list[dict], start: str, end: str) -> dict:
        """Extract all feature values from a list of events."""
        return {
            "start": start,
            "end": end,
            "event_count": len(events),
            "light_transitions": self._count_light_transitions(events),
            "motion_events": self._count_motion_events(events),
            "unique_entities_active": len({e["entity_id"] for e in events}),
            "per_area_activity": self._compute_per_area_activity(events),
            "domain_entropy": self._compute_domain_entropy(events),
            "per_domain_counts": dict(Counter(e["domain"] for e in events)),
        }

    @staticmethod
    def _count_light_transitions(events: list[dict]) -> int:
        return sum(
            1
            for e in events
            if e["domain"] == "light"
            and e.get("old_state") in ("on", "off")
            and e.get("new_state") in ("on", "off")
            and e.get("old_state") != e.get("new_state")
        )

    @staticmethod
    def _count_motion_events(events: list[dict]) -> int:
        count = 0
        for e in events:
            if e["domain"] == "binary_sensor" and e.get("new_state") == "on":
                attrs = e.get("attributes_json")
                if attrs:
                    try:
                        parsed = json.loads(attrs) if isinstance(attrs, str) else attrs
                        if parsed.get("device_class") == "motion":
                            count += 1
                    except (json.JSONDecodeError, TypeError):
                        pass
        return count

    @staticmethod
    def _compute_domain_entropy(events: list[dict]) -> float:
        if not events:
            return 0.0
        counts = Counter(e["domain"] for e in events)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    @staticmethod
    def _compute_per_area_activity(events: list[dict]) -> dict:
        area_counts: dict[str, int] = defaultdict(int)
        for e in events:
            area = e.get("area_id")
            if area:
                area_counts[area] += 1
        return dict(area_counts)
