"""Entity health scoring — grades entities by availability for filtering."""

import logging
from typing import Literal

from aria.automation.models import EntityHealth

logger = logging.getLogger(__name__)

UNAVAILABLE_STATES = {"unavailable", "unknown"}


def compute_entity_health(
    entity_id: str,
    events: list[dict],
    total_events: int,
    min_healthy_pct: float = 0.95,
    min_available_pct: float = 0.80,
) -> EntityHealth:
    """Compute health grade for an entity based on unavailable transitions.

    Args:
        entity_id: The entity to score.
        events: All events for this entity in the analysis window.
        total_events: Total event count (for percentage calculation).
        min_healthy_pct: Above this = healthy (default 95%).
        min_available_pct: Below this = unreliable (default 80%).
    """
    if total_events == 0:
        return EntityHealth(
            entity_id=entity_id,
            availability_pct=0.0,
            unavailable_transitions=0,
            longest_outage_hours=0.0,
            health_grade="unreliable",
        )

    unavailable_count = sum(1 for e in events if e.get("new_state") in UNAVAILABLE_STATES)
    entity_event_count = len(events)
    availability_pct = 1.0 - (unavailable_count / entity_event_count) if entity_event_count > 0 else 0.0

    # Longest outage would require timestamp analysis — simplified for now
    longest_outage = 0.0

    grade: Literal["healthy", "flaky", "unreliable"]
    if availability_pct >= min_healthy_pct:
        grade = "healthy"
    elif availability_pct >= min_available_pct:
        grade = "flaky"
    else:
        grade = "unreliable"

    return EntityHealth(
        entity_id=entity_id,
        availability_pct=availability_pct,
        unavailable_transitions=unavailable_count,
        longest_outage_hours=longest_outage,
        health_grade=grade,
    )
