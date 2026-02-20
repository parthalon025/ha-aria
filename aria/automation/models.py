"""Shared data models for Phase 3 automation generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChainLink:
    """One step in a detected behavioral sequence."""

    entity_id: str
    state: str
    offset_seconds: float  # seconds after chain trigger (0 for first link)


@dataclass
class DetectionResult:
    """Unified output from pattern engine or gap analyzer."""

    source: Literal["pattern", "gap"]
    trigger_entity: str
    action_entities: list[str]
    entity_chain: list[ChainLink]
    area_id: str | None
    confidence: float
    recency_weight: float
    observation_count: int
    first_seen: str  # ISO 8601
    last_seen: str  # ISO 8601
    day_type: str  # workday, weekend, holiday, wfh
    combined_score: float = 0.0  # computed by scoring step


@dataclass
class DayContext:
    """Classification of a single day for analysis segmentation."""

    date: str  # YYYY-MM-DD
    day_type: Literal["workday", "weekend", "holiday", "vacation", "wfh"]
    calendar_events: list[str] = field(default_factory=list)
    away_all_day: bool = False


@dataclass
class NormalizedEvent:
    """Event after normalization pipeline — ready for detection engines."""

    timestamp: str
    entity_id: str
    domain: str
    normalized_state: str  # "positive" or "negative"
    raw_state: str  # original state value
    area_id: str | None
    device_id: str | None
    day_type: str
    is_manual: bool  # True if context_parent_id is None
    attributes_json: str | None = None


@dataclass
class EntityHealth:
    """Availability scoring for an entity over analysis window."""

    entity_id: str
    availability_pct: float  # 0.0-1.0
    unavailable_transitions: int
    longest_outage_hours: float
    health_grade: Literal["healthy", "flaky", "unreliable"]


@dataclass
class ShadowResult:
    """Annotation on a candidate automation after shadow comparison."""

    candidate: dict[str, Any]  # the generated HA automation dict
    status: Literal["new", "duplicate", "conflict", "gap_fill"]
    duplicate_score: float  # 0.0-1.0
    conflicting_automation: str | None
    gap_source_automation: str | None
    reason: str  # human-readable explanation
