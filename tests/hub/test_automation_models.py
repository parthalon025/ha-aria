"""Tests for Phase 3 automation data models."""

from aria.automation.models import (
    ChainLink,
    DayContext,
    DetectionResult,
    EntityHealth,
    NormalizedEvent,
    ShadowResult,
)


class TestDetectionResult:
    def test_create_from_pattern(self):
        result = DetectionResult(
            source="pattern",
            trigger_entity="binary_sensor.bedroom_motion",
            action_entities=["light.bedroom"],
            entity_chain=[
                ChainLink(entity_id="binary_sensor.bedroom_motion", state="on", offset_seconds=0),
                ChainLink(entity_id="light.bedroom", state="on", offset_seconds=30),
            ],
            area_id="bedroom",
            confidence=0.85,
            recency_weight=0.9,
            observation_count=47,
            first_seen="2026-01-01T06:30:00",
            last_seen="2026-02-19T06:45:00",
            day_type="workday",
            combined_score=0.0,  # computed later
        )
        assert result.source == "pattern"
        assert result.trigger_entity == "binary_sensor.bedroom_motion"
        assert len(result.entity_chain) == 2

    def test_create_from_gap(self):
        result = DetectionResult(
            source="gap",
            trigger_entity="light.kitchen",
            action_entities=["light.kitchen"],
            entity_chain=[
                ChainLink(entity_id="light.kitchen", state="on", offset_seconds=0),
            ],
            area_id="kitchen",
            confidence=0.72,
            recency_weight=0.95,
            observation_count=40,
            first_seen="2026-01-05T06:45:00",
            last_seen="2026-02-20T07:10:00",
            day_type="workday",
            combined_score=0.0,
        )
        assert result.source == "gap"


class TestDayContext:
    def test_workday(self):
        ctx = DayContext(
            date="2026-02-20",
            day_type="workday",
            calendar_events=[],
            away_all_day=False,
        )
        assert ctx.day_type == "workday"

    def test_holiday(self):
        ctx = DayContext(
            date="2026-12-25",
            day_type="holiday",
            calendar_events=["Christmas Day"],
            away_all_day=False,
        )
        assert ctx.day_type == "holiday"


class TestNormalizedEvent:
    def test_create(self):
        evt = NormalizedEvent(
            timestamp="2026-02-20T07:00:00",
            entity_id="binary_sensor.bedroom_motion",
            domain="binary_sensor",
            normalized_state="positive",
            raw_state="on",
            area_id="bedroom",
            device_id="device_123",
            day_type="workday",
            is_manual=True,
            attributes_json=None,
        )
        assert evt.normalized_state == "positive"
        assert evt.is_manual is True


class TestEntityHealth:
    def test_healthy(self):
        h = EntityHealth(
            entity_id="light.bedroom",
            availability_pct=0.98,
            unavailable_transitions=5,
            longest_outage_hours=0.5,
            health_grade="healthy",
        )
        assert h.health_grade == "healthy"

    def test_unreliable(self):
        h = EntityHealth(
            entity_id="sensor.flaky",
            availability_pct=0.65,
            unavailable_transitions=200,
            longest_outage_hours=12.0,
            health_grade="unreliable",
        )
        assert h.health_grade == "unreliable"


class TestShadowResult:
    def test_new_suggestion(self):
        r = ShadowResult(
            candidate={"id": "test", "alias": "Test"},
            status="new",
            duplicate_score=0.0,
            conflicting_automation=None,
            gap_source_automation=None,
            reason="No matching existing automation found.",
        )
        assert r.status == "new"

    def test_duplicate(self):
        r = ShadowResult(
            candidate={"id": "test", "alias": "Test"},
            status="duplicate",
            duplicate_score=0.92,
            conflicting_automation=None,
            gap_source_automation=None,
            reason="Existing automation 'Bedroom lights' covers this.",
        )
        assert r.status == "duplicate"
        assert r.duplicate_score > 0.8
