"""Tests for aria.iw.discovery — three-stage discovery engine.

Tests the pipeline: gather sources → build indicator chains → deduplicate/merge.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.automation.models import ChainLink, DetectionResult
from aria.iw.discovery import DiscoveryEngine
from aria.iw.models import BehavioralStateDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hub():
    """Build a minimal mock hub with cache.get_config_value."""
    hub = MagicMock()
    hub.modules = {}

    config_store = {
        "iw.min_discovery_confidence": 0.60,
        "iw.detector_window_seconds": 60,
    }

    async def _get_config_value(key, fallback=None):
        return config_store.get(key, fallback)

    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(side_effect=_get_config_value)
    return hub


def _pattern_dict(  # noqa: PLR0913
    *,
    trigger_entity: str = "binary_sensor.bedroom_motion",
    entity_chain: list[dict] | None = None,
    area: str = "bedroom",
    day_type: str = "workday",
    confidence: float = 0.75,
    name: str = "Morning Routine",
) -> dict:
    """Build a pattern dict as returned by patterns.detect_patterns()."""
    if entity_chain is None:
        entity_chain = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
            {"entity_id": "binary_sensor.bathroom_motion", "state": "on", "offset_seconds": 120},
        ]
    return {
        "pattern_id": f"{area}_{day_type}_cluster_1",
        "name": name,
        "area": area,
        "day_type": day_type,
        "typical_time": "07:30",
        "variance_minutes": 15,
        "frequency": 18,
        "total_days": 24,
        "confidence": confidence,
        "associated_signals": [],
        "cluster_size": 18,
        "entity_chain": entity_chain,
        "trigger_entity": trigger_entity,
        "first_seen": "2026-01-01T07:30:00",
        "last_seen": "2026-02-20T07:28:00",
        "source_event_count": 54,
        "llm_description": name,
    }


def _gap_result(  # noqa: PLR0913
    *,
    trigger_entity: str = "light.porch",
    action_entities: list[str] | None = None,
    entity_chain: list[ChainLink] | None = None,
    area_id: str | None = "porch",
    confidence: float = 0.70,
    day_type: str = "all",
) -> DetectionResult:
    """Build a DetectionResult as returned by anomaly_gap.analyze_gaps()."""
    if action_entities is None:
        action_entities = []
    if entity_chain is None:
        entity_chain = [ChainLink(entity_id=trigger_entity, state="on", offset_seconds=0)]
    return DetectionResult(
        source="gap",
        trigger_entity=trigger_entity,
        action_entities=action_entities,
        entity_chain=entity_chain,
        area_id=area_id,
        confidence=confidence,
        recency_weight=0.9,
        observation_count=14,
        first_seen="2026-01-15T18:00:00",
        last_seen="2026-02-20T18:05:00",
        day_type=day_type,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverFromPatterns:
    """Stage 1+2: pattern dicts → BehavioralStateDefinitions."""

    @pytest.mark.asyncio
    async def test_discover_from_patterns(self):
        hub = _make_hub()
        pattern = _pattern_dict()

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        assert len(defs) == 1
        defn = defs[0]
        assert isinstance(defn, BehavioralStateDefinition)
        assert defn.trigger.entity_id == "binary_sensor.bedroom_motion"
        assert defn.trigger.role == "trigger"
        assert defn.trigger.mode == "state_change"
        assert defn.trigger.expected_state == "on"
        assert len(defn.confirming) == 2
        assert "bedroom" in defn.areas
        assert "workday" in defn.day_types

    @pytest.mark.asyncio
    async def test_low_confidence_pattern_excluded(self):
        hub = _make_hub()
        pattern = _pattern_dict(confidence=0.30)  # below 0.60 threshold

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()
        assert len(defs) == 0


class TestDiscoverFromGapAnalyzer:
    """Stage 1+2: DetectionResult objects → BehavioralStateDefinitions."""

    @pytest.mark.asyncio
    async def test_discover_from_gap_analyzer(self):
        hub = _make_hub()

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[])
        hub.modules["patterns"] = patterns_mod

        gap = _gap_result()
        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[gap])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        assert len(defs) == 1
        defn = defs[0]
        assert defn.trigger.entity_id == "light.porch"
        assert defn.trigger.role == "trigger"
        assert len(defn.confirming) == 0  # solo toggle has no confirming


class TestIndicatorChainConstruction:
    """Verify trigger = first entity, confirming = remaining with timing."""

    @pytest.mark.asyncio
    async def test_indicator_chain_construction(self):
        hub = _make_hub()
        chain = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 45},
            {"entity_id": "light.bathroom", "state": "on", "offset_seconds": 180},
        ]
        pattern = _pattern_dict(entity_chain=chain)

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        defn = defs[0]
        # Trigger is the first entity in the chain
        assert defn.trigger.entity_id == "binary_sensor.bedroom_motion"
        assert defn.trigger.max_delay_seconds == 0

        # Confirming indicators have timing from the chain offsets
        assert defn.confirming[0].entity_id == "light.bedroom"
        assert defn.confirming[0].max_delay_seconds == 45
        assert defn.confirming[1].entity_id == "light.bathroom"
        assert defn.confirming[1].max_delay_seconds == 180


class TestPreconditionQuietPeriod:
    """When entity_chain shows >4h gap context, assert quiet_period precondition."""

    @pytest.mark.asyncio
    async def test_precondition_quiet_period(self):
        hub = _make_hub()
        # Pattern with a quiet period hint: trigger entity had >4h gap (14400s)
        chain = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
        ]
        pattern = _pattern_dict(entity_chain=chain)
        # Simulate quiet period info: add a gap_before_trigger_seconds field
        pattern["gap_before_trigger_seconds"] = 18000  # 5 hours

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        defn = defs[0]
        assert len(defn.trigger_preconditions) >= 1
        quiet = defn.trigger_preconditions[0]
        assert quiet.mode == "quiet_period"
        assert quiet.entity_id == "binary_sensor.bedroom_motion"
        assert quiet.quiet_seconds == 18000


class TestDeduplication:
    """Stage 3: merge overlapping definitions, keep distinct ones separate."""

    @pytest.mark.asyncio
    async def test_deduplication_merge(self):
        """Two patterns with >60% indicator overlap merge into one (higher confidence wins)."""
        hub = _make_hub()

        # Pattern A: bedroom motion → bedroom light → bathroom motion → bathroom light
        chain_a = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
            {"entity_id": "binary_sensor.bathroom_motion", "state": "on", "offset_seconds": 120},
            {"entity_id": "light.bathroom", "state": "on", "offset_seconds": 150},
        ]
        # Pattern B: same trigger + 3 of 4 entities overlap (75% Jaccard > 60%)
        chain_b = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 35},
            {"entity_id": "binary_sensor.bathroom_motion", "state": "on", "offset_seconds": 125},
        ]
        pattern_a = _pattern_dict(entity_chain=chain_a, confidence=0.80, name="Morning A")
        pattern_b = _pattern_dict(entity_chain=chain_b, confidence=0.70, name="Morning B")

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern_a, pattern_b])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        # Should merge into one definition
        assert len(defs) == 1

    @pytest.mark.asyncio
    async def test_deduplication_distinct(self):
        """Two patterns with <60% overlap remain as two separate definitions."""
        hub = _make_hub()

        # Pattern A: bedroom
        chain_a = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
        ]
        # Pattern B: kitchen — completely different entities
        chain_b = [
            {"entity_id": "binary_sensor.kitchen_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.kitchen", "state": "on", "offset_seconds": 25},
        ]
        pattern_a = _pattern_dict(
            trigger_entity="binary_sensor.bedroom_motion",
            entity_chain=chain_a,
            area="bedroom",
            confidence=0.75,
        )
        pattern_b = _pattern_dict(
            trigger_entity="binary_sensor.kitchen_motion",
            entity_chain=chain_b,
            area="kitchen",
            confidence=0.80,
        )

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern_a, pattern_b])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()

        assert len(defs) == 2


class TestDeterministicId:
    """Same indicators, same area/day_type → same ID regardless of run order."""

    @pytest.mark.asyncio
    async def test_deterministic_id(self):
        hub = _make_hub()

        chain = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
        ]
        pattern = _pattern_dict(entity_chain=chain)

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)

        # Run twice — same input → same ID
        defs1 = await engine.discover()
        defs2 = await engine.discover()

        assert defs1[0].id == defs2[0].id

    @pytest.mark.asyncio
    async def test_deterministic_id_order_independent(self):
        """Confirming entities in different order → still same ID."""
        hub = _make_hub()

        chain_a = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 30},
            {"entity_id": "light.bathroom", "state": "on", "offset_seconds": 60},
        ]
        chain_b = [
            {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0},
            {"entity_id": "light.bathroom", "state": "on", "offset_seconds": 25},
            {"entity_id": "light.bedroom", "state": "on", "offset_seconds": 55},
        ]
        pattern_a = _pattern_dict(entity_chain=chain_a, confidence=0.80)
        pattern_b = _pattern_dict(entity_chain=chain_b, confidence=0.75)

        # Run A first
        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern_a])
        hub.modules["patterns"] = patterns_mod
        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs_a = await engine.discover()

        # Run B (same entities, different order)
        patterns_mod.detect_patterns = AsyncMock(return_value=[pattern_b])
        defs_b = await engine.discover()

        assert defs_a[0].id == defs_b[0].id


class TestEmptyPatterns:
    """No patterns → no definitions, no error."""

    @pytest.mark.asyncio
    async def test_empty_patterns(self):
        hub = _make_hub()

        patterns_mod = MagicMock()
        patterns_mod.detect_patterns = AsyncMock(return_value=[])
        hub.modules["patterns"] = patterns_mod

        gap_mod = MagicMock()
        gap_mod.analyze_gaps = AsyncMock(return_value=[])
        hub.modules["anomaly_gap"] = gap_mod

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()
        assert defs == []

    @pytest.mark.asyncio
    async def test_missing_modules_returns_empty(self):
        """If patterns/gap modules not registered, discover returns empty."""
        hub = _make_hub()
        hub.modules = {}

        engine = DiscoveryEngine(hub)
        defs = await engine.discover()
        assert defs == []
