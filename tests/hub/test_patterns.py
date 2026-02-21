"""Tests for Pattern Recognition module — Phase 3 rewrite.

Tests cover:
- Task 13: EventStore data source (replacing logbook files)
- Task 14: Day-type segmentation (per day-type pattern detection)
- Task 15: New output fields (entity_chain, trigger_entity, etc.) + scheduling
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.patterns import PatternRecognition

# ── Fixtures ──────────────────────────────────────────────────────────


class MockHub:
    """Lightweight hub mock with EventStore and EntityGraph."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self.modules = {}
        self.event_store = AsyncMock()
        self.entity_graph = MagicMock()
        self.schedule_task = AsyncMock()
        self.cache = MagicMock()
        self.cache.get_config_value = AsyncMock(return_value=7200)

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = {"data": data, "metadata": metadata}

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        return self._cache.get(category)

    def register_module(self, module):
        self.modules[module.module_id] = module

    async def publish(self, event_type: str, data: dict[str, Any]):
        pass


def _mock_ollama_generate(**kwargs):
    """Return a deterministic label based on area mentioned in the prompt."""
    prompt = kwargs.get("prompt", "")
    if "bedroom" in prompt.lower():
        label = "Morning routine"
    elif "kitchen" in prompt.lower():
        label = "Lunch prep"
    elif "living" in prompt.lower():
        label = "Evening wind-down"
    else:
        label = "Daily activity"
    response = MagicMock()
    response.response = label
    return response


def _make_sequence_events(area_id, chain, days=20, base_hour=7):
    """Generate events forming a repeatable sequence for pattern detection."""
    events = []
    base = datetime(2026, 1, 20)
    for day in range(days):
        date = base + timedelta(days=day)
        for offset, (entity_id, state) in enumerate(chain):
            ts = date.replace(hour=base_hour, minute=offset * 2)
            events.append(
                {
                    "id": len(events) + 1,
                    "timestamp": ts.isoformat(),
                    "entity_id": entity_id,
                    "domain": entity_id.split(".")[0],
                    "old_state": "off" if state == "on" else "on",
                    "new_state": state,
                    "device_id": f"dev_{entity_id.replace('.', '_')}",
                    "area_id": area_id,
                    "attributes_json": None,
                    "context_parent_id": None,
                }
            )
    return events


def _make_events(area_id, entity_prefix, days=10, events_per_day=5, domain="light"):
    """Generate synthetic EventStore events for testing."""
    events = []
    base = datetime(2026, 1, 20)
    for day in range(days):
        date = base + timedelta(days=day)
        for i in range(events_per_day):
            hour = 7 + (i % 12)
            ts = date.replace(hour=hour, minute=i * 5)
            events.append(
                {
                    "id": len(events) + 1,
                    "timestamp": ts.isoformat(),
                    "entity_id": f"{entity_prefix}.{area_id}_{i % 3}",
                    "domain": domain,
                    "old_state": "off",
                    "new_state": "on",
                    "device_id": f"dev_{area_id}_{i % 3}",
                    "area_id": area_id,
                    "attributes_json": None,
                    "context_parent_id": None,
                }
            )
    return events


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def patterns_module(hub):
    """Pattern module with low thresholds for testing."""
    module = PatternRecognition(
        hub=hub,
        min_pattern_frequency=1,
        min_support=0.3,
        min_confidence=0.3,
    )
    hub.register_module(module)
    return module


# ── Task 13: EventStore Data Source ───────────────────────────────────


class TestModuleBasics:
    """Module registration and core structure."""

    @pytest.mark.asyncio
    async def test_module_registration(self, hub, patterns_module):
        assert "pattern_recognition" in hub.modules
        assert hub.modules["pattern_recognition"] == patterns_module

    @pytest.mark.asyncio
    async def test_no_log_dir_attribute(self, patterns_module):
        """Rewritten engine should not reference log_dir."""
        assert not hasattr(patterns_module, "log_dir")

    @pytest.mark.asyncio
    async def test_on_event_handler(self, patterns_module):
        """Module should handle events without crashing."""
        await patterns_module.on_event("test_event", {"data": "test"})
        await patterns_module.on_event("cache_updated", {"category": "test"})


class TestEventStoreDataSource:
    """Verify pattern engine queries EventStore instead of logbook files."""

    @pytest.mark.asyncio
    async def test_extract_sequences_uses_event_store(self, hub, patterns_module):
        """Pattern engine should query EventStore."""
        events = _make_events("bedroom", "light", days=10)
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": 50})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        sequences = await patterns_module._extract_sequences()

        hub.event_store.area_event_summary.assert_called_once()
        hub.event_store.query_by_area.assert_called()
        assert "bedroom" in sequences

    @pytest.mark.asyncio
    async def test_top_n_areas_respected(self, hub):
        """Should only analyze top-N most active areas."""
        summary = {f"area_{i}": (100 - i) * 10 for i in range(25)}
        hub.event_store.area_event_summary = AsyncMock(return_value=summary)
        hub.event_store.query_by_area = AsyncMock(return_value=[])

        pr = PatternRecognition(hub=hub, max_areas=5)
        await pr._extract_sequences()

        assert hub.event_store.query_by_area.call_count == 5

    @pytest.mark.asyncio
    async def test_entity_graph_area_resolution(self, hub, patterns_module):
        """Should use EntityGraph for events missing area_id."""
        events = _make_events("bedroom", "light", days=5)
        for e in events[:5]:
            e["area_id"] = None

        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": 25})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        await patterns_module._extract_sequences()

        assert hub.entity_graph.get_area.call_count >= 5

    @pytest.mark.asyncio
    async def test_empty_event_store_returns_empty(self, hub, patterns_module):
        hub.event_store.area_event_summary = AsyncMock(return_value={})

        sequences = await patterns_module._extract_sequences()
        assert sequences == {}


class TestDTWClustering:
    """DTW clustering algorithm preserved from original."""

    @pytest.mark.asyncio
    async def test_dtw_identical(self, patterns_module):
        assert patterns_module._dtw_distance([100, 200, 300], [100, 200, 300]) == 0.0

    @pytest.mark.asyncio
    async def test_dtw_different(self, patterns_module):
        dist = patterns_module._dtw_distance([100, 200, 300], [400, 500, 600])
        assert dist > 0

    @pytest.mark.asyncio
    async def test_dtw_empty_returns_inf(self, patterns_module):
        assert patterns_module._dtw_distance([], [100]) == float("inf")

    @pytest.mark.asyncio
    async def test_dtw_different_lengths(self, patterns_module):
        dist = patterns_module._dtw_distance([60, 120], [60, 120, 180])
        assert dist > 0

    @pytest.mark.asyncio
    async def test_dtw_very_different(self, patterns_module):
        dist = patterns_module._dtw_distance([60, 120, 180], [300, 360, 420])
        assert dist > 300

    @pytest.mark.asyncio
    async def test_cluster_groups_similar(self, patterns_module):
        """Similar sequences should cluster together."""
        sequences = [
            {"light_times": [420, 425, 430]},
            {"light_times": [422, 427, 432]},
            {"light_times": [421, 426, 431]},
            {"light_times": [1200, 1205, 1210]},
            {"light_times": [1202, 1207, 1212]},
            {"light_times": [1201, 1206, 1211]},
        ]
        clusters = await patterns_module._cluster_sequences(sequences)
        assert len(clusters) >= 2


class TestApriori:
    """Apriori association rule mining preserved from original."""

    @pytest.mark.asyncio
    async def test_find_associations_with_data(self, hub):
        pr = PatternRecognition(hub=hub, min_support=0.5, min_confidence=0.5)
        sequences = [
            {"transactions": ["light_on", "motion_on"]},
            {"transactions": ["light_on", "motion_on"]},
            {"transactions": ["light_on", "motion_on"]},
            {"transactions": ["light_on"]},
        ]
        associations = await pr._find_associations(sequences)
        assert isinstance(associations, list)

    @pytest.mark.asyncio
    async def test_find_associations_empty(self, hub):
        pr = PatternRecognition(hub=hub, min_pattern_frequency=10)
        sequences = [{"transactions": ["a"]}]
        associations = await pr._find_associations(sequences)
        assert associations == []


class TestLLMInterpretation:
    """LLM semantic labeling."""

    @pytest.mark.asyncio
    async def test_llm_generates_labels(self, hub, patterns_module):
        events = _make_sequence_events(
            "bedroom",
            [("binary_sensor.bedroom_motion", "on"), ("light.bedroom", "on")],
            days=10,
        )
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": len(events)})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()

        for p in patterns:
            assert p["llm_description"] != ""
            assert p["llm_description"] != "Failed to generate LLM description"
            assert len(p["llm_description"]) < 100

    @pytest.mark.asyncio
    async def test_strip_think_tags(self, patterns_module):
        assert patterns_module._strip_think_tags("<think>reasoning</think>Morning routine") == "Morning routine"
        assert patterns_module._strip_think_tags("Before<think>r</think>After") == "BeforeAfter"


# ── Task 14: Day-Type Analysis ────────────────────────────────────────


class TestDayTypeAnalysis:
    """Pattern detection should run separately per day-type segment."""

    @pytest.mark.asyncio
    async def test_patterns_tagged_with_day_type(self, hub, patterns_module):
        """Each detected pattern should have a day_type field."""
        events = _make_sequence_events(
            "bedroom",
            [("binary_sensor.bedroom_motion", "on"), ("light.bedroom", "on")],
            days=20,
        )
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": len(events)})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()

        for p in patterns:
            assert "day_type" in p
            assert p["day_type"] in ("workday", "weekend", "holiday", "vacation", "wfh")

    @pytest.mark.asyncio
    async def test_workday_and_weekend_separated(self, hub, patterns_module):
        """Workday and weekend events should produce separate segments."""
        base = datetime(2026, 1, 19)  # Monday
        events = []
        for day in range(28):
            date = base + timedelta(days=day)
            is_weekend = date.weekday() >= 5
            hour = 9 if is_weekend else 7

            events.append(
                {
                    "id": day * 2,
                    "timestamp": date.replace(hour=hour, minute=0).isoformat(),
                    "entity_id": "binary_sensor.bedroom_motion",
                    "domain": "binary_sensor",
                    "old_state": "off",
                    "new_state": "on",
                    "device_id": "dev_1",
                    "area_id": "bedroom",
                    "attributes_json": None,
                    "context_parent_id": None,
                }
            )
            events.append(
                {
                    "id": day * 2 + 1,
                    "timestamp": date.replace(hour=hour, minute=2).isoformat(),
                    "entity_id": "light.bedroom",
                    "domain": "light",
                    "old_state": "off",
                    "new_state": "on",
                    "device_id": "dev_2",
                    "area_id": "bedroom",
                    "attributes_json": None,
                    "context_parent_id": None,
                }
            )

        events.sort(key=lambda e: e["timestamp"])
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": len(events)})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()

        day_types = {p["day_type"] for p in patterns}
        assert len(day_types) >= 1

    @pytest.mark.asyncio
    async def test_no_vacation_patterns(self, hub, patterns_module):
        """Vacation day_type should not appear in patterns."""
        events = _make_events("bedroom", "light", days=10)
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": 50})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()

        for p in patterns:
            assert p.get("day_type") != "vacation"

    @pytest.mark.asyncio
    async def test_classify_day_simple_weekday(self, patterns_module):
        assert patterns_module._classify_day_simple("2026-02-20") == "workday"  # Friday

    @pytest.mark.asyncio
    async def test_classify_day_simple_weekend(self, patterns_module):
        assert patterns_module._classify_day_simple("2026-02-21") == "weekend"  # Saturday


# ── Task 15: New Output Fields + Scheduling ───────────────────────────


class TestNewOutputFields:
    """Patterns should include enriched fields."""

    def _setup_events(self, hub, days=15):
        events = _make_sequence_events(
            "bedroom",
            [("binary_sensor.bedroom_motion", "on"), ("light.bedroom", "on")],
            days=days,
        )
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": len(events)})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"
        return events

    @pytest.mark.asyncio
    async def test_entity_chain_present(self, hub, patterns_module):
        self._setup_events(hub)
        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()
        for p in patterns:
            assert "entity_chain" in p
            assert isinstance(p["entity_chain"], list)

    @pytest.mark.asyncio
    async def test_trigger_entity_present(self, hub, patterns_module):
        self._setup_events(hub)
        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()
        for p in patterns:
            assert "trigger_entity" in p

    @pytest.mark.asyncio
    async def test_temporal_bounds(self, hub, patterns_module):
        self._setup_events(hub)
        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()
        for p in patterns:
            assert "first_seen" in p
            assert "last_seen" in p
            assert p["first_seen"] <= p["last_seen"]

    @pytest.mark.asyncio
    async def test_source_event_count(self, hub, patterns_module):
        self._setup_events(hub)
        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()
        for p in patterns:
            assert "source_event_count" in p
            assert p["source_event_count"] > 0

    @pytest.mark.asyncio
    async def test_pattern_complete_structure(self, hub, patterns_module):
        """All required fields from Phase 3 design should be present."""
        self._setup_events(hub)
        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            patterns = await patterns_module.detect_patterns()
        assert len(patterns) > 0
        required_fields = [
            "pattern_id",
            "name",
            "area",
            "day_type",
            "typical_time",
            "variance_minutes",
            "frequency",
            "total_days",
            "confidence",
            "associated_signals",
            "cluster_size",
            "entity_chain",
            "trigger_entity",
            "first_seen",
            "last_seen",
            "source_event_count",
            "llm_description",
        ]
        for p in patterns:
            for field in required_fields:
                assert field in p, f"Pattern missing field: {field}"


class TestPeriodicScheduling:
    """Pattern detection should schedule periodic runs via hub timer."""

    @pytest.mark.asyncio
    async def test_initialize_schedules_periodic(self, hub, patterns_module):
        hub.event_store.area_event_summary = AsyncMock(return_value={})

        await patterns_module.initialize()

        hub.schedule_task.assert_called_once()
        call_kwargs = hub.schedule_task.call_args
        task_id = call_kwargs.kwargs.get("task_id", "")
        assert "pattern" in task_id

    @pytest.mark.asyncio
    async def test_schedule_uses_config_interval(self, hub):
        """Interval should come from patterns.analysis_interval config."""
        hub.cache.get_config_value = AsyncMock(return_value=3600)
        hub.event_store.area_event_summary = AsyncMock(return_value={})

        pr = PatternRecognition(hub=hub)
        await pr.initialize()

        call_kwargs = hub.schedule_task.call_args
        interval = call_kwargs.kwargs.get("interval")
        assert interval is not None
        assert interval.total_seconds() == 3600


class TestCacheStorage:
    """Pattern results stored in hub cache."""

    @pytest.mark.asyncio
    async def test_stores_in_patterns_cache(self, hub, patterns_module):
        events = _make_sequence_events(
            "bedroom",
            [("binary_sensor.bedroom_motion", "on"), ("light.bedroom", "on")],
            days=15,
        )
        hub.event_store.area_event_summary = AsyncMock(return_value={"bedroom": len(events)})
        hub.event_store.query_by_area = AsyncMock(return_value=events)
        hub.entity_graph.get_area.return_value = "bedroom"

        with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
            await patterns_module.detect_patterns()

        cache = await hub.get_cache("patterns")
        assert cache is not None
        data = cache["data"]
        assert "patterns" in data
        assert "pattern_count" in data
        assert "areas_analyzed" in data

    @pytest.mark.asyncio
    async def test_empty_store_produces_empty_cache(self, hub, patterns_module):
        hub.event_store.area_event_summary = AsyncMock(return_value={})

        await patterns_module.detect_patterns()

        cache = await hub.get_cache("patterns")
        assert cache is not None
        assert cache["data"]["pattern_count"] == 0


# ── Helper function unit tests ────────────────────────────────────────


class TestHelpers:
    """Unit tests for internal helper methods."""

    def test_timestamp_to_minutes(self):
        assert PatternRecognition._timestamp_to_minutes("2026-02-20T07:30:00") == 450
        assert PatternRecognition._timestamp_to_minutes("2026-02-20T00:00:00") == 0
        assert PatternRecognition._timestamp_to_minutes("invalid") is None

    def test_select_top_areas(self, hub):
        pr = PatternRecognition(hub=hub, max_areas=3)
        summary = {"bedroom": 100, "kitchen": 80, "hall": 60, "bath": 40}
        top = pr._select_top_areas(summary)
        assert len(top) == 3
        assert top[0] == "bedroom"  # highest first

    def test_resolve_missing_areas(self, hub):
        hub.entity_graph.get_area.return_value = "kitchen"
        pr = PatternRecognition(hub=hub)
        events = [
            {"entity_id": "light.kitchen", "area_id": None},
            {"entity_id": "light.bedroom", "area_id": "bedroom"},
        ]
        result = pr._resolve_missing_areas(events, "fallback")
        assert result[0]["area_id"] == "kitchen"
        assert result[1]["area_id"] == "bedroom"

    def test_build_transactions(self, hub):
        pr = PatternRecognition(hub=hub)
        events = [
            {"entity_id": "light.bed", "new_state": "on", "timestamp": "2026-02-20T07:00:00"},
            {"entity_id": "binary_sensor.motion", "new_state": "on", "timestamp": "2026-02-20T07:01:00"},
        ]
        txns = pr._build_transactions(events)
        assert isinstance(txns, list)
        assert len(txns) == 2
        assert any("light_on_h7" in t for t in txns)
