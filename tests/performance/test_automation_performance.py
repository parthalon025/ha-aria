"""Performance tests for the automation generation pipeline.

Validates:
- Generation time < 5s for 100K events worth of detections
- Memory delta < 50MB during generation
- Incremental sync (re-generation) < 1s
"""

import gc
import resource
import time
from unittest.mock import AsyncMock, patch

import pytest

from aria.automation.models import DetectionResult
from aria.hub.core import IntelligenceHub
from aria.modules.automation_generator import (
    AutomationGeneratorModule,
    compute_combined_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_memory_mb() -> float:
    """Get current RSS memory usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024  # Linux: ru_maxrss is in KB


def _make_bulk_detections(count: int, source: str = "pattern") -> list[dict]:
    """Generate a large number of detection dicts for perf testing.

    Each detection is unique (different trigger/action entities) to
    exercise the full template + validation + shadow pipeline.
    """
    detections = []
    for i in range(count):
        area = f"room_{i % 50}"  # 50 unique areas
        detections.append(
            {
                "source": source,
                "trigger_entity": f"binary_sensor.motion_{i}",
                "action_entities": [f"light.light_{i}"],
                "entity_chain": [
                    {"entity_id": f"binary_sensor.motion_{i}", "state": "on", "offset_seconds": 0.0},
                    {"entity_id": f"light.light_{i}", "state": "on", "offset_seconds": 3.0},
                ],
                "area_id": area,
                "confidence": 0.80 + (i % 20) * 0.005,
                "recency_weight": 0.70 + (i % 30) * 0.005,
                "observation_count": 10 + i % 50,
                "first_seen": "2026-01-01T07:00:00",
                "last_seen": "2026-02-15T07:00:00",
                "day_type": ["workday", "weekend", "all"][i % 3],
            }
        )
    return detections


def _build_entity_map(detections: list[dict]) -> dict[str, dict]:
    """Build entity graph data from detection dicts."""
    entities = {}
    for det in detections:
        area = det.get("area_id")
        for eid in [det["trigger_entity"]] + det["action_entities"]:
            entities[eid] = {"area_id": area, "device_id": f"dev_{eid.replace('.', '_')}"}
    return entities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def hub(tmp_path):
    """Minimal IntelligenceHub for performance tests."""
    h = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    await h.initialize()
    yield h
    await h.shutdown()


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_time_100k_events(hub):
    """Pipeline generation should complete in < 5s for detections representing 100K events.

    We simulate 100K events worth of detections by creating 200 unique
    detections (each representing ~500 observed event sequences). The
    pipeline processes top_n=10, so the critical path is scoring + filtering
    + 10 template builds + 10 validations + 10 shadow comparisons.
    """
    detections = _make_bulk_detections(200)
    entities = _build_entity_map(detections)
    hub.entity_graph.update(entities, {}, [])

    await hub.set_cache("patterns", {"detections": detections}, {"source": "perf"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto

        gc.collect()
        start = time.perf_counter()
        suggestions = await module.generate_suggestions()
        elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Generation took {elapsed:.2f}s, budget is 5s"
    assert len(suggestions) >= 1, "Should produce at least 1 suggestion"
    assert len(suggestions) <= 10, "top_n=10 should cap output"


@pytest.mark.asyncio
async def test_generation_time_large_detection_pool(hub):
    """Scoring and filtering 1000 detections should complete in < 5s."""
    detections = _make_bulk_detections(1000)
    entities = _build_entity_map(detections)
    hub.entity_graph.update(entities, {}, [])

    await hub.set_cache("patterns", {"detections": detections}, {"source": "perf"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto

        start = time.perf_counter()
        suggestions = await module.generate_suggestions()
        elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Generation with 1000 detections took {elapsed:.2f}s, budget is 5s"
    assert len(suggestions) <= 10


@pytest.mark.asyncio
async def test_memory_delta_under_budget(hub):
    """Memory growth during generation should be < 50MB.

    Measures RSS delta before and after pipeline execution. This tests
    that the pipeline doesn't accumulate large intermediate structures.
    """
    detections = _make_bulk_detections(500)
    entities = _build_entity_map(detections)
    hub.entity_graph.update(entities, {}, [])

    await hub.set_cache("patterns", {"detections": detections}, {"source": "perf"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto

        gc.collect()
        mem_before = _get_memory_mb()
        await module.generate_suggestions()
        gc.collect()
        mem_after = _get_memory_mb()

    delta = mem_after - mem_before
    # ru_maxrss only grows, never shrinks — so we allow negative delta
    # (means no new peak) and check positive delta is within budget
    assert delta < 50.0, f"Memory grew by {delta:.1f}MB, budget is 50MB"


@pytest.mark.asyncio
async def test_incremental_sync_under_1s(hub):
    """Re-running the pipeline (incremental sync) should complete in < 1s.

    The second run should benefit from the cache already containing
    suggestions, making the merge step fast.
    """
    detections = _make_bulk_detections(50)
    entities = _build_entity_map(detections)
    hub.entity_graph.update(entities, {}, [])

    await hub.set_cache("patterns", {"detections": detections}, {"source": "perf"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto

        # First run (cold)
        await module.generate_suggestions()

        # Second run (incremental) — this is what we measure
        gc.collect()
        start = time.perf_counter()
        suggestions = await module.generate_suggestions()
        elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Incremental sync took {elapsed:.2f}s, budget is 1s"
    assert len(suggestions) >= 1


@pytest.mark.asyncio
async def test_scoring_throughput():
    """compute_combined_score should handle 100K detections in < 1s."""
    detections = []
    for i in range(100_000):
        det = DetectionResult(
            source="pattern" if i % 2 == 0 else "gap",
            trigger_entity=f"binary_sensor.motion_{i}",
            action_entities=[f"light.light_{i}"],
            entity_chain=[],
            area_id=f"room_{i % 100}",
            confidence=0.80,
            recency_weight=0.70,
            observation_count=10,
            first_seen="2026-01-01T07:00:00",
            last_seen="2026-02-15T07:00:00",
            day_type="workday",
        )
        detections.append(det)

    gc.collect()
    start = time.perf_counter()
    for det in detections:
        compute_combined_score(det)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Scoring 100K detections took {elapsed:.2f}s, budget is 1s"

    # Verify scores are reasonable
    sample = detections[:10]
    for det in sample:
        assert 0.0 <= det.combined_score <= 1.0


@pytest.mark.asyncio
async def test_entity_graph_scaling(hub):
    """Pipeline should handle entity graphs with 3000+ entities."""
    # Build a large entity graph (simulating a real HA instance)
    entities = {}
    for i in range(3000):
        domain = ["light", "switch", "binary_sensor", "sensor", "media_player"][i % 5]
        area = f"room_{i % 60}"
        entities[f"{domain}.entity_{i}"] = {
            "area_id": area,
            "device_id": f"dev_{i}",
        }
    hub.entity_graph.update(entities, {}, [])

    # Small set of detections referencing entities in the graph
    detections = []
    for i in range(20):
        trigger = f"binary_sensor.entity_{i * 5 + 2}"
        action = f"light.entity_{i * 5}"
        detections.append(
            {
                "source": "pattern",
                "trigger_entity": trigger,
                "action_entities": [action],
                "entity_chain": [
                    {"entity_id": trigger, "state": "on", "offset_seconds": 0.0},
                    {"entity_id": action, "state": "on", "offset_seconds": 3.0},
                ],
                "area_id": f"room_{i % 60}",
                "confidence": 0.85,
                "recency_weight": 0.80,
                "observation_count": 15,
                "first_seen": "2026-01-01T07:00:00",
                "last_seen": "2026-02-15T07:00:00",
                "day_type": "workday",
            }
        )

    await hub.set_cache("patterns", {"detections": detections}, {"source": "perf"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto

        start = time.perf_counter()
        suggestions = await module.generate_suggestions()
        elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Pipeline with 3000 entities took {elapsed:.2f}s"
    assert len(suggestions) >= 1
