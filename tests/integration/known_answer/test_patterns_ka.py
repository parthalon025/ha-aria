"""Known-answer tests for PatternRecognition module.

Validates pattern detection, hub cache storage, and golden snapshot stability
against hand-crafted EventStore data with clear recurring routines.

Phase 3 rewrite: uses EventStore events instead of logbook files.
"""

import json
from unittest.mock import patch

import pytest

from aria.modules.patterns import PatternRecognition
from tests.integration.known_answer.conftest import FIXTURES_DIR, golden_compare


def _load_fixture_as_eventstore_events() -> list[dict]:
    """Convert logbook fixture data to EventStore event format.

    Reads the original logbook_patterns.json and transforms each logbook
    entry into EventStore-compatible dict format with all required columns.
    """
    fixture_path = FIXTURES_DIR / "logbook_patterns.json"
    fixture_data = json.loads(fixture_path.read_text())

    events = []
    event_id = 1
    for _date_str, day_events in sorted(fixture_data.items()):
        for entry in day_events:
            entity_id = entry["entity_id"]
            domain = entity_id.split(".")[0]
            # Derive area from entity name (e.g., "Kitchen Ceiling Light" -> "kitchen")
            name = entry.get("name", "")
            area_id = _area_from_name(name)

            events.append(
                {
                    "id": event_id,
                    "timestamp": entry["when"],
                    "entity_id": entity_id,
                    "domain": domain,
                    "old_state": None,
                    "new_state": entry["state"],
                    "device_id": f"dev_{entity_id.replace('.', '_')}",
                    "area_id": area_id,
                    "attributes_json": None,
                    "context_parent_id": None,
                }
            )
            event_id += 1

    return events


def _area_from_name(name: str) -> str:
    """Extract area from entity name for fixture data."""
    name_lower = name.lower()
    for area in ("kitchen", "living_room", "bedroom", "bathroom", "hallway"):
        if area.replace("_", " ") in name_lower:
            return area
    return "general"


@pytest.fixture
async def patterns_module(hub, tmp_path):
    """Create a PatternRecognition module backed by fixture EventStore data.

    Loads hand-crafted logbook data as EventStore events and mocks the
    EventStore query methods to return them.
    """
    all_events = _load_fixture_as_eventstore_events()

    # Populate the real event store with fixture data
    await hub.event_store.initialize()
    for event in all_events:
        await hub.event_store.insert_event(
            timestamp=event["timestamp"],
            entity_id=event["entity_id"],
            domain=event["domain"],
            old_state=event.get("old_state"),
            new_state=event.get("new_state"),
            device_id=event.get("device_id"),
            area_id=event.get("area_id"),
            attributes_json=event.get("attributes_json"),
            context_parent_id=event.get("context_parent_id"),
        )

    # Set up EntityGraph with area resolution
    entities = {}
    for event in all_events:
        eid = event["entity_id"]
        if eid not in entities:
            entities[eid] = {
                "area_id": event.get("area_id"),
                "device_id": event.get("device_id"),
            }
    hub.entity_graph.update(entities, {}, [])

    module = PatternRecognition(
        hub=hub,
        min_pattern_frequency=2,
        min_support=0.5,
        min_confidence=0.5,
        analysis_days=30,
    )

    return module


def _mock_ollama_generate(**kwargs):
    """Deterministic stand-in for ``ollama.generate``."""

    class _Response:
        response = "Morning routine"

    return _Response()


@pytest.mark.asyncio
async def test_detects_recurring_patterns(patterns_module):
    """Pattern detection should find >= 1 pattern from the kitchen morning routine."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        patterns = await patterns_module.detect_patterns()

    assert len(patterns) >= 1, f"Expected at least 1 pattern, got {len(patterns)}"

    # At least one pattern should be in the kitchen area
    areas = {p["area"] for p in patterns}
    assert "kitchen" in areas, f"Expected 'kitchen' area in patterns, got areas: {areas}"


@pytest.mark.asyncio
async def test_patterns_cached(patterns_module, hub):
    """Detected patterns should be stored in hub cache under 'patterns' key."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        patterns = await patterns_module.detect_patterns()

    cache_entry = await hub.get_cache("patterns")
    assert cache_entry is not None, "patterns cache entry should exist after detect_patterns()"

    data = cache_entry["data"]
    assert "patterns" in data
    assert "pattern_count" in data
    assert "areas_analyzed" in data
    assert data["pattern_count"] == len(patterns)
    assert data["pattern_count"] >= 1


@pytest.mark.asyncio
async def test_golden_snapshot(patterns_module, hub, update_golden):
    """Golden snapshot of normalized pattern output."""
    with patch("aria.modules.patterns.ollama.generate", side_effect=_mock_ollama_generate):
        patterns = await patterns_module.detect_patterns()

    # Normalize patterns for deterministic comparison
    normalized = []
    for p in sorted(patterns, key=lambda x: (x["area"], x.get("typical_time", ""))):
        normalized.append(
            {
                "pattern_id": p["pattern_id"],
                "area": p["area"],
                "day_type": p.get("day_type", "unknown"),
                "typical_time": p["typical_time"],
                "frequency": p["frequency"],
                "confidence": round(p["confidence"], 2),
                "associated_signals": sorted(p.get("associated_signals", [])),
                "llm_description": p.get("llm_description", ""),
                "has_entity_chain": len(p.get("entity_chain", [])) > 0,
                "has_trigger_entity": bool(p.get("trigger_entity")),
            }
        )

    snapshot = {"patterns": normalized, "pattern_count": len(normalized)}
    golden_compare(snapshot, "patterns_detection", update=update_golden)

    # Verify snapshot is non-empty
    assert len(normalized) >= 1, "Golden snapshot should contain at least 1 pattern"
