"""Known-answer tests for OrchestratorModule.

Validates suggestion generation from pre-populated pattern cache,
confidence filtering, required field structure, cache storage,
and golden snapshot stability.
"""

from typing import Any

import pytest

from aria.modules.orchestrator import OrchestratorModule
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Known patterns fixture — pre-populated into hub cache before each test.
#
# Field names match what OrchestratorModule._pattern_to_suggestion reads:
#   pattern_id, area, typical_time, confidence, frequency, total_days,
#   associated_signals, llm_description
# ---------------------------------------------------------------------------

KNOWN_PATTERNS: dict[str, Any] = {
    "patterns": [
        {
            "pattern_id": "ka_morning_kitchen",
            "name": "Morning Kitchen Routine",
            "area": "Kitchen",
            "entities": ["light.kitchen", "switch.coffee_maker"],
            "typical_time": "07:00",
            "variance_minutes": 10,
            "confidence": 0.92,
            "frequency": 15,
            "total_days": 20,
            "type": "temporal_sequence",
            "associated_signals": ["Kitchen_light_on_h7"],
            "llm_description": "Morning kitchen routine: lights then coffee maker",
        },
        {
            "pattern_id": "ka_evening_living",
            "name": "Evening Living Room",
            "area": "Living Room",
            "entities": ["light.living_room", "media_player.tv"],
            "typical_time": "18:00",
            "variance_minutes": 15,
            "confidence": 0.88,
            "frequency": 12,
            "total_days": 20,
            "type": "temporal_sequence",
            "associated_signals": [],
            "llm_description": "Evening living room: lights then TV",
        },
    ],
    "pattern_count": 2,
    "areas_analyzed": ["Kitchen", "Living Room"],
}

# A low-confidence pattern that should be filtered out at default threshold.
LOW_CONFIDENCE_PATTERN: dict[str, Any] = {
    "pattern_id": "ka_low_conf",
    "name": "Low Confidence Pattern",
    "area": "Garage",
    "entities": ["light.garage"],
    "typical_time": "22:00",
    "variance_minutes": 30,
    "confidence": 0.45,
    "frequency": 3,
    "total_days": 20,
    "type": "temporal_sequence",
    "associated_signals": [],
    "llm_description": "Infrequent garage light at night",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def orchestrator(hub) -> OrchestratorModule:
    """Create an OrchestratorModule with known patterns pre-loaded in cache.

    Does NOT call initialize() — that would create an aiohttp session and
    schedule tasks.  Tests call generate_suggestions() directly.
    """
    # Pre-populate the patterns cache so generate_suggestions() finds data.
    await hub.set_cache("patterns", KNOWN_PATTERNS)

    module = OrchestratorModule(
        hub=hub,
        ha_url="http://test-host:8123",
        ha_token="test-token",
        min_confidence=0.7,
    )
    return module


@pytest.fixture
async def orchestrator_with_low_conf(hub) -> OrchestratorModule:
    """Orchestrator with an extra low-confidence pattern in the cache."""
    patterns_with_low = {
        **KNOWN_PATTERNS,
        "patterns": KNOWN_PATTERNS["patterns"] + [LOW_CONFIDENCE_PATTERN],
        "pattern_count": 3,
    }
    await hub.set_cache("patterns", patterns_with_low)

    module = OrchestratorModule(
        hub=hub,
        ha_url="http://test-host:8123",
        ha_token="test-token",
        min_confidence=0.7,
    )
    return module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generates_suggestions_from_patterns(orchestrator):
    """Pre-populated patterns cache produces at least 1 suggestion."""
    suggestions = await orchestrator.generate_suggestions()
    assert len(suggestions) >= 1, "Expected at least 1 suggestion from 2 high-confidence patterns"


@pytest.mark.asyncio
async def test_suggestion_has_required_fields(orchestrator):
    """Each suggestion contains fields needed for HA automation creation."""
    suggestions = await orchestrator.generate_suggestions()
    assert suggestions, "No suggestions generated"

    required_keys = {
        "suggestion_id",
        "pattern_id",
        "automation_yaml",
        "confidence",
        "status",
        "created_at",
        "metadata",
    }

    for suggestion in suggestions:
        missing = required_keys - set(suggestion.keys())
        assert not missing, f"Suggestion missing keys: {missing}"

        # automation_yaml must have trigger and action
        yaml = suggestion["automation_yaml"]
        assert "trigger" in yaml, "automation_yaml missing 'trigger'"
        assert "action" in yaml, "automation_yaml missing 'action'"
        assert "alias" in yaml, "automation_yaml missing 'alias'"


@pytest.mark.asyncio
async def test_low_confidence_filtered(orchestrator_with_low_conf):
    """Patterns below min_confidence=0.7 produce no suggestion."""
    suggestions = await orchestrator_with_low_conf.generate_suggestions()

    suggestion_pattern_ids = {s["pattern_id"] for s in suggestions}
    assert "ka_low_conf" not in suggestion_pattern_ids, "Low-confidence pattern should have been filtered out"
    # The two high-confidence patterns should still be present.
    assert "ka_morning_kitchen" in suggestion_pattern_ids
    assert "ka_evening_living" in suggestion_pattern_ids


@pytest.mark.asyncio
async def test_suggestions_cached(orchestrator):
    """Generated suggestions are stored in hub cache under 'automation_suggestions'."""
    suggestions = await orchestrator.generate_suggestions()

    cache_entry = await orchestrator.hub.get_cache("automation_suggestions")
    assert cache_entry is not None, "automation_suggestions not found in cache"

    cached_data = cache_entry["data"]
    assert "suggestions" in cached_data
    assert cached_data["count"] == len(suggestions)
    assert cached_data["suggestions"] == suggestions


@pytest.mark.asyncio
async def test_golden_snapshot(orchestrator, update_golden):
    """Golden snapshot of suggestions output — drift reported as warning, not failure."""
    suggestions = await orchestrator.generate_suggestions()

    # Strip non-deterministic fields before comparison.
    stable_suggestions = []
    for s in suggestions:
        stable = {k: v for k, v in s.items() if k != "created_at"}
        stable_suggestions.append(stable)

    snapshot = {
        "suggestion_count": len(stable_suggestions),
        "pattern_ids": sorted(s["pattern_id"] for s in stable_suggestions),
        "suggestions": stable_suggestions,
    }

    golden_compare(snapshot, "orchestrator_suggestions", update=update_golden)
