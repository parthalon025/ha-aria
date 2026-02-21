"""Known-answer tests for OrchestratorModule.

Validates suggestion generation through the AutomationGeneratorModule
delegation path, confidence filtering, required field structure, cache
storage, and golden snapshot stability.

Updated for Task 28 (orchestrator thinning): orchestrator delegates
suggestion generation to AutomationGeneratorModule.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aria.automation.models import ShadowResult
from aria.modules.automation_generator import AutomationGeneratorModule
from aria.modules.orchestrator import OrchestratorModule
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Known patterns fixture — pre-populated into hub cache before each test.
#
# Uses the "detections" format that AutomationGeneratorModule expects,
# replacing the old "patterns" format that _pattern_to_suggestion read.
# ---------------------------------------------------------------------------

KNOWN_DETECTIONS: dict[str, Any] = {
    "detections": [
        {
            "source": "pattern",
            "trigger_entity": "binary_sensor.motion_kitchen",
            "action_entities": ["light.kitchen"],
            "entity_chain": [
                {"entity_id": "binary_sensor.motion_kitchen", "state": "on", "offset_seconds": 0},
            ],
            "area_id": "kitchen",
            "confidence": 0.92,
            "recency_weight": 0.8,
            "observation_count": 15,
            "first_seen": "2026-01-01T00:00:00",
            "last_seen": "2026-02-10T00:00:00",
            "day_type": "workday",
        },
        {
            "source": "pattern",
            "trigger_entity": "binary_sensor.motion_living_room",
            "action_entities": ["light.living_room", "media_player.tv"],
            "entity_chain": [
                {"entity_id": "binary_sensor.motion_living_room", "state": "on", "offset_seconds": 0},
            ],
            "area_id": "living_room",
            "confidence": 0.88,
            "recency_weight": 0.7,
            "observation_count": 12,
            "first_seen": "2026-01-05T00:00:00",
            "last_seen": "2026-02-08T00:00:00",
            "day_type": "workday",
        },
    ],
}

# A low-confidence detection that should be filtered out at default threshold.
LOW_CONFIDENCE_DETECTION: dict[str, Any] = {
    "source": "pattern",
    "trigger_entity": "binary_sensor.motion_garage",
    "action_entities": ["light.garage"],
    "entity_chain": [
        {"entity_id": "binary_sensor.motion_garage", "state": "on", "offset_seconds": 0},
    ],
    "area_id": "garage",
    "confidence": 0.45,
    "recency_weight": 0.3,
    "observation_count": 3,
    "first_seen": "2026-01-15T00:00:00",
    "last_seen": "2026-02-05T00:00:00",
    "day_type": "all",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def orchestrator(hub) -> OrchestratorModule:
    """Create an OrchestratorModule with AutomationGeneratorModule registered.

    Pre-populates patterns cache and patches LLM refiner + validator + shadow.
    Does NOT call initialize() — tests call generate_suggestions() directly.
    """
    # Pre-populate the patterns cache
    await hub.set_cache("patterns", KNOWN_DETECTIONS)

    # Create and register the generator module
    generator = AutomationGeneratorModule(hub=hub, top_n=10, min_confidence=0.7)
    hub.modules["automation_generator"] = generator

    module = OrchestratorModule(
        hub=hub,
        ha_url="http://test-host:8123",
        ha_token="test-token",
        min_confidence=0.7,
    )
    return module


@pytest.fixture
async def orchestrator_with_low_conf(hub) -> OrchestratorModule:
    """Orchestrator with an extra low-confidence detection in the cache."""
    patterns_with_low = {
        "detections": KNOWN_DETECTIONS["detections"] + [LOW_CONFIDENCE_DETECTION],
    }
    await hub.set_cache("patterns", patterns_with_low)

    generator = AutomationGeneratorModule(hub=hub, top_n=10, min_confidence=0.7)
    hub.modules["automation_generator"] = generator

    module = OrchestratorModule(
        hub=hub,
        ha_url="http://test-host:8123",
        ha_token="test-token",
        min_confidence=0.7,
    )
    return module


def _patch_pipeline():
    """Context manager that patches LLM refiner, validator, and shadow."""
    refine_patch = patch(
        "aria.modules.automation_generator.refine_automation",
        new_callable=AsyncMock,
        side_effect=lambda auto, **kw: auto,
    )
    validate_patch = patch(
        "aria.modules.automation_generator.validate_automation",
        return_value=(True, []),
    )
    shadow_patch = patch(
        "aria.modules.automation_generator.compare_candidate",
        return_value=ShadowResult(
            candidate={},
            status="new",
            duplicate_score=0.0,
            conflicting_automation=None,
            gap_source_automation=None,
            reason="No match found",
        ),
    )
    return refine_patch, validate_patch, shadow_patch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generates_suggestions_from_patterns(orchestrator):
    """Pre-populated patterns cache produces at least 1 suggestion."""
    rp, vp, sp = _patch_pipeline()
    with rp, vp, sp:
        suggestions = await orchestrator.generate_suggestions()
    assert len(suggestions) >= 1, "Expected at least 1 suggestion from 2 high-confidence detections"


@pytest.mark.asyncio
async def test_suggestion_has_required_fields(orchestrator):
    """Each suggestion contains fields needed for HA automation creation."""
    rp, vp, sp = _patch_pipeline()
    with rp, vp, sp:
        suggestions = await orchestrator.generate_suggestions()
    assert suggestions, "No suggestions generated"

    required_keys = {
        "suggestion_id",
        "automation_yaml",
        "combined_score",
        "source",
        "status",
        "created_at",
        "metadata",
    }

    for suggestion in suggestions:
        missing = required_keys - set(suggestion.keys())
        assert not missing, f"Suggestion missing keys: {missing}"

        # automation_yaml must have trigger-like and action-like keys
        yaml = suggestion["automation_yaml"]
        assert "triggers" in yaml or "trigger" in yaml, "automation_yaml missing triggers"
        assert "actions" in yaml or "action" in yaml, "automation_yaml missing actions"
        assert "alias" in yaml, "automation_yaml missing 'alias'"


@pytest.mark.asyncio
async def test_low_confidence_filtered(orchestrator_with_low_conf):
    """Detections below min_confidence=0.7 produce no suggestion."""
    rp, vp, sp = _patch_pipeline()
    with rp, vp, sp:
        suggestions = await orchestrator_with_low_conf.generate_suggestions()

    # The low-confidence detection's trigger should not appear
    trigger_entities = {s["metadata"]["trigger_entity"] for s in suggestions}
    assert "binary_sensor.motion_garage" not in trigger_entities, (
        "Low-confidence detection should have been filtered out"
    )
    # The two high-confidence detections should still produce suggestions
    assert len(suggestions) >= 2


@pytest.mark.asyncio
async def test_suggestions_cached(orchestrator):
    """Generated suggestions are stored in hub cache under 'automation_suggestions'."""
    rp, vp, sp = _patch_pipeline()
    with rp, vp, sp:
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
    rp, vp, sp = _patch_pipeline()
    with rp, vp, sp:
        suggestions = await orchestrator.generate_suggestions()

    # Strip non-deterministic fields before comparison.
    stable_suggestions = []
    for s in suggestions:
        stable = {k: v for k, v in s.items() if k != "created_at"}
        stable_suggestions.append(stable)

    snapshot = {
        "suggestion_count": len(stable_suggestions),
        "trigger_entities": sorted(s["metadata"]["trigger_entity"] for s in stable_suggestions),
        "suggestions": stable_suggestions,
    }

    golden_compare(snapshot, "orchestrator_suggestions", update=update_golden)
