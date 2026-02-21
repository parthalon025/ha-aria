"""Known-answer golden tests for the automation generation pipeline.

Predefined event sequences with expected automation outputs. Uses the
golden comparison infrastructure from conftest.py for snapshot stability.

Each scenario represents a common HA pattern that ARIA should detect
and generate a valid automation for.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aria.automation.models import ChainLink, DetectionResult
from aria.automation.template_engine import AutomationTemplate
from aria.automation.validator import validate_automation
from aria.hub.core import IntelligenceHub
from aria.modules.automation_generator import (
    AutomationGeneratorModule,
    compute_combined_score,
)
from aria.shared.shadow_comparison import compare_candidate
from tests.integration.known_answer.conftest import golden_compare

GOLDEN_DIR = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "motion_light": {
        "description": "Motion sensor triggers light in the same area",
        "detection": {
            "source": "pattern",
            "trigger_entity": "binary_sensor.kitchen_motion",
            "action_entities": ["light.kitchen_ceiling"],
            "entity_chain": [
                {"entity_id": "binary_sensor.kitchen_motion", "state": "on", "offset_seconds": 0.0},
                {"entity_id": "light.kitchen_ceiling", "state": "on", "offset_seconds": 3.0},
            ],
            "area_id": "kitchen",
            "confidence": 0.92,
            "recency_weight": 0.88,
            "observation_count": 45,
            "first_seen": "2026-01-01T06:30:00",
            "last_seen": "2026-02-18T07:15:00",
            "day_type": "workday",
        },
    },
    "evening_routine": {
        "description": "Evening arrival triggers multiple lights in living room",
        "detection": {
            "source": "pattern",
            "trigger_entity": "person.alice",
            "action_entities": ["light.living_room_lamp", "light.living_room_ceiling"],
            "entity_chain": [
                {"entity_id": "person.alice", "state": "home", "offset_seconds": 0.0},
                {"entity_id": "light.living_room_lamp", "state": "on", "offset_seconds": 15.0},
                {"entity_id": "light.living_room_ceiling", "state": "on", "offset_seconds": 18.0},
            ],
            "area_id": "living_room",
            "confidence": 0.88,
            "recency_weight": 0.95,
            "observation_count": 30,
            "first_seen": "2026-01-05T17:00:00",
            "last_seen": "2026-02-17T18:30:00",
            "day_type": "workday",
        },
    },
    "gap_hallway_light": {
        "description": "Gap analyzer finds hallway motion without automation",
        "detection": {
            "source": "gap",
            "trigger_entity": "binary_sensor.hallway_motion",
            "action_entities": ["light.hallway_sconce"],
            "entity_chain": [
                {"entity_id": "binary_sensor.hallway_motion", "state": "on", "offset_seconds": 0.0},
                {"entity_id": "light.hallway_sconce", "state": "on", "offset_seconds": 2.0},
            ],
            "area_id": "hallway",
            "confidence": 0.78,
            "recency_weight": 0.65,
            "observation_count": 8,
            "first_seen": "2026-02-01T08:00:00",
            "last_seen": "2026-02-15T20:00:00",
            "day_type": "all",
        },
    },
    "weekend_media": {
        "description": "Weekend media player pattern in living room",
        "detection": {
            "source": "pattern",
            "trigger_entity": "binary_sensor.living_room_motion",
            "action_entities": ["media_player.living_room_tv"],
            "entity_chain": [
                {"entity_id": "binary_sensor.living_room_motion", "state": "on", "offset_seconds": 0.0},
                {"entity_id": "media_player.living_room_tv", "state": "playing", "offset_seconds": 10.0},
            ],
            "area_id": "living_room",
            "confidence": 0.75,
            "recency_weight": 0.80,
            "observation_count": 15,
            "first_seen": "2026-01-11T10:00:00",
            "last_seen": "2026-02-15T11:00:00",
            "day_type": "weekend",
        },
    },
    "multi_action_bedroom": {
        "description": "Bedtime routine: motion triggers light off and fan on",
        "detection": {
            "source": "pattern",
            "trigger_entity": "binary_sensor.bedroom_motion",
            "action_entities": ["light.bedroom_lamp", "fan.bedroom_fan"],
            "entity_chain": [
                {"entity_id": "binary_sensor.bedroom_motion", "state": "on", "offset_seconds": 0.0},
                {"entity_id": "light.bedroom_lamp", "state": "off", "offset_seconds": 5.0},
                {"entity_id": "fan.bedroom_fan", "state": "on", "offset_seconds": 8.0},
            ],
            "area_id": "bedroom",
            "confidence": 0.82,
            "recency_weight": 0.70,
            "observation_count": 20,
            "first_seen": "2026-01-10T22:00:00",
            "last_seen": "2026-02-18T23:00:00",
            "day_type": "all",
        },
    },
}


def _build_detection(raw: dict) -> DetectionResult:
    """Parse a raw scenario dict into a DetectionResult."""
    chain = [
        ChainLink(
            entity_id=c["entity_id"],
            state=c["state"],
            offset_seconds=c["offset_seconds"],
        )
        for c in raw["entity_chain"]
    ]
    return DetectionResult(
        source=raw["source"],
        trigger_entity=raw["trigger_entity"],
        action_entities=raw["action_entities"],
        entity_chain=chain,
        area_id=raw.get("area_id"),
        confidence=raw["confidence"],
        recency_weight=raw["recency_weight"],
        observation_count=raw["observation_count"],
        first_seen=raw["first_seen"],
        last_seen=raw["last_seen"],
        day_type=raw.get("day_type", "all"),
    )


def _build_entity_graph_data(scenarios: dict) -> dict[str, dict]:
    """Build entity_graph entities from all scenarios."""
    entities = {}
    for _name, scenario in scenarios.items():
        det = scenario["detection"]
        area = det.get("area_id")
        for eid in [det["trigger_entity"]] + det["action_entities"]:
            entities[eid] = {
                "area_id": area,
                "device_id": f"dev_{eid.replace('.', '_')}",
            }
        for link in det["entity_chain"]:
            eid = link["entity_id"]
            if eid not in entities:
                entities[eid] = {
                    "area_id": area,
                    "device_id": f"dev_{eid.replace('.', '_')}",
                }
    return entities


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def hub(tmp_path):
    """Create a minimal IntelligenceHub for golden tests."""
    h = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    await h.initialize()
    yield h
    await h.shutdown()


@pytest.fixture
def entity_graph_data():
    """Entity graph data covering all scenarios."""
    return _build_entity_graph_data(SCENARIOS)


# ---------------------------------------------------------------------------
# Template engine golden tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
async def test_template_output_golden(scenario_name, hub, entity_graph_data, update_golden):
    """Template engine output should match golden snapshot for each scenario."""
    scenario = SCENARIOS[scenario_name]
    detection = _build_detection(scenario["detection"])
    compute_combined_score(detection)

    hub.entity_graph.update(entity_graph_data, {}, [])
    template = AutomationTemplate(hub.entity_graph)
    automation = template.build(detection)

    # Normalize for deterministic comparison
    normalized = {
        "id": automation["id"],
        "alias": automation["alias"],
        "mode": automation["mode"],
        "trigger_count": len(automation.get("triggers", [])),
        "condition_count": len(automation.get("conditions", [])),
        "action_count": len(automation.get("actions", [])),
        "has_description": bool(automation.get("description")),
        "trigger_entity": detection.trigger_entity,
        "action_entities": sorted(detection.action_entities),
        "area_id": detection.area_id,
        "day_type": detection.day_type,
    }

    golden_compare(normalized, f"automation_template_{scenario_name}", update=update_golden)

    # Structural assertions that must always hold
    assert automation["id"].startswith("aria_")
    assert len(automation["triggers"]) >= 1
    assert len(automation["actions"]) >= 1


# ---------------------------------------------------------------------------
# Validator golden tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
async def test_validator_passes_golden(scenario_name, hub, entity_graph_data):
    """All golden scenarios should pass validation."""
    scenario = SCENARIOS[scenario_name]
    detection = _build_detection(scenario["detection"])

    hub.entity_graph.update(entity_graph_data, {}, [])
    template = AutomationTemplate(hub.entity_graph)
    automation = template.build(detection)

    valid, errors = validate_automation(automation, hub.entity_graph, set())
    assert valid, f"Scenario '{scenario_name}' failed validation: {errors}"


# ---------------------------------------------------------------------------
# Shadow comparison golden tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_name", list(SCENARIOS.keys()))
async def test_shadow_new_golden(scenario_name, hub, entity_graph_data):
    """With no existing HA automations, shadow should classify as 'new'."""
    scenario = SCENARIOS[scenario_name]
    detection = _build_detection(scenario["detection"])

    hub.entity_graph.update(entity_graph_data, {}, [])
    template = AutomationTemplate(hub.entity_graph)
    automation = template.build(detection)

    shadow = compare_candidate(automation, [], hub.entity_graph)
    assert shadow.status == "new", (
        f"Scenario '{scenario_name}' expected 'new' shadow status, got '{shadow.status}': {shadow.reason}"
    )


# ---------------------------------------------------------------------------
# Combined scoring golden tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoring_golden(update_golden):
    """Combined scoring snapshot for all scenarios."""
    results = {}
    for name, scenario in sorted(SCENARIOS.items()):
        detection = _build_detection(scenario["detection"])
        score = compute_combined_score(detection)
        results[name] = {
            "combined_score": round(score, 4),
            "source": detection.source,
            "confidence": detection.confidence,
            "recency_weight": detection.recency_weight,
        }

    golden_compare(results, "automation_scoring", update=update_golden)

    # Pattern sources should score higher with pattern weight
    motion_det = _build_detection(SCENARIOS["motion_light"]["detection"])
    gap_det = _build_detection(SCENARIOS["gap_hallway_light"]["detection"])
    compute_combined_score(motion_det)
    compute_combined_score(gap_det)
    assert motion_det.combined_score > gap_det.combined_score


# ---------------------------------------------------------------------------
# Full pipeline golden test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_golden(hub, entity_graph_data, update_golden):
    """Full pipeline output should match golden snapshot."""
    hub.entity_graph.update(entity_graph_data, {}, [])

    # Seed all detections into caches
    all_detections = [s["detection"] for s in SCENARIOS.values()]
    pattern_dets = [d for d in all_detections if d["source"] == "pattern"]
    gap_dets = [d for d in all_detections if d["source"] == "gap"]

    await hub.set_cache("patterns", {"detections": pattern_dets}, {"source": "test"})
    await hub.set_cache("gaps", {"detections": gap_dets}, {"source": "test"})

    module = AutomationGeneratorModule(hub, top_n=10, min_confidence=0.7)

    with patch("aria.modules.automation_generator.refine_automation", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = lambda auto: auto
        suggestions = await module.generate_suggestions()

    # Normalize for golden comparison
    normalized = {
        "suggestion_count": len(suggestions),
        "suggestions": [],
    }
    for s in sorted(suggestions, key=lambda x: x["suggestion_id"]):
        normalized["suggestions"].append(
            {
                "suggestion_id": s["suggestion_id"],
                "source": s["source"],
                "combined_score": round(s["combined_score"], 4),
                "shadow_status": s["shadow_status"],
                "status": s["status"],
                "trigger_entity": s["metadata"]["trigger_entity"],
                "action_entities": sorted(s["metadata"]["action_entities"]),
                "area_id": s["metadata"]["area_id"],
                "day_type": s["metadata"]["day_type"],
                "automation_id": s["automation_yaml"].get("id", ""),
                "automation_mode": s["automation_yaml"].get("mode", ""),
            }
        )

    golden_compare(normalized, "automation_pipeline_full", update=update_golden)

    # Structural assertions
    assert len(suggestions) >= 1
    for s in suggestions:
        assert s["status"] == "pending"
        assert s["shadow_status"] in {"new", "gap_fill", "conflict"}
