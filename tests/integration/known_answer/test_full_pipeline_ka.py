"""Known-answer tests for the full ARIA pipeline (end-to-end).

Traces one scenario through the complete data flow:
  engine output (HouseholdSimulator + PipelineRunner)
  -> hub modules (IntelligenceModule reads files)
  -> final recommendations (OrchestratorModule generates suggestions)

Uses scope="module" on the engine fixture since it's expensive (generates
21 days of intraday data, trains ML models, and scores predictions).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from aria.automation.models import ShadowResult
from aria.hub.constants import CACHE_INTELLIGENCE
from aria.hub.core import IntelligenceHub
from aria.modules.automation_generator import AutomationGeneratorModule
from aria.modules.intelligence import IntelligenceModule
from aria.modules.orchestrator import OrchestratorModule
from tests.integration.known_answer.conftest import golden_compare
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator

# ---------------------------------------------------------------------------
# Module-scoped engine fixture — expensive, run once per test module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine_output(tmp_path_factory):
    """Run the full engine pipeline once and cache results for all tests.

    Generates 21 days of household data using the 'couple' scenario,
    runs baselines + training + predictions + scoring.
    """
    data_dir = tmp_path_factory.mktemp("engine_data")

    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42, start_date="2026-02-01")
    snapshots = sim.generate()

    runner = PipelineRunner(snapshots=snapshots, data_dir=data_dir)
    result = runner.run_full()

    return {
        "result": result,
        "snapshots": snapshots,
        "data_dir": data_dir,
    }


# ---------------------------------------------------------------------------
# Test 1: Engine produces predictions and scores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_produces_predictions(engine_output):
    """Run engine pipeline (HouseholdSimulator + PipelineRunner), verify
    predictions and scores exist with expected structure."""
    result = engine_output["result"]
    snapshots = engine_output["snapshots"]

    # Pipeline ran to completion
    assert result["snapshots_saved"] == len(snapshots)
    assert result["snapshots_saved"] > 0

    # Baselines computed
    baselines = result["baselines"]
    assert isinstance(baselines, dict)
    assert len(baselines) > 0, "Baselines should have at least one metric"

    # Training produced results for at least one metric
    training = result["training"]
    assert isinstance(training, dict)
    assert len(training) > 0, "Training should attempt at least one metric"

    trained_metrics = [m for m, r in training.items() if "error" not in r]
    assert len(trained_metrics) > 0, "At least one metric should train successfully"

    # Predictions generated
    predictions = result["predictions"]
    assert isinstance(predictions, dict)
    assert len(predictions) > 0, "Predictions dict should not be empty"

    # Scores computed
    scores = result["scores"]
    assert isinstance(scores, dict)
    assert len(scores) > 0, "Scores dict should not be empty"


# ---------------------------------------------------------------------------
# Test 2: Hub reads engine output from disk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hub_reads_engine_output(engine_output, tmp_path):
    """Write engine output to disk in the format IntelligenceModule expects,
    then verify it reads into hub cache correctly.

    Note: engine_output is module-scoped (shared, expensive), while tmp_path is
    function-scoped — each test gets its own hub DB and intelligence directory.
    """
    result = engine_output["result"]

    # Build the intelligence directory structure
    intel_dir = tmp_path / "intelligence"
    intel_dir.mkdir()

    # Write predictions.json
    (intel_dir / "predictions.json").write_text(json.dumps(result["predictions"]))

    # Write baselines.json
    (intel_dir / "baselines.json").write_text(json.dumps(result["baselines"]))

    # Write accuracy.json from scores
    # scores has: {"overall": int, "metrics": {"power_watts": {"accuracy": ...}, ...}}
    accuracy = {
        "overall": result["scores"].get("overall", 0),
        "by_metric": {
            k: v.get("accuracy", 0)
            for k, v in result["scores"].get("metrics", {}).items()
            if isinstance(v, dict) and "accuracy" in v
        },
    }
    (intel_dir / "accuracy.json").write_text(json.dumps(accuracy))

    # Create required subdirectories
    (intel_dir / "daily").mkdir()
    (intel_dir / "insights").mkdir()
    (intel_dir / "models").mkdir()
    (intel_dir / "meta-learning").mkdir()

    # Write a daily snapshot from the last engine snapshot
    last_snap = engine_output["snapshots"][-1]
    daily_snapshot = {
        "power": {"total_watts": last_snap.get("power", {}).get("total_watts", 0)},
        "lights": {"on": last_snap.get("lights", {}).get("on", 0)},
        "occupancy": {"device_count_home": last_snap.get("occupancy", {}).get("device_count_home", 0)},
        "entities": {"unavailable": last_snap.get("entities_summary", {}).get("unavailable_count", 0)},
        "logbook_summary": {"useful_events": last_snap.get("logbook_summary", {}).get("useful_events", 0)},
    }
    (intel_dir / "daily" / "2026-02-21.json").write_text(json.dumps(daily_snapshot))

    # Create hub and intelligence module
    hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    try:
        await hub.initialize()

        module = IntelligenceModule(hub=hub, intelligence_dir=str(intel_dir))

        # Mock out external data reads
        async def mock_activity():
            return {"activity_log": None, "activity_summary": None}

        module._read_activity_data = mock_activity
        module._parse_error_log = lambda: []

        await module.initialize()

        # Verify intelligence cache is populated
        entry = await hub.get_cache(CACHE_INTELLIGENCE)
        assert entry is not None, "Intelligence cache should be populated"

        data = entry["data"]

        # Predictions should match what the engine produced
        assert data["predictions"] is not None
        assert data["predictions"] == result["predictions"]

        # Baselines should match
        assert data["baselines"] is not None
        assert data["baselines"] == result["baselines"]

        # Data maturity reflects our 1 daily file
        maturity = data["data_maturity"]
        assert maturity["days_of_data"] == 1
        assert maturity["phase"] == "collecting"

        # Trend data extracted from our daily snapshot
        assert len(data["trend_data"]) == 1
        assert data["trend_data"][0]["date"] == "2026-02-21"
    finally:
        await hub.shutdown()


# ---------------------------------------------------------------------------
# Test 3: Patterns -> suggestions flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patterns_to_suggestions_flow(tmp_path):
    """Pre-populate patterns cache, then verify orchestrator delegates to
    AutomationGeneratorModule and produces suggestions."""
    hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    try:
        await hub.initialize()

        # Pre-populate patterns cache with detection-format data
        patterns_data = {
            "detections": [
                {
                    "source": "pattern",
                    "trigger_entity": "binary_sensor.motion_bedroom",
                    "action_entities": ["light.bedroom"],
                    "entity_chain": [
                        {"entity_id": "binary_sensor.motion_bedroom", "state": "on", "offset_seconds": 0},
                    ],
                    "area_id": "bedroom",
                    "confidence": 0.85,
                    "recency_weight": 0.7,
                    "observation_count": 18,
                    "first_seen": "2026-01-01T00:00:00",
                    "last_seen": "2026-02-10T00:00:00",
                    "day_type": "workday",
                },
                {
                    "source": "pattern",
                    "trigger_entity": "binary_sensor.motion_kitchen",
                    "action_entities": ["light.kitchen"],
                    "entity_chain": [
                        {"entity_id": "binary_sensor.motion_kitchen", "state": "on", "offset_seconds": 0},
                    ],
                    "area_id": "kitchen",
                    "confidence": 0.78,
                    "recency_weight": 0.6,
                    "observation_count": 14,
                    "first_seen": "2026-01-05T00:00:00",
                    "last_seen": "2026-02-08T00:00:00",
                    "day_type": "workday",
                },
            ],
        }
        await hub.set_cache("patterns", patterns_data)

        # Register AutomationGeneratorModule
        generator = AutomationGeneratorModule(hub=hub, top_n=10, min_confidence=0.7)
        hub.modules["automation_generator"] = generator

        # Create orchestrator (no initialize — skip aiohttp session)
        orchestrator = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )

        # Patch LLM refiner, validator, and shadow comparison
        with (
            patch(
                "aria.modules.automation_generator.refine_automation",
                new_callable=AsyncMock,
                side_effect=lambda auto, **kw: auto,
            ),
            patch(
                "aria.modules.automation_generator.validate_automation",
                return_value=(True, []),
            ),
            patch(
                "aria.modules.automation_generator.compare_candidate",
                return_value=ShadowResult(
                    candidate={},
                    status="new",
                    duplicate_score=0.0,
                    conflicting_automation=None,
                    gap_source_automation=None,
                    reason="No match found",
                ),
            ),
        ):
            suggestions = await orchestrator.generate_suggestions()

        # Both detections are above min_confidence, should produce suggestions
        assert len(suggestions) == 2, f"Expected 2 suggestions, got {len(suggestions)}"

        # Each suggestion references the correct trigger entity
        trigger_entities = {s["metadata"]["trigger_entity"] for s in suggestions}
        assert "binary_sensor.motion_bedroom" in trigger_entities
        assert "binary_sensor.motion_kitchen" in trigger_entities

        # Suggestions have automation YAML with actions
        for s in suggestions:
            yaml = s["automation_yaml"]
            assert "triggers" in yaml or "trigger" in yaml
            assert "actions" in yaml or "action" in yaml

        # Suggestions are cached
        cache_entry = await hub.get_cache("automation_suggestions")
        assert cache_entry is not None
        assert cache_entry["data"]["count"] == 2
    finally:
        await hub.shutdown()


# ---------------------------------------------------------------------------
# Test 4: Golden snapshot of full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_golden_snapshot(engine_output, tmp_path, update_golden):
    """Golden snapshot of full pipeline output — engine scores + suggestion count.

    Captures a deterministic summary of the end-to-end flow: engine pipeline
    results, hub intelligence loading, and orchestrator suggestion generation.

    Note: engine_output is module-scoped (shared), tmp_path is function-scoped.

    # This test must run AFTER test_engine_produces_predictions and
    # test_hub_reads_engine_output because they share the module-scoped
    # engine_output fixture. pytest runs tests in file order by default,
    # so placement in this file defines the execution sequence.
    """
    result = engine_output["result"]

    # --- Engine summary (deterministic with seed=42) ---
    training = result["training"]
    trained_metrics = sorted([m for m, r in training.items() if "error" not in r])
    failed_metrics = sorted([m for m, r in training.items() if "error" in r])

    scores = result["scores"]
    # Per-metric scores live under scores["metrics"], not at the top level
    score_summary = {}
    for metric, score_data in sorted(scores.get("metrics", {}).items()):
        if isinstance(score_data, dict) and "accuracy" in score_data:
            score_summary[metric] = round(score_data["accuracy"], 1)

    # --- Hub intelligence loading ---
    intel_dir = tmp_path / "intelligence"
    intel_dir.mkdir()
    (intel_dir / "predictions.json").write_text(json.dumps(result["predictions"]))
    (intel_dir / "baselines.json").write_text(json.dumps(result["baselines"]))
    (intel_dir / "daily").mkdir()
    (intel_dir / "insights").mkdir()
    (intel_dir / "models").mkdir()
    (intel_dir / "meta-learning").mkdir()

    hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    try:
        await hub.initialize()

        module = IntelligenceModule(hub=hub, intelligence_dir=str(intel_dir))

        async def mock_activity():
            return {"activity_log": None, "activity_summary": None}

        module._read_activity_data = mock_activity
        module._parse_error_log = lambda: []

        await module.initialize()

        intel_entry = await hub.get_cache(CACHE_INTELLIGENCE)
        intel_loaded = intel_entry is not None and intel_entry["data"]["predictions"] is not None

        # --- Orchestrator suggestion count ---
        patterns_data = {
            "detections": [
                {
                    "source": "pattern",
                    "trigger_entity": "binary_sensor.motion_bedroom",
                    "action_entities": ["light.bedroom"],
                    "entity_chain": [
                        {"entity_id": "binary_sensor.motion_bedroom", "state": "on", "offset_seconds": 0},
                    ],
                    "area_id": "bedroom",
                    "confidence": 0.90,
                    "recency_weight": 0.8,
                    "observation_count": 18,
                    "first_seen": "2026-01-01T00:00:00",
                    "last_seen": "2026-02-10T00:00:00",
                    "day_type": "workday",
                },
            ],
        }
        await hub.set_cache("patterns", patterns_data)

        # Register AutomationGeneratorModule
        generator = AutomationGeneratorModule(hub=hub, top_n=10, min_confidence=0.7)
        hub.modules["automation_generator"] = generator

        orchestrator = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )

        with (
            patch(
                "aria.modules.automation_generator.refine_automation",
                new_callable=AsyncMock,
                side_effect=lambda auto, **kw: auto,
            ),
            patch(
                "aria.modules.automation_generator.validate_automation",
                return_value=(True, []),
            ),
            patch(
                "aria.modules.automation_generator.compare_candidate",
                return_value=ShadowResult(
                    candidate={},
                    status="new",
                    duplicate_score=0.0,
                    conflicting_automation=None,
                    gap_source_automation=None,
                    reason="No match found",
                ),
            ),
        ):
            suggestions = await orchestrator.generate_suggestions()

        # Build golden snapshot
        snapshot = {
            "engine": {
                "snapshots_saved": result["snapshots_saved"],
                "baseline_metrics": sorted(result["baselines"].keys()),
                "trained_metrics": trained_metrics,
                "failed_metrics": failed_metrics,
                "score_summary": score_summary,
            },
            "hub": {
                "intelligence_loaded": intel_loaded,
                "predictions_present": result["predictions"] is not None,
                "baselines_present": result["baselines"] is not None,
            },
            "orchestrator": {
                "suggestion_count": len(suggestions),
                "trigger_entities": sorted(s["metadata"]["trigger_entity"] for s in suggestions),
            },
        }

        golden_compare(snapshot, "full_pipeline", update=update_golden)

        # Structural assertions that must always hold
        assert snapshot["engine"]["snapshots_saved"] > 0
        assert len(snapshot["engine"]["trained_metrics"]) > 0
        assert snapshot["hub"]["intelligence_loaded"] is True
        assert snapshot["orchestrator"]["suggestion_count"] >= 1
    finally:
        await hub.shutdown()
