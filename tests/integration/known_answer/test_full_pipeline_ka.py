"""Known-answer tests for the full ARIA pipeline (end-to-end).

Traces one scenario through the complete data flow:
  engine output (HouseholdSimulator + PipelineRunner)
  -> hub modules (IntelligenceModule reads files)
  -> final recommendations (OrchestratorModule generates suggestions)

Uses scope="module" on the engine fixture since it's expensive (generates
21 days of intraday data, trains ML models, and scores predictions).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from aria.hub.constants import CACHE_INTELLIGENCE
from aria.hub.core import IntelligenceHub
from aria.modules.intelligence import IntelligenceModule
from aria.modules.orchestrator import OrchestratorModule
from tests.integration.known_answer.conftest import golden_compare
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator

# ---------------------------------------------------------------------------
# Module-scoped engine fixture — expensive, run once per test module
# ---------------------------------------------------------------------------

_engine_result: dict[str, Any] | None = None
_engine_snapshots: list[dict] | None = None
_engine_data_dir: Path | None = None


@pytest.fixture(scope="module")
def engine_output(tmp_path_factory):
    """Run the full engine pipeline once and cache results for all tests.

    Generates 21 days of household data using the 'couple' scenario,
    runs baselines + training + predictions + scoring.
    """
    global _engine_result, _engine_snapshots, _engine_data_dir

    data_dir = tmp_path_factory.mktemp("engine_data")

    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42, start_date="2026-02-01")
    snapshots = sim.generate()

    runner = PipelineRunner(snapshots=snapshots, data_dir=data_dir)
    result = runner.run_full()

    _engine_result = result
    _engine_snapshots = snapshots
    _engine_data_dir = data_dir

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
    """Pre-populate patterns cache, then verify orchestrator produces
    automation suggestions from those patterns."""
    hub = IntelligenceHub(cache_path=str(tmp_path / "hub.db"))
    try:
        await hub.initialize()

        # Pre-populate patterns cache with realistic patterns
        patterns_data = {
            "patterns": [
                {
                    "pattern_id": "ka_pipeline_morning",
                    "name": "Morning Lights",
                    "area": "bedroom",
                    "entities": ["light.bedroom"],
                    "typical_time": "07:30",
                    "variance_minutes": 10,
                    "confidence": 0.85,
                    "frequency": 18,
                    "total_days": 21,
                    "type": "temporal_sequence",
                    "associated_signals": ["bedroom_light_on_h7"],
                    "llm_description": "Bedroom lights turn on at wake-up time",
                },
                {
                    "pattern_id": "ka_pipeline_evening",
                    "name": "Evening Routine",
                    "area": "kitchen",
                    "entities": ["light.kitchen"],
                    "typical_time": "18:00",
                    "variance_minutes": 15,
                    "confidence": 0.78,
                    "frequency": 14,
                    "total_days": 21,
                    "type": "temporal_sequence",
                    "associated_signals": ["kitchen_light_on_h18"],
                    "llm_description": "Kitchen lights on for evening cooking",
                },
            ],
            "pattern_count": 2,
            "areas_analyzed": ["bedroom", "kitchen"],
        }
        await hub.set_cache("patterns", patterns_data)

        # Create orchestrator (no initialize — skip aiohttp session)
        orchestrator = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )

        suggestions = await orchestrator.generate_suggestions()

        # Both patterns are above min_confidence, should produce suggestions
        assert len(suggestions) == 2, f"Expected 2 suggestions, got {len(suggestions)}"

        # Each suggestion references the correct pattern
        pattern_ids = {s["pattern_id"] for s in suggestions}
        assert "ka_pipeline_morning" in pattern_ids
        assert "ka_pipeline_evening" in pattern_ids

        # Suggestions have automation YAML with actions
        for s in suggestions:
            yaml = s["automation_yaml"]
            assert "trigger" in yaml
            assert "action" in yaml
            assert len(yaml["action"]) >= 1

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
            "patterns": [
                {
                    "pattern_id": "golden_morning",
                    "name": "Morning Pattern",
                    "area": "bedroom",
                    "entities": ["light.bedroom"],
                    "typical_time": "07:00",
                    "variance_minutes": 10,
                    "confidence": 0.90,
                    "frequency": 18,
                    "total_days": 21,
                    "type": "temporal_sequence",
                    "associated_signals": ["bedroom_light_on_h7"],
                    "llm_description": "Wake-up lights",
                },
            ],
            "pattern_count": 1,
            "areas_analyzed": ["bedroom"],
        }
        await hub.set_cache("patterns", patterns_data)

        orchestrator = OrchestratorModule(
            hub=hub,
            ha_url="http://test-host:8123",
            ha_token="test-token",
            min_confidence=0.7,
        )
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
                "pattern_ids": sorted(s["pattern_id"] for s in suggestions),
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
