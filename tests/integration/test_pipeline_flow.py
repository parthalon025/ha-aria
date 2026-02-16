# tests/integration/test_pipeline_flow.py
"""Tier 3: End-to-end pipeline flow and handoff validation."""

import json

import pytest

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import INTRADAY_HOURS, HouseholdSimulator


class TestFullPipelineCompletes:
    """Full pipeline should run to completion with various scenarios."""

    @pytest.mark.parametrize(
        "scenario,days",
        [
            ("stable_couple", 21),
            ("vacation", 14),
            ("work_from_home", 14),
            ("sensor_degradation", 30),
        ],
    )
    def test_scenario_completes(self, tmp_path, scenario, days):
        sim = HouseholdSimulator(scenario=scenario, days=days, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()
        assert result["snapshots_saved"] == days * len(INTRADAY_HOURS)
        assert result["scores"] is not None


class TestIntermediateFormats:
    """Each stage's output should match the expected schema."""

    def test_snapshot_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()

        loaded = runner.store.load_snapshot(snapshots[0]["date"])
        assert loaded is not None
        required_keys = [
            "date",
            "day_of_week",
            "power",
            "lights",
            "occupancy",
            "climate",
            "locks",
            "motion",
            "entities",
            "weather",
        ]
        for key in required_keys:
            assert key in loaded, f"Snapshot missing key: {key}"

    def test_baselines_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        baselines = runner.compute_baselines()

        for _day_name, day_data in baselines.items():
            assert "sample_count" in day_data
            assert "power_watts" in day_data
            assert "mean" in day_data["power_watts"]

    def test_predictions_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.run_full()

        predictions = runner._predictions
        assert "prediction_method" in predictions or "power_watts" in predictions

    def test_scores_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()

        scores = result["scores"]
        assert "overall" in scores
        assert "metrics" in scores
        assert isinstance(scores["overall"], (int, float))


class TestHubReadsEngineOutput:
    """Hub's IntelligenceModule should be able to read engine-produced files."""

    def test_hub_loads_baselines(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        runner.compute_baselines()

        baselines_path = runner.paths.baselines_path
        assert baselines_path.exists()
        with open(baselines_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_hub_loads_predictions(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.run_full()

        predictions_path = runner.paths.predictions_path
        assert predictions_path.exists()
        with open(predictions_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "prediction_method" in data or "power_watts" in data

    def test_hub_loads_models(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        runner.train_models()

        models_dir = runner.paths.models_dir
        assert models_dir.exists()
        pkl_files = list(models_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1
