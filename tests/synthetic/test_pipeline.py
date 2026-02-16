"""Tests for PipelineRunner — full pipeline orchestration."""

import pytest

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import INTRADAY_HOURS, HouseholdSimulator


@pytest.fixture
def runner(tmp_path):
    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
    snapshots = sim.generate()
    return PipelineRunner(snapshots, data_dir=tmp_path)


class TestPipelineRunner:
    def test_save_snapshots(self, runner):
        runner.save_snapshots()
        daily_dir = runner.store.paths.daily_dir
        assert daily_dir.exists()
        files = list(daily_dir.glob("*.json"))
        # Multiple intraday snapshots per day overwrite the same {date}.json
        assert len(files) == 21

    def test_compute_baselines(self, runner):
        runner.save_snapshots()
        baselines = runner.compute_baselines()
        assert isinstance(baselines, dict)
        assert len(baselines) > 0

    def test_build_training_data(self, runner):
        runner.save_snapshots()
        names, X, targets = runner.build_training_data()
        assert len(names) > 0
        assert len(X) > 0
        assert "power_watts" in targets

    def test_train_models(self, runner):
        runner.save_snapshots()
        results = runner.train_models()
        assert isinstance(results, dict)
        models_dir = runner.store.paths.models_dir
        assert models_dir.exists()

    def test_generate_predictions(self, runner):
        runner.save_snapshots()
        runner.compute_baselines()
        runner.train_models()
        predictions = runner.generate_predictions()
        assert "prediction_method" in predictions or "power_watts" in predictions

    def test_score_predictions(self, runner):
        runner.save_snapshots()
        runner.compute_baselines()
        runner.train_models()
        runner.generate_predictions()
        scores = runner.score_predictions()
        assert "overall" in scores
        assert "metrics" in scores

    def test_run_full_pipeline(self, runner):
        """End-to-end: all stages complete without error."""
        result = runner.run_full()
        assert "snapshots_saved" in result
        assert "baselines" in result
        assert "training" in result
        assert "predictions" in result
        assert "scores" in result
        assert result["snapshots_saved"] == 21 * len(INTRADAY_HOURS)
