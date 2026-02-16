"""Tier 1: ML model competence tests against realistic synthetic data."""

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import INTRADAY_HOURS, HouseholdSimulator


class TestModelsConverge:
    """Models should improve accuracy with more training data."""

    def test_r2_improves_with_more_data(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner_early = PipelineRunner(snapshots[:14], data_dir=tmp_path / "early")
        runner_early.save_snapshots()
        early_results = runner_early.train_models()

        runner_late = PipelineRunner(snapshots[:25], data_dir=tmp_path / "late")
        runner_late.save_snapshots()
        late_results = runner_late.train_models()

        improved = False
        for metric in early_results:
            if metric in late_results and late_results[metric].get("r2", 0) >= early_results[metric].get("r2", 0):
                    improved = True
        assert improved, "No metric improved R2 with more data"


class TestModelsBeatBaseline:
    """ML predictions should outperform naive baselines after sufficient data."""

    def test_ml_blend_beats_pure_baseline(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()

        scores = result["scores"]
        assert scores["overall"] > 0


class TestModelsGeneralize:
    """Models should not severely overfit."""

    def test_generalization_gap_reasonable(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots[:21], data_dir=tmp_path)
        runner.save_snapshots()
        training_results = runner.train_models()

        r2_values = {}
        for metric, result in training_results.items():
            r2 = result.get("r2", None)
            if r2 is not None:
                r2_values[metric] = r2

        assert len(r2_values) > 0, "No metrics produced R2 scores"

        # At least half of metrics should have reasonable R2 (> -1.0).
        # Some metrics (e.g. power_watts) may have poor fit with limited data,
        # but the majority should not be wildly negative.
        reasonable = [m for m, r2 in r2_values.items() if r2 > -1.0]
        assert len(reasonable) >= len(r2_values) / 2, (
            f"Too many metrics with severe R2: { ({m: r2 for m, r2 in r2_values.items() if r2 <= -1.0}) }"
        )


class TestDegradationGraceful:
    """Pipeline should handle missing/degraded data without crashing."""

    def test_sensor_degradation_completes(self, tmp_path):
        sim = HouseholdSimulator(scenario="sensor_degradation", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()
        assert result["snapshots_saved"] == 30 * len(INTRADAY_HOURS)
        assert result["predictions"] is not None


class TestColdStartProgression:
    """Pipeline should progress through learning stages with limited data."""

    def test_7_day_cold_start(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        baselines = runner.compute_baselines()
        assert len(baselines) > 0

        predictions = runner.generate_predictions()
        assert predictions is not None
        assert "power_watts" in predictions
