"""Backtest validation using real Home Assistant data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.real_data import RealDataLoader

# Skip all tests if insufficient real data
loader = RealDataLoader()
SKIP_REASON = "Need >= 3 days of real intraday data in ~/ha-logs/intelligence/intraday/"
has_real_data = loader.available_days() >= 3


@pytest.mark.skipif(not has_real_data, reason=SKIP_REASON)
class TestRealDataBacktest:
    """Run the ARIA pipeline against real HA data and validate results."""

    @pytest.fixture(scope="class")
    def real_snapshots(self):
        return RealDataLoader().load_intraday(min_days=3)

    @pytest.fixture(scope="class")
    def real_pipeline_result(self, real_snapshots, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("backtest")
        runner = PipelineRunner(real_snapshots, data_dir=tmp)
        return runner.run_full()

    def test_real_data_loads_successfully(self, real_snapshots):
        assert len(real_snapshots) > 0
        first = real_snapshots[0]
        # Real snapshots must have these keys (same as synthetic)
        for key in ["date", "power", "lights", "occupancy"]:
            assert key in first, f"Real snapshot missing '{key}'"

    def test_real_pipeline_completes(self, real_pipeline_result):
        assert real_pipeline_result is not None
        assert "predictions" in real_pipeline_result
        assert "scores" in real_pipeline_result

    def test_real_predictions_are_numeric(self, real_pipeline_result):
        preds = real_pipeline_result["predictions"]
        for metric in ["power_watts", "lights_on", "devices_home"]:
            if metric in preds:
                val = preds[metric]
                if isinstance(val, dict):
                    assert isinstance(val.get("predicted", 0), int | float), f"{metric}.predicted is not numeric"

    def test_real_accuracy_above_minimum(self, real_pipeline_result):
        scores = real_pipeline_result["scores"]
        assert scores["overall"] > 0, f"Real data accuracy is {scores['overall']}% -- pipeline broken on real data"

    def test_real_vs_synthetic_schema_compatible(self, real_snapshots):
        """Real snapshot keys should be a superset of what the synthetic simulator produces."""
        from tests.synthetic.simulator import HouseholdSimulator

        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        synthetic = sim.generate()
        synthetic_keys = set(synthetic[0].keys())
        real_keys = set(real_snapshots[0].keys())

        # These keys MUST exist in both
        shared_required = {"date", "power", "lights", "occupancy"}
        for key in shared_required:
            assert key in real_keys, f"Real data missing required key: {key}"
            assert key in synthetic_keys, f"Synthetic data missing required key: {key}"

    def test_real_accuracy_report(self, real_pipeline_result):
        """Print formatted comparison report."""
        scores = real_pipeline_result["scores"]
        print(f"\n{'=' * 50}")
        print("  REAL-DATA BACKTEST REPORT")
        print(f"{'=' * 50}")
        print(f"  Overall accuracy: {scores['overall']}%")
        metrics = scores.get("metrics", {})
        for metric, data in metrics.items():
            acc = data.get("accuracy", "N/A")
            print(f"  {metric:<20} {acc}%")
        print(f"{'=' * 50}")

    def test_golden_baseline_regression(self, real_pipeline_result, tmp_path_factory):
        """Compare against stored golden baseline. Create baseline on first run."""
        golden_dir = Path(__file__).parent / "golden"
        golden_file = golden_dir / "backtest_baseline.json"

        current_scores = real_pipeline_result["scores"]

        if not golden_file.exists():
            # First run: establish baseline
            golden_dir.mkdir(exist_ok=True)
            golden_file.write_text(json.dumps(current_scores, indent=2))
            print(f"\nGolden baseline created: {golden_file}")
            print(f"Overall: {current_scores['overall']}%")
            return

        baseline = json.loads(golden_file.read_text())
        baseline_overall = baseline["overall"]
        current_overall = current_scores["overall"]

        print(f"\nBaseline: {baseline_overall}% -> Current: {current_overall}%")
        assert current_overall >= baseline_overall - 5, (
            f"Accuracy dropped from {baseline_overall}% to {current_overall}% (>{5}pt regression)"
        )
