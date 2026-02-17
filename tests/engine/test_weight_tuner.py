"""Tests for MAE-based ensemble weight auto-tuner."""

import pytest

from aria.engine.weight_tuner import EnsembleWeightTuner


class TestEnsembleWeightTuner:
    def test_record_and_compute_weights(self):
        tuner = EnsembleWeightTuner(window_days=7)
        # Model A: low MAE (good), Model B: high MAE (bad)
        for _ in range(10):
            tuner.record("gb", prediction=100.0, actual=102.0)  # MAE ~2
            tuner.record("rf", prediction=100.0, actual=110.0)  # MAE ~10
            tuner.record("lgbm", prediction=100.0, actual=103.0)  # MAE ~3
        weights = tuner.compute_weights()
        # GB should have highest weight (lowest MAE)
        assert weights["gb"] > weights["rf"]
        assert weights["lgbm"] > weights["rf"]
        assert pytest.approx(sum(weights.values()), abs=0.001) == 1.0

    def test_compute_weights_empty(self):
        tuner = EnsembleWeightTuner(window_days=7)
        weights = tuner.compute_weights()
        assert weights == {}

    def test_compute_weights_single_model(self):
        tuner = EnsembleWeightTuner(window_days=7)
        tuner.record("lgbm", prediction=100.0, actual=105.0)
        weights = tuner.compute_weights()
        assert weights == {"lgbm": 1.0}

    def test_record_with_online_source(self):
        """Online model predictions can also be tracked."""
        tuner = EnsembleWeightTuner(window_days=7)
        tuner.record("online_arf", prediction=100.0, actual=101.0)
        tuner.record("gb", prediction=100.0, actual=108.0)
        weights = tuner.compute_weights()
        assert weights["online_arf"] > weights["gb"]

    def test_to_dict(self):
        tuner = EnsembleWeightTuner(window_days=7)
        tuner.record("gb", prediction=100.0, actual=102.0)
        data = tuner.to_dict()
        assert "model_maes" in data
        assert "computed_weights" in data
        assert "total_observations" in data

    def test_window_pruning(self):
        """Records older than window_days should be pruned."""
        from datetime import datetime, timedelta

        tuner = EnsembleWeightTuner(window_days=7)
        # Manually add old records
        old_time = datetime.now() - timedelta(days=8)
        tuner._records.append(
            {
                "model": "gb",
                "prediction": 100.0,
                "actual": 102.0,
                "timestamp": old_time,
            }
        )
        tuner._prune_old_records()
        assert len(tuner._records) == 0
