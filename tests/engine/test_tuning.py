import numpy as np
import pytest

try:
    import optuna  # noqa: F401

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from aria.engine.tuning import optimize_hyperparams


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestHyperparamOptimization:
    def test_optimize_returns_params(self):
        np.random.seed(42)
        X = np.random.randn(60, 5)
        y = X[:, 0] * 2 + np.random.randn(60) * 0.1
        result = optimize_hyperparams(X, y, n_trials=5, n_folds=3)
        assert "best_params" in result
        assert "best_score" in result
        assert "n_estimators" in result["best_params"]

    def test_optimize_respects_trial_budget(self):
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        result = optimize_hyperparams(X, y, n_trials=3, n_folds=2)
        assert result["trials_completed"] <= 3

    def test_optimize_returns_fallback_on_error(self):
        """Empty data should return default params, not crash."""
        X = np.empty((0, 5))
        y = np.empty(0)
        result = optimize_hyperparams(X, y, n_trials=5, n_folds=3)
        assert result["fallback"] is True
