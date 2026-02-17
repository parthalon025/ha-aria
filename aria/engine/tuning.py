"""Optuna-based hyperparameter optimization for ML models."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 4,
    "subsample": 0.8,
    "num_leaves": 15,
}


def optimize_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    n_folds: int = 3,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Optuna optimization with expanding-window CV objective.

    Returns dict with best_params, best_score, trials_completed, fallback.
    Falls back to DEFAULT_PARAMS on any error.
    """
    if len(X) < 10:
        logger.warning("Too few samples for optimization, using defaults")
        return {
            "best_params": DEFAULT_PARAMS.copy(),
            "best_score": None,
            "trials_completed": 0,
            "fallback": True,
        }

    try:
        import optuna

        from aria.engine.evaluation import expanding_window_cv

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            }
            import lightgbm as lgb

            fold_scores = []
            for X_tr, y_tr, X_val, y_val in expanding_window_cv(X, y, n_folds):
                model = lgb.LGBMRegressor(
                    **params,
                    random_state=42,
                    verbosity=-1,
                    importance_type="gain",
                    min_child_samples=max(3, len(X_tr) // 20),
                )
                w = sample_weights[: len(X_tr)] if sample_weights is not None else None
                model.fit(X_tr, y_tr, sample_weight=w)
                pred = model.predict(X_val)
                mae = np.mean(np.abs(pred - y_val))
                fold_scores.append(mae)
            return np.mean(fold_scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "trials_completed": len(study.trials),
            "fallback": False,
        }

    except Exception as e:
        logger.warning(f"Optimization failed, using defaults: {e}")
        return {
            "best_params": DEFAULT_PARAMS.copy(),
            "best_score": None,
            "trials_completed": 0,
            "fallback": True,
        }
