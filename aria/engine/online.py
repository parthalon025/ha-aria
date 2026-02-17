"""River-based online learning models for real-time adaptation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MIN_SAMPLES_DEFAULT = 5


class OnlineModelWrapper:
    """Thin wrapper around River's ARFRegressor for ARIA integration.

    Provides learn_one/predict_one interface with a minimum-samples gate
    and reset capability for drift recovery.
    """

    def __init__(self, target: str, min_samples: int = MIN_SAMPLES_DEFAULT):
        self.target = target
        self.min_samples = min_samples
        self.samples_seen = 0
        self._model = self._create_model()

    def _create_model(self):
        from river.forest import ARFRegressor

        return ARFRegressor(
            n_models=10,
            seed=42,
        )

    def learn_one(self, features: dict[str, float], y: float) -> None:
        """Update model with one observation."""
        self._model.learn_one(features, y)
        self.samples_seen += 1

    def predict_one(self, features: dict[str, float]) -> float | None:
        """Predict for one observation. Returns None if insufficient data."""
        if self.samples_seen < self.min_samples:
            return None
        try:
            return float(self._model.predict_one(features))
        except Exception as e:
            logger.warning(f"Online prediction failed for {self.target}: {e}")
            return None

    def reset(self) -> None:
        """Reset model state (e.g., on drift detection)."""
        self._model = self._create_model()
        self.samples_seen = 0
        logger.info(f"Online model reset for {self.target}")

    def get_stats(self) -> dict[str, Any]:
        """Return model metadata and readiness info."""
        return {
            "target": self.target,
            "model_type": "ARFRegressor",
            "samples_seen": self.samples_seen,
            "min_samples": self.min_samples,
            "ready": self.samples_seen >= self.min_samples,
        }
