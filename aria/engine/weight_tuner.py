"""MAE-based ensemble weight auto-tuner."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class EnsembleWeightTuner:
    """Track rolling MAE per model and compute inverse-MAE weights."""

    def __init__(self, window_days: int = 7):
        self.window = timedelta(days=window_days)
        self._records: list[dict[str, Any]] = []

    def record(self, model: str, prediction: float, actual: float) -> None:
        self._records.append(
            {
                "model": model,
                "prediction": prediction,
                "actual": actual,
                "timestamp": datetime.now(),
            }
        )

    def _prune_old_records(self) -> None:
        cutoff = datetime.now() - self.window
        self._records = [r for r in self._records if r["timestamp"] > cutoff]

    def compute_weights(self) -> dict[str, float]:
        """Compute inverse-MAE weights. Higher weight = lower MAE = better model."""
        self._prune_old_records()

        if not self._records:
            return {}

        # Group by model, compute MAE
        errors: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            errors[r["model"]].append(abs(r["prediction"] - r["actual"]))

        maes: dict[str, float] = {}
        for model, errs in errors.items():
            maes[model] = sum(errs) / len(errs) if errs else float("inf")

        # Inverse-MAE weighting (add small epsilon to avoid division by zero)
        eps = 1e-6
        inv_maes = {m: 1.0 / (mae + eps) for m, mae in maes.items()}
        total = sum(inv_maes.values())

        if total == 0:
            return {}

        return {m: v / total for m, v in inv_maes.items()}

    def to_dict(self) -> dict[str, Any]:
        self._prune_old_records()
        errors: dict[str, list[float]] = defaultdict(list)
        for r in self._records:
            errors[r["model"]].append(abs(r["prediction"] - r["actual"]))

        return {
            "total_observations": len(self._records),
            "model_maes": {m: round(sum(e) / len(e), 4) if e else None for m, e in errors.items()},
            "computed_weights": self.compute_weights(),
            "window_days": self.window.days,
        }
