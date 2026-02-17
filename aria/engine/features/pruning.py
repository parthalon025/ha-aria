"""Feature importance tracking and auto-pruning."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class FeaturePruner:
    def __init__(self, threshold: float = 0.01, required_cycles: int = 3):
        self.threshold = threshold
        self.required_cycles = required_cycles
        self._consecutive_low: dict[str, int] = {}  # feature -> consecutive low cycles
        self._pruned: set[str] = set()

    def identify_low_importance(self, importances: dict[str, float]) -> set[str]:
        """Return features with importance below threshold."""
        return {f for f, imp in importances.items() if imp < self.threshold}

    def record_cycle(self, low_features: set[str]) -> None:
        """Record which features were low-importance this training cycle."""
        # Increment consecutive count for features still low
        for feat in low_features:
            self._consecutive_low[feat] = self._consecutive_low.get(feat, 0) + 1
        # Reset count for features no longer low
        for feat in list(self._consecutive_low):
            if feat not in low_features:
                self._consecutive_low[feat] = 0
        # Auto-prune if threshold met
        newly_pruned = {
            f for f, count in self._consecutive_low.items() if count >= self.required_cycles and f not in self._pruned
        }
        if newly_pruned:
            self._pruned.update(newly_pruned)
            logger.info(f"Auto-pruned features: {newly_pruned}")

    def should_prune(self, feature: str) -> bool:
        return feature in self._pruned

    def on_drift_detected(self) -> None:
        """Reset all pruning on concept drift — features may become relevant again."""
        logger.info(f"Drift detected: resetting {len(self._pruned)} pruned features")
        self._pruned.clear()
        self._consecutive_low.clear()

    def get_active_features(self, all_features: list[str]) -> list[str]:
        return [f for f in all_features if f not in self._pruned]

    def to_dict(self) -> dict:
        return {
            "pruned": sorted(self._pruned),
            "consecutive_low": {f: c for f, c in self._consecutive_low.items() if c > 0},
        }
