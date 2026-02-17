"""Pattern recognition hub module.

Orchestrates sequence classification, pattern scale tagging, and anomaly
explanation. Subscribes to shadow_resolved events to maintain a sliding
window of recent feature snapshots, runs trajectory classification, and
caches results for ML engine consumption.

Tier 3+ only — self-gates on hardware tier.
"""

import logging
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from aria.engine.anomaly_explainer import AnomalyExplainer
from aria.engine.hardware import recommend_tier, scan_hardware
from aria.engine.pattern_scale import PatternScale
from aria.engine.sequence import SequenceClassifier
from aria.hub.core import Module

logger = logging.getLogger(__name__)

MIN_TIER = 3
DEFAULT_WINDOW_SIZE = 6


class PatternRecognitionModule(Module):
    """Hub module for trajectory classification and pattern analysis."""

    def __init__(self, hub):
        super().__init__("pattern_recognition", hub)
        self.active = False
        self.sequence_classifier = SequenceClassifier(window_size=DEFAULT_WINDOW_SIZE)
        self.anomaly_explainer = AnomalyExplainer()

        # Sliding window of recent feature snapshots per target
        self._feature_windows: dict[str, deque] = {}
        self._max_window = DEFAULT_WINDOW_SIZE * 2  # Keep extra for lag

        # Current state
        self.current_trajectory: str | None = None
        self._last_anomaly_explanations: list[dict] = []
        self._shadow_event_count = 0

        # Subscribe to events
        hub.subscribe("shadow_resolved", self._on_shadow_resolved)

    async def initialize(self):
        """Initialize — check hardware tier and activate if sufficient."""
        profile = scan_hardware()
        tier = recommend_tier(profile)

        if tier < MIN_TIER:
            logger.info(
                f"Pattern recognition disabled: tier {tier} < {MIN_TIER} "
                f"({profile.ram_gb:.1f}GB RAM, {profile.cpu_cores} cores)"
            )
            self.active = False
            return

        self.active = True
        logger.info(f"Pattern recognition active at tier {tier}")

    async def _on_shadow_resolved(self, event: dict[str, Any]):
        """Handle shadow_resolved events — update feature windows."""
        if not self.active:
            return

        target = event.get("target", "")
        features = event.get("features", {})
        timestamp = event.get("timestamp", "")

        if not features:
            return

        self._shadow_event_count += 1

        # Build numeric vector from features
        feature_names = sorted(features.keys())
        feature_vec = [float(features.get(k, 0)) for k in feature_names]

        # Maintain per-target sliding window
        if target not in self._feature_windows:
            self._feature_windows[target] = deque(maxlen=self._max_window)

        self._feature_windows[target].append(
            {
                "vector": feature_vec,
                "feature_names": feature_names,
                "timestamp": timestamp,
            }
        )

        # Classify trajectory when we have enough data
        window = self._feature_windows[target]
        if len(window) >= self.sequence_classifier.window_size:
            await self._classify_trajectory(target, window)

    async def _classify_trajectory(self, target: str, window: deque):
        """Run trajectory classification on the current window."""
        ws = self.sequence_classifier.window_size
        recent = list(window)[-ws:]
        window_array = np.array([entry["vector"] for entry in recent])

        if self.sequence_classifier.is_trained:
            trajectory = self.sequence_classifier.predict(window_array)
        else:
            # Fall back to heuristic when classifier not yet trained
            trajectory = SequenceClassifier.label_window_heuristic(window_array, target_col_idx=0)

        self.current_trajectory = trajectory

        # Cache for ML engine consumption
        await self.hub.set_cache(
            "pattern_trajectory",
            {
                "trajectory": trajectory,
                "target": target,
                "timestamp": datetime.now().isoformat(),
                "window_size": ws,
                "method": "dtw" if self.sequence_classifier.is_trained else "heuristic",
            },
        )

    def store_anomaly_explanations(self, explanations: list[dict]):
        """Store anomaly explanations from ML engine for API access."""
        self._last_anomaly_explanations = explanations

    def get_current_state(self) -> dict[str, Any]:
        """Return current pattern recognition state."""
        return {
            "trajectory": self.current_trajectory,
            "pattern_scales": {scale.value: scale.description for scale in PatternScale},
            "anomaly_explanations": self._last_anomaly_explanations,
            "shadow_events_processed": self._shadow_event_count,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return module statistics."""
        return {
            "active": self.active,
            "sequence_classifier": self.sequence_classifier.get_stats(),
            "window_count": {target: len(window) for target, window in self._feature_windows.items()},
            "current_trajectory": self.current_trajectory,
            "shadow_events_processed": self._shadow_event_count,
        }
