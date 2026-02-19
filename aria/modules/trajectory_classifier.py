"""Trajectory classifier hub module.

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

from aria.capabilities import Capability
from aria.engine.anomaly_explainer import AnomalyExplainer
from aria.engine.hardware import recommend_tier, scan_hardware
from aria.engine.pattern_scale import PatternScale
from aria.engine.sequence import SequenceClassifier
from aria.hub.core import Module

logger = logging.getLogger(__name__)

# Fallback constants — used only when config is unavailable
_DEFAULT_MIN_TIER = 3
_DEFAULT_WINDOW_SIZE = 6


class TrajectoryClassifier(Module):
    """Hub module for trajectory classification and pattern analysis."""

    CAPABILITIES = [
        Capability(
            id="trajectory_classifier",
            name="Trajectory Classifier",
            description=(
                "Trajectory classification, pattern scale tagging, and anomaly explanation from shadow engine events."
            ),
            module="trajectory_classifier",
            layer="hub",
            config_keys=["pattern.min_tier", "pattern.sequence_window_size"],
            test_paths=["tests/hub/test_trajectory_classifier.py"],
            runtime_deps=["numpy"],
            pipeline_stage="shadow",
            status="stable",
            depends_on=["shadow_predictions"],
        )
    ]

    def __init__(self, hub):
        super().__init__("trajectory_classifier", hub)
        self.active = False
        # Classifier and window size are set in initialize() after reading config
        self.sequence_classifier = SequenceClassifier(window_size=_DEFAULT_WINDOW_SIZE)
        self.anomaly_explainer = AnomalyExplainer()
        self.attention_explainer = None  # Tier 4 only

        # Sliding window of recent feature snapshots per target
        self._feature_windows: dict[str, deque] = {}
        self._max_window = _DEFAULT_WINDOW_SIZE * 2  # Keep extra for lag

        # Current state
        self.current_trajectory: str | None = None
        self._last_anomaly_explanations: list[dict] = []
        self._shadow_event_count = 0

    async def initialize(self):
        """Initialize — check hardware tier and activate if sufficient."""
        # Read config values with fallbacks to module-level defaults
        min_tier = int(
            await self.hub.cache.get_config_value("pattern.min_tier", _DEFAULT_MIN_TIER) or _DEFAULT_MIN_TIER
        )
        window_size = int(
            await self.hub.cache.get_config_value("pattern.sequence_window_size", _DEFAULT_WINDOW_SIZE)
            or _DEFAULT_WINDOW_SIZE
        )
        dtw_neighbors = int(await self.hub.cache.get_config_value("pattern.dtw_neighbors", 3) or 3)
        anomaly_top_n = int(await self.hub.cache.get_config_value("pattern.anomaly_top_n", 3) or 3)
        trajectory_change_threshold = float(
            await self.hub.cache.get_config_value("pattern.trajectory_change_threshold", 0.20) or 0.20
        )

        # Reinitialize classifier with config-driven values
        self.sequence_classifier = SequenceClassifier(window_size=window_size, n_neighbors=dtw_neighbors)
        self._anomaly_top_n = anomaly_top_n
        self._trajectory_change_threshold = trajectory_change_threshold
        self._max_window = window_size * 2

        profile = scan_hardware()
        tier = recommend_tier(profile)

        if tier < min_tier:
            logger.info(
                f"Pattern recognition disabled: tier {tier} < {min_tier} "
                f"({profile.ram_gb:.1f}GB RAM, {profile.cpu_cores} cores)"
            )
            self.active = False
            return

        self.active = True
        self.hub.subscribe("shadow_resolved", self._on_shadow_resolved)

        # Tier 4: attention-based anomaly explainer
        if tier >= 4:
            try:
                from aria.engine.attention_explainer import AttentionExplainer

                self.attention_explainer = AttentionExplainer(
                    n_features=window_size,
                    sequence_length=window_size,
                )
                logger.info("Attention explainer initialized (Tier 4)")
            except Exception as e:
                logger.warning(f"Attention explainer failed (non-fatal): {e}")

        logger.info(f"Pattern recognition active at tier {tier} (min_tier={min_tier}, window_size={window_size})")

    async def shutdown(self):
        """Unsubscribe from events on shutdown."""
        if self.active:
            self.hub.unsubscribe("shadow_resolved", self._on_shadow_resolved)

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

        trajectory = self.sequence_classifier.predict(window_array) if self.sequence_classifier.is_trained else None

        # If model prediction returned None (untrained/unavailable/failed),
        # fall back to heuristic labeling
        if trajectory is None:
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
            "attention_explainer": (
                self.attention_explainer.get_stats() if self.attention_explainer is not None else None
            ),
            "window_count": {target: len(window) for target, window in self._feature_windows.items()},
            "current_trajectory": self.current_trajectory,
            "shadow_events_processed": self._shadow_event_count,
        }
