"""Online learning module -- feeds shadow resolution outcomes to River models."""

from __future__ import annotations

import logging
from typing import Any

from aria.engine.hardware import recommend_tier, scan_hardware
from aria.engine.online import OnlineModelWrapper

logger = logging.getLogger(__name__)

PREDICTION_TARGETS = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
MIN_TIER = 3  # Online learning only at Tier 3+


class OnlineLearnerModule:
    """Hub module that maintains per-target online models."""

    def __init__(self, hub):
        self.hub = hub
        self.models: dict[str, OnlineModelWrapper] = {}
        self._active = False
        self.logger = logger

    async def initialize(self) -> None:
        hw_profile = scan_hardware()
        current_tier = recommend_tier(hw_profile)

        if current_tier < MIN_TIER:
            self.logger.info(f"Online learning disabled: tier {current_tier} < {MIN_TIER}")
            for target in PREDICTION_TARGETS:
                self.models[target] = OnlineModelWrapper(target=target)
            return

        self._active = True
        for target in PREDICTION_TARGETS:
            self.models[target] = OnlineModelWrapper(target=target)

        self.hub.subscribe("shadow_resolved", self._on_shadow_resolved)
        self.logger.info(f"Online learning active at tier {current_tier}")

    async def shutdown(self) -> None:
        if self._active:
            self.hub.unsubscribe("shadow_resolved", self._on_shadow_resolved)

    async def on_event(self, event_type: str, data: dict[str, Any]) -> None:
        if event_type == "drift_detected":
            target = data.get("target")
            if target and target in self.models:
                self.models[target].reset()
                self.logger.info(f"Online model reset on drift: {target}")

    async def _on_shadow_resolved(self, data: dict[str, Any]) -> None:
        """Called when shadow engine resolves a prediction with actual outcome."""
        target = data.get("target")
        features = data.get("features")
        actual = data.get("actual_value")

        if not target or not features or actual is None:
            return
        if target not in self.models:
            return

        try:
            self.models[target].learn_one(features, float(actual))
        except Exception as e:
            self.logger.warning(f"Online learn_one failed for {target}: {e}")

    def get_prediction(self, target: str, features: dict[str, float]) -> float | None:
        """Get online prediction for a target. Returns None if model not ready."""
        model = self.models.get(target)
        if model is None:
            return None
        return model.predict_one(features)

    def get_all_stats(self) -> dict[str, dict]:
        return {target: model.get_stats() for target, model in self.models.items()}
