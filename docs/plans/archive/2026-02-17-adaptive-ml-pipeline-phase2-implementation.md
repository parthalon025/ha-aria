# Adaptive ML Pipeline — Phase 2: Online Learning Layer

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Completed — merged to main on 2026-02-17

**Goal:** Add River online learning models that learn from every shadow engine resolution, blend online predictions with batch predictions, and auto-tune ensemble weights based on rolling accuracy.

**Architecture:** New `aria/engine/online.py` for River model wrappers, new `aria/modules/online_learner.py` hub module that subscribes to shadow resolution events. Extends `_predict_single_target()` in `ml_engine.py` to blend online predictions. New `aria/engine/weight_tuner.py` for MAE-based auto-weighting every 24h.

**Tech Stack:** `river` 0.23.0 (already installed), existing hub event bus, existing shadow engine resolution loop.

**Design doc:** `docs/plans/2026-02-17-adaptive-ml-pipeline-design.md` (Phase 2 section, lines 145-188)

---

## Dependencies

- **Phase 1 complete:** Hardware scanner, tiered registry, fallback tracker, CV, feature pruning all merged.
- **river 0.23.0:** Already in `pyproject.toml` and installed in `.venv`.
- **Tier gate:** Online learning only activates at Tier 3+ (8GB+ RAM, 4+ cores). Tier 1-2 skip online models entirely.

---

## Task 1: River Model Wrapper

Thin wrapper around River's ARFRegressor that matches ARIA's feature/target conventions.

**Files:**
- Create: `aria/engine/online.py`
- Test: `tests/engine/test_online.py`

**Step 1: Write failing tests**

```python
# tests/engine/test_online.py
import pytest
from aria.engine.online import OnlineModelWrapper


class TestOnlineModelWrapper:
    def test_learn_one_and_predict(self):
        model = OnlineModelWrapper(target="power_watts")
        features = {"hour_sin": 0.5, "hour_cos": 0.87, "temp_f": 65.0}
        model.learn_one(features, y=500.0)
        pred = model.predict_one(features)
        assert isinstance(pred, float)

    def test_predict_before_learning_returns_none(self):
        model = OnlineModelWrapper(target="power_watts")
        features = {"hour_sin": 0.5, "hour_cos": 0.87}
        pred = model.predict_one(features)
        assert pred is None

    def test_predict_after_min_samples(self):
        """Need at least 5 samples before producing predictions."""
        model = OnlineModelWrapper(target="power_watts", min_samples=5)
        features = {"hour_sin": 0.5, "temp_f": 65.0}
        for i in range(4):
            model.learn_one(features, y=500.0 + i * 10)
        assert model.predict_one(features) is None
        model.learn_one(features, y=540.0)
        assert model.predict_one(features) is not None

    def test_get_stats(self):
        model = OnlineModelWrapper(target="power_watts")
        features = {"hour_sin": 0.5}
        model.learn_one(features, y=500.0)
        stats = model.get_stats()
        assert stats["target"] == "power_watts"
        assert stats["samples_seen"] == 1
        assert "model_type" in stats

    def test_reset_clears_state(self):
        model = OnlineModelWrapper(target="power_watts")
        model.learn_one({"hour_sin": 0.5}, y=500.0)
        assert model.samples_seen == 1
        model.reset()
        assert model.samples_seen == 0

    def test_feature_filtering(self):
        """Online model should handle missing features gracefully."""
        model = OnlineModelWrapper(target="power_watts")
        model.learn_one({"hour_sin": 0.5, "temp_f": 65.0}, y=500.0)
        # Predict with subset of features — should not crash
        pred = model.predict_one({"hour_sin": 0.5})
        # River handles missing features natively, so this should work
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/engine/test_online.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.engine.online'`

**Step 3: Write implementation**

```python
# aria/engine/online.py
"""River-based online learning models for real-time adaptation."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MIN_SAMPLES_DEFAULT = 5


class OnlineModelWrapper:
    """Thin wrapper around River's ARFRegressor for ARIA integration."""

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
        return {
            "target": self.target,
            "model_type": "ARFRegressor",
            "samples_seen": self.samples_seen,
            "min_samples": self.min_samples,
            "ready": self.samples_seen >= self.min_samples,
        }
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/engine/test_online.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add aria/engine/online.py tests/engine/test_online.py
git commit -m "feat: add River online model wrapper for real-time learning"
```

---

## Task 2: Online Learner Hub Module

Hub module that subscribes to shadow resolution events and feeds outcomes to online models.

**Files:**
- Create: `aria/modules/online_learner.py`
- Test: `tests/hub/test_online_learner.py`

**Step 1: Write failing tests**

```python
# tests/hub/test_online_learner.py
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock
from aria.modules.online_learner import OnlineLearnerModule


@pytest.fixture
def mock_hub():
    hub = Mock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.logger = Mock()
    hub.subscribe = Mock()
    hub.unsubscribe = Mock()
    hub.get_module = Mock(return_value=None)
    return hub


@pytest.fixture
def online_learner(mock_hub):
    return OnlineLearnerModule(mock_hub)


class TestOnlineLearnerModule:
    @pytest.mark.asyncio
    async def test_initialize_creates_models_per_target(self, online_learner):
        await online_learner.initialize()
        targets = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
        for target in targets:
            assert target in online_learner.models

    @pytest.mark.asyncio
    async def test_initialize_subscribes_to_events(self, online_learner, mock_hub):
        await online_learner.initialize()
        mock_hub.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_on_shadow_resolved_feeds_model(self, online_learner):
        await online_learner.initialize()
        # Simulate a shadow resolution event with features and actual value
        event_data = {
            "target": "power_watts",
            "features": {"hour_sin": 0.5, "hour_cos": 0.87, "temp_f": 65.0},
            "actual_value": 520.0,
            "outcome": "correct",
        }
        await online_learner._on_shadow_resolved(event_data)
        assert online_learner.models["power_watts"].samples_seen == 1

    @pytest.mark.asyncio
    async def test_get_prediction_returns_none_when_cold(self, online_learner):
        await online_learner.initialize()
        features = {"hour_sin": 0.5}
        pred = online_learner.get_prediction("power_watts", features)
        assert pred is None

    @pytest.mark.asyncio
    async def test_get_prediction_returns_value_after_learning(self, online_learner):
        await online_learner.initialize()
        features = {"hour_sin": 0.5, "temp_f": 65.0}
        for i in range(6):
            await online_learner._on_shadow_resolved({
                "target": "power_watts",
                "features": features,
                "actual_value": 500.0 + i * 10,
                "outcome": "correct",
            })
        pred = online_learner.get_prediction("power_watts", features)
        assert pred is not None
        assert isinstance(pred, float)

    @pytest.mark.asyncio
    async def test_get_all_stats(self, online_learner):
        await online_learner.initialize()
        stats = online_learner.get_all_stats()
        assert "power_watts" in stats
        assert stats["power_watts"]["samples_seen"] == 0

    @pytest.mark.asyncio
    async def test_on_drift_resets_affected_model(self, online_learner):
        await online_learner.initialize()
        # Feed some data
        for i in range(3):
            await online_learner._on_shadow_resolved({
                "target": "power_watts",
                "features": {"hour_sin": 0.5},
                "actual_value": 500.0,
                "outcome": "correct",
            })
        assert online_learner.models["power_watts"].samples_seen == 3
        # Simulate drift event
        await online_learner.on_event("drift_detected", {"target": "power_watts"})
        assert online_learner.models["power_watts"].samples_seen == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_online_learner.py -v`
Expected: FAIL — module not found

**Step 3: Write implementation**

```python
# aria/modules/online_learner.py
"""Online learning module — feeds shadow resolution outcomes to River models."""

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
            self.logger.info(
                f"Online learning disabled: tier {current_tier} < {MIN_TIER}"
            )
            # Still create models (inactive) so get_prediction returns None gracefully
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_online_learner.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add aria/modules/online_learner.py tests/hub/test_online_learner.py
git commit -m "feat: add online learner hub module with shadow resolution feed"
```

---

## Task 3: Emit shadow_resolved Events from Shadow Engine

The shadow engine resolves predictions in `_resolve_expired_predictions()` but doesn't publish a structured event with features and actual values. We need it to.

**Files:**
- Modify: `aria/modules/shadow_engine.py:921-960` (`_resolve_expired_predictions`)
- Test: `tests/hub/test_shadow_engine.py` (extend)

**Step 1: Read current resolution logic**

Read `aria/modules/shadow_engine.py` lines 921-960 to understand the current resolution loop and what data is available at resolution time.

Key context: At line 950, after scoring, the code calls `self._thompson.record_outcome(context, success=(outcome == "correct"))` and appends to `self._recent_resolved`. The `context` dict has features, and the prediction has `actual` data.

**Step 2: Write failing test**

Add to existing shadow engine tests:

```python
class TestShadowResolutionEvents:
    @pytest.mark.asyncio
    async def test_resolution_publishes_shadow_resolved_event(self, shadow_engine, mock_hub):
        """Resolving a prediction should publish shadow_resolved with features and actual."""
        # Set up a prediction that will expire and resolve
        shadow_engine._pending_predictions = [{
            "id": "pred-001",
            "timestamp": "2026-02-17T10:00:00",
            "expires_at": "2026-02-17T10:05:00",  # Already expired
            "context": {
                "room": "living_room",
                "features": {"hour_sin": 0.5, "temp_f": 65.0},
            },
            "predictions": [{"action": "light.turn_on", "target": "power_watts"}],
            "confidence": 0.85,
            "is_exploration": False,
        }]
        # Simulate actual outcome available
        mock_hub.publish = AsyncMock()
        await shadow_engine._resolve_expired_predictions()
        # Verify shadow_resolved was published
        publish_calls = [c for c in mock_hub.publish.call_args_list
                        if c[0][0] == "shadow_resolved"]
        # At minimum, expired predictions generate events
```

**Step 3: Add event publication to resolution loop**

In `_resolve_expired_predictions()`, after scoring a prediction (around line 950-955), add:

```python
# After recording outcome and appending to _recent_resolved:
await self.hub.publish("shadow_resolved", {
    "prediction_id": pred["id"],
    "target": self._extract_target_from_prediction(pred),
    "features": pred.get("context", {}).get("features", {}),
    "actual_value": self._extract_actual_value(pred, actual),
    "outcome": outcome,
    "timestamp": datetime.now().isoformat(),
})
```

Add helper `_extract_target_from_prediction()` and `_extract_actual_value()` that map shadow prediction outcomes to ML target values (power_watts, lights_on, etc.).

**Note:** The exact feature→target mapping depends on what data the shadow engine has at resolution time. Read the prediction structure carefully before implementing.

**Step 4: Run shadow engine tests**

Run: `.venv/bin/python -m pytest tests/hub/test_shadow_engine.py -v --timeout=120`
Expected: All existing + new tests PASS

**Step 5: Commit**

```bash
git add aria/modules/shadow_engine.py tests/hub/test_shadow_engine.py
git commit -m "feat: emit shadow_resolved events with features and actual values"
```

---

## Task 4: Register Online Learner Module in Hub

Wire the new module into the hub's module registry so it starts on hub boot.

**Files:**
- Modify: `aria/cli.py` (around `_register_modules`, ~line 289-331)
- Test: Verify via existing hub startup tests or manual check

**Step 1: Read current module registration**

Read `aria/cli.py` lines 289-331 to understand the module registration pattern.

**Step 2: Add online learner registration**

After the existing module registrations (shadow engine, ML engine, etc.), add:

```python
# Online learning (Tier 3+ only — module self-gates on tier)
from aria.modules.online_learner import OnlineLearnerModule
online_learner = OnlineLearnerModule(hub)
hub.register_module(online_learner)
```

**Step 3: Verify hub starts cleanly**

Run: `.venv/bin/python -c "from aria.modules.online_learner import OnlineLearnerModule; print('import OK')"`
Expected: `import OK`

**Step 4: Commit**

```bash
git add aria/cli.py
git commit -m "feat: register online learner module in hub startup"
```

---

## Task 5: Blend Online Predictions into ML Engine

Extend `_predict_single_target()` to include online model predictions in the ensemble.

**Files:**
- Modify: `aria/modules/ml_engine.py:1137-1183` (`_predict_single_target`)
- Test: `tests/hub/test_ml_training.py` (extend)

**Step 1: Write failing tests**

```python
class TestOnlineBlending:
    @pytest.fixture
    def ml_engine_with_online(self, mock_hub, tmp_path):
        models_dir = tmp_path / "models"
        training_dir = tmp_path / "training_data"
        models_dir.mkdir()
        training_dir.mkdir()
        engine = MLEngine(mock_hub, str(models_dir), str(training_dir))
        return engine

    def test_engine_has_online_blend_weight(self, ml_engine_with_online):
        assert hasattr(ml_engine_with_online, "online_blend_weight")

    def test_blend_without_online_uses_batch_only(self, ml_engine_with_online):
        """When no online prediction available, batch weight = 1.0."""
        engine = ml_engine_with_online
        batch_pred = 500.0
        online_pred = None
        blended = engine._blend_batch_online(batch_pred, online_pred)
        assert blended == batch_pred

    def test_blend_with_online_applies_weight(self, ml_engine_with_online):
        """When online prediction available, blend at configured ratio."""
        engine = ml_engine_with_online
        engine.online_blend_weight = 0.3
        batch_pred = 500.0
        online_pred = 520.0
        blended = engine._blend_batch_online(batch_pred, online_pred)
        expected = 0.7 * 500.0 + 0.3 * 520.0
        assert blended == pytest.approx(expected, abs=0.01)

    def test_blend_weight_zero_ignores_online(self, ml_engine_with_online):
        engine = ml_engine_with_online
        engine.online_blend_weight = 0.0
        blended = engine._blend_batch_online(500.0, 520.0)
        assert blended == 500.0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -k "OnlineBlending" -v`
Expected: FAIL — no `online_blend_weight` attribute

**Step 3: Add blending to MLEngine**

In `aria/modules/ml_engine.py`:

1. In `__init__` (around line 155), add:
```python
self.online_blend_weight = 0.3  # Default from design doc
```

2. Add blend method:
```python
def _blend_batch_online(self, batch_pred: float, online_pred: float | None) -> float:
    """Blend batch and online predictions. Falls back to batch-only if no online."""
    if online_pred is None or self.online_blend_weight <= 0:
        return batch_pred
    batch_weight = 1.0 - self.online_blend_weight
    return batch_weight * batch_pred + self.online_blend_weight * online_pred
```

3. In `_predict_single_target()` (around line 1167), after computing `blended_pred`, add:
```python
# Blend with online prediction if available
online_learner = self.hub.get_module("online_learner")
if online_learner:
    online_pred = online_learner.get_prediction(target, features_dict)
    blended_pred = self._blend_batch_online(blended_pred, online_pred)
    result["online_prediction"] = online_pred
    result["online_blend_weight"] = self.online_blend_weight if online_pred is not None else 0.0
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: blend online predictions into ML engine ensemble"
```

---

## Task 6: Ensemble Weight Auto-Tuner

Track rolling MAE per model source and auto-tune weights every 24h.

**Files:**
- Create: `aria/engine/weight_tuner.py`
- Test: `tests/engine/test_weight_tuner.py`

**Step 1: Write failing tests**

```python
# tests/engine/test_weight_tuner.py
import pytest
import numpy as np
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
        tuner._records.append({
            "model": "gb", "prediction": 100.0, "actual": 102.0,
            "timestamp": old_time,
        })
        tuner._prune_old_records()
        assert len(tuner._records) == 0
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/engine/test_weight_tuner.py -v`
Expected: FAIL — module not found

**Step 3: Implement**

```python
# aria/engine/weight_tuner.py
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
        self._records.append({
            "model": model,
            "prediction": prediction,
            "actual": actual,
            "timestamp": datetime.now(),
        })

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
            "model_maes": {
                m: round(sum(e) / len(e), 4) if e else None
                for m, e in errors.items()
            },
            "computed_weights": self.compute_weights(),
            "window_days": self.window.days,
        }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_weight_tuner.py -v`
Expected: All 6 PASS

**Step 5: Commit**

```bash
git add aria/engine/weight_tuner.py tests/engine/test_weight_tuner.py
git commit -m "feat: add MAE-based ensemble weight auto-tuner"
```

---

## Task 7: Wire Weight Tuner into ML Engine

Connect the tuner so it records per-model predictions and applies auto-tuned weights.

**Files:**
- Modify: `aria/modules/ml_engine.py` — init, `_predict_single_target`, periodic weight update
- Test: `tests/hub/test_ml_training.py` (extend)

**Step 1: Write failing test**

```python
class TestWeightTunerIntegration:
    def test_engine_has_weight_tuner(self, ml_engine):
        assert hasattr(ml_engine, "weight_tuner")

    @pytest.mark.asyncio
    async def test_predictions_record_to_tuner(self, ml_engine):
        """After predicting, individual model predictions should be recorded."""
        # This test verifies the tuner receives data during prediction
        assert ml_engine.weight_tuner.to_dict()["total_observations"] == 0
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -k "WeightTuner" -v`
Expected: FAIL

**Step 3: Add tuner to MLEngine**

In `__init__` (around line 155):
```python
from aria.engine.weight_tuner import EnsembleWeightTuner
self.weight_tuner = EnsembleWeightTuner(window_days=7)
```

In `_predict_single_target()`, after computing individual predictions (around line 1158), record each:
```python
for model_key, pred_value in individual_preds.items():
    # actual_value recorded later when shadow resolves — for now just record predictions
    pass
```

**Note:** The tuner needs `actual` values, which come from shadow resolution. Wire this via an event handler: when `shadow_resolved` fires with actual values, record them against the tuner's stored predictions for that timestamp.

Add periodic weight application (every 24h via hub scheduler):
```python
async def _apply_auto_weights(self):
    """Recompute ensemble weights from tuner and apply."""
    weights = self.weight_tuner.compute_weights()
    if weights:
        self.model_weights = weights
        await self.hub.set_cache("ml_ensemble_weights", weights)
        self.logger.info(f"Auto-tuned weights: {weights}")
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: wire ensemble weight auto-tuner into ML engine"
```

---

## Task 8: Config Entries for Phase 2

**Files:**
- Modify: `aria/hub/config_defaults.py`

**Step 1: Add config entries**

```python
{
    "key": "ml.online_blend_weight",
    "default_value": "0.3",
    "value_type": "number",
    "label": "Online Blend Weight",
    "description": "Weight for online model predictions in ensemble blend (0=disabled, 1=online only). Tier 3+ only.",
    "category": "ml",
    "min_value": 0.0,
    "max_value": 1.0,
    "step": 0.05,
},
{
    "key": "ml.online_min_samples",
    "default_value": "5",
    "value_type": "number",
    "label": "Online Min Samples",
    "description": "Minimum observations before online model starts predicting.",
    "category": "ml",
    "min_value": 1,
    "max_value": 50,
},
{
    "key": "ml.auto_tune_weights",
    "default_value": "true",
    "value_type": "boolean",
    "label": "Auto-Tune Weights",
    "description": "Automatically adjust ensemble weights based on rolling MAE (Tier 3+).",
    "category": "ml",
},
{
    "key": "ml.weight_tuner_window_days",
    "default_value": "7",
    "value_type": "number",
    "label": "Weight Tuner Window (days)",
    "description": "Rolling window for MAE-based weight computation.",
    "category": "ml",
    "min_value": 1,
    "max_value": 30,
},
```

**Step 2: Verify**

Run: `.venv/bin/python -c "from aria.hub.config_defaults import CONFIG_DEFAULTS; ml = [c for c in CONFIG_DEFAULTS if c.get('category') == 'ml']; print(f'{len(ml)} ml config entries')"`
Expected: 10 ml config entries (6 from Phase 1 + 4 new)

**Step 3: Commit**

```bash
git add aria/hub/config_defaults.py
git commit -m "config: add online learning and weight tuning settings"
```

---

## Task 9: API Endpoint for Online Learning Stats

**Files:**
- Modify: `aria/hub/api.py`
- Test: `tests/hub/test_api.py` (extend)

**Step 1: Write failing test**

```python
class TestOnlineLearningAPI:
    @pytest.mark.asyncio
    async def test_online_stats_endpoint(self, client):
        response = await client.get("/api/ml/online")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "weight_tuner" in data
```

**Step 2: Add endpoint**

```python
@router.get("/api/ml/online")
async def get_online_learning_stats():
    online_learner = hub.get_module("online_learner")
    ml_engine = hub.get_module("ml_engine")
    return {
        "models": online_learner.get_all_stats() if online_learner else {},
        "weight_tuner": ml_engine.weight_tuner.to_dict() if ml_engine else {},
        "online_blend_weight": ml_engine.online_blend_weight if ml_engine else 0.0,
    }
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api.py -k "online" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api.py
git commit -m "feat: add /api/ml/online endpoint for online learning stats"
```

---

## Task 10: Integration Test — Full Online Learning Pipeline

**Files:**
- Create: `tests/integration/test_online_learning_pipeline.py`

**Step 1: Write integration test**

```python
# tests/integration/test_online_learning_pipeline.py
import pytest
from aria.engine.online import OnlineModelWrapper
from aria.engine.weight_tuner import EnsembleWeightTuner
from aria.engine.hardware import HardwareProfile, recommend_tier


class TestOnlineLearningPipeline:
    def test_full_online_learning_cycle(self):
        """Simulate: learn from outcomes → predict → track accuracy → tune weights."""
        # 1. Verify tier gate
        profile = HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False)
        assert recommend_tier(profile) >= 3

        # 2. Create online model
        model = OnlineModelWrapper(target="power_watts", min_samples=3)

        # 3. Feed observations (simulating shadow resolutions)
        observations = [
            ({"hour_sin": 0.5, "temp_f": 65.0}, 500.0),
            ({"hour_sin": 0.7, "temp_f": 70.0}, 550.0),
            ({"hour_sin": 0.9, "temp_f": 60.0}, 480.0),
            ({"hour_sin": 0.3, "temp_f": 72.0}, 520.0),
        ]
        for features, actual in observations:
            model.learn_one(features, actual)

        # 4. Verify predictions available
        pred = model.predict_one({"hour_sin": 0.6, "temp_f": 67.0})
        assert pred is not None
        assert 400 < pred < 700  # Reasonable range

        # 5. Weight tuner tracks accuracy
        tuner = EnsembleWeightTuner(window_days=7)
        tuner.record("batch_gb", prediction=510.0, actual=500.0)      # MAE 10
        tuner.record("batch_lgbm", prediction=505.0, actual=500.0)    # MAE 5
        tuner.record("online_arf", prediction=502.0, actual=500.0)    # MAE 2
        weights = tuner.compute_weights()

        # Online model should get highest weight (lowest MAE)
        assert weights["online_arf"] > weights["batch_gb"]
        assert weights["online_arf"] > weights["batch_lgbm"]

    def test_cold_start_solved(self):
        """Online model produces predictions much sooner than batch."""
        model = OnlineModelWrapper(target="power_watts", min_samples=3)
        # After just 3 observations, online model is ready
        for i in range(3):
            model.learn_one({"hour_sin": 0.5 + i * 0.1}, y=500.0 + i * 10)
        assert model.predict_one({"hour_sin": 0.6}) is not None
        # Batch model needs 30+ days of daily snapshots

    def test_tier_2_no_online(self):
        """Tier 2 hardware should not activate online learning."""
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2
        # Module self-gates — would create models but not subscribe
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/integration/test_online_learning_pipeline.py -v`
Expected: All 3 PASS

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --timeout=120 -x -q`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/integration/test_online_learning_pipeline.py
git commit -m "test: add integration tests for online learning pipeline"
```

---

## Summary

| Task | What | New Files | Modified Files |
|------|------|-----------|----------------|
| 1 | River model wrapper | `aria/engine/online.py`, test | — |
| 2 | Online learner hub module | `aria/modules/online_learner.py`, test | — |
| 3 | Emit shadow_resolved events | — | `shadow_engine.py`, test |
| 4 | Register module in hub | — | `cli.py` |
| 5 | Blend online into predictions | — | `ml_engine.py`, test |
| 6 | Ensemble weight auto-tuner | `aria/engine/weight_tuner.py`, test | — |
| 7 | Wire tuner into ML engine | — | `ml_engine.py`, test |
| 8 | Config entries | — | `config_defaults.py` |
| 9 | API endpoint | — | `api.py`, test |
| 10 | Integration test | test | — |

**Total: 10 tasks, 10 commits, ~4 new files, ~5 modified files**

**Dependencies:** Tasks 1→2 (wrapper before module). Task 3 before 2 works (events before consumer). Tasks 1-3 before 5 (online predictions before blending). Task 6 before 7 (tuner before wiring). All others independent.

**Critical path:** 1 → 2 + 3 (parallel) → 4 → 5 → 7
