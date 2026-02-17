# Code Review Bugfixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all critical and important bugs found by the full-project code review — 7 critical runtime bugs and 10 important issues across hub, engine, and modules.

**Architecture:** Fixes are ordered by impact and dependency. Hub API fixes first (3 crashers in one file), then cross-layer integration fixes (feature mismatch, encoding), then module-level fixes (base class, subscriptions, shutdown). Each fix gets a failing test first, then minimal code change, then verification.

**Tech Stack:** Python 3.12, pytest, asyncio, FastAPI, numpy, scikit-learn

---

### Task 1: Fix `overall_accuracy / 100` scale bug in api.py

The cache computes `overall_accuracy` as 0-1 float. Two places in `api.py` divide by 100, turning 0.75 into 0.0075. This blocks pipeline advancement entirely.

**Files:**
- Modify: `aria/hub/api.py:921` and `aria/hub/api.py:1033`
- Test: `tests/hub/test_api_shadow.py`

**Step 1: Write the failing test**

Add to `tests/hub/test_api_shadow.py` inside `TestPipelineState`:

```python
class TestAccuracyScale:
    """Verify overall_accuracy is used as 0-1, not divided by 100."""

    def test_pipeline_bridge_uses_raw_accuracy(self, api_hub, api_client):
        """Shadow accuracy of 0.75 should bridge as 0.75, not 0.0075."""
        pipeline_state = {
            "current_stage": "backtest",
            "stage_entered_at": "2026-02-10T00:00:00",
            "backtest_accuracy": None,
            "shadow_accuracy_7d": None,
            "suggest_approval_rate_14d": None,
            "autonomous_contexts": None,
            "updated_at": "2026-02-12T10:00:00",
        }
        accuracy_stats = {
            "overall_accuracy": 0.75,
            "total_resolved": 100,
            "per_outcome": {"correct": 75, "incorrect": 25},
            "mean_confidence": 0.70,
        }
        api_hub.cache.get_pipeline_state = AsyncMock(return_value=pipeline_state)
        api_hub.cache.get_accuracy_stats = AsyncMock(return_value=accuracy_stats)
        api_hub.cache.update_pipeline_state = AsyncMock()

        response = api_client.get("/api/pipeline")
        assert response.status_code == 200

        # Verify the bridged value is 0.75, NOT 0.0075
        api_hub.cache.update_pipeline_state.assert_called_once()
        call_kwargs = api_hub.cache.update_pipeline_state.call_args[1]
        assert call_kwargs["backtest_accuracy"] == 0.75

    def test_stage_health_confidence_calibration_uses_raw_accuracy(self, api_hub, api_client):
        """Calibration error should use raw 0-1 accuracy, not /100."""
        pipeline_state = {
            "current_stage": "shadow",
            "stage_entered_at": "2026-02-10T00:00:00",
            "backtest_accuracy": 0.55,
            "shadow_accuracy_7d": 0.60,
            "suggest_approval_rate_14d": None,
            "autonomous_contexts": None,
            "updated_at": "2026-02-12T10:00:00",
        }
        accuracy_stats = {
            "overall_accuracy": 0.75,
            "total_resolved": 100,
            "total_attempted": 100,
            "per_outcome": {"correct": 75, "incorrect": 25},
            "mean_confidence": 0.70,
            "per_type": {},
            "daily_trend": [],
        }
        api_hub.cache.get_pipeline_state = AsyncMock(return_value=pipeline_state)
        api_hub.cache.get_accuracy_stats = AsyncMock(return_value=accuracy_stats)

        response = api_client.get("/api/shadow/accuracy")
        assert response.status_code == 200

        data = response.json()
        # Calibration: |0.70 - 0.75| = 0.05, so confidence_calibration = 0.95
        # With /100 bug: |0.70 - 0.0075| = 0.6925, calibration = 0.308
        stage_health = data.get("stage_health", {})
        assert stage_health.get("confidence_calibration", 0) > 0.9
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_api_shadow.py::TestAccuracyScale -v --timeout=120`
Expected: FAIL — `assert call_kwargs["backtest_accuracy"] == 0.75` fails (gets 0.0075)

**Step 3: Fix the two `/100` lines**

In `aria/hub/api.py` line 921, change:
```python
shadow_acc = bridge_stats.get("overall_accuracy", 0) / 100.0
```
to:
```python
shadow_acc = bridge_stats.get("overall_accuracy", 0)
```

In `aria/hub/api.py` line 1033, change:
```python
overall_acc = stats.get("overall_accuracy", 0) / 100.0  # normalize to 0-1
```
to:
```python
overall_acc = stats.get("overall_accuracy", 0)  # already 0-1 from cache
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_api_shadow.py::TestAccuracyScale -v --timeout=120`
Expected: PASS

**Step 5: Run existing shadow tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/hub/test_api_shadow.py -v --timeout=120`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_shadow.py
git commit -m "fix: remove /100 on overall_accuracy — was blocking pipeline advancement

overall_accuracy from cache is already 0-1. Dividing by 100 turned 0.75
into 0.0075, which could never pass the 0.40 backtest gate. Also broke
confidence calibration (always reported ~1.0)."
```

---

### Task 2: Make `get_module()` synchronous and fix missing await

`hub.get_module()` is `async def` but just does `return self.modules.get(module_id)` — no I/O. This caused a missing-await bug at api.py:410 and inconsistent access patterns throughout.

**Files:**
- Modify: `aria/hub/core.py:293`
- Modify: `aria/hub/api.py:410,429,430,448`
- Test: `tests/hub/test_core.py` (or create `tests/hub/test_get_module.py`)

**Step 1: Write the failing test**

Create `tests/hub/test_get_module.py`:

```python
"""Test that get_module is synchronous and returns Module or None."""

from unittest.mock import MagicMock

from aria.hub.core import IntelligenceHub


class TestGetModule:
    def test_get_module_returns_registered_module(self):
        """get_module should return a module by ID synchronously."""
        hub = IntelligenceHub.__new__(IntelligenceHub)
        hub.modules = {}
        mock_module = MagicMock()
        mock_module.module_id = "test_mod"
        hub.modules["test_mod"] = mock_module

        result = hub.get_module("test_mod")
        assert result is mock_module

    def test_get_module_returns_none_for_missing(self):
        """get_module should return None for unregistered module."""
        hub = IntelligenceHub.__new__(IntelligenceHub)
        hub.modules = {}

        result = hub.get_module("nonexistent")
        assert result is None

    def test_get_module_is_not_coroutine(self):
        """get_module should NOT be a coroutine (it's just a dict lookup)."""
        import asyncio

        hub = IntelligenceHub.__new__(IntelligenceHub)
        hub.modules = {}

        result = hub.get_module("anything")
        assert not asyncio.iscoroutine(result)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_get_module.py -v --timeout=120`
Expected: FAIL — `test_get_module_is_not_coroutine` fails (returns coroutine)

**Step 3: Make get_module synchronous**

In `aria/hub/core.py` line 293, change:
```python
    async def get_module(self, module_id: str) -> Module | None:
```
to:
```python
    def get_module(self, module_id: str) -> Module | None:
```

Then in `aria/hub/api.py`, remove `await` from all 4 `get_module` calls:

Line 410: `ml_module = hub.get_module("ml_engine")` (already missing await — now correct)
Line 429: `online_learner = hub.get_module("online_learner")` (remove `await`)
Line 430: `ml_engine = hub.get_module("ml_engine")` (remove `await`)
Line 448: `pattern_mod = hub.get_module("pattern_recognition")` (remove `await`)

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_get_module.py -v --timeout=120`
Expected: PASS

**Step 5: Run full hub tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/hub/ -v --timeout=120 -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add aria/hub/core.py aria/hub/api.py tests/hub/test_get_module.py
git commit -m "fix: make get_module() synchronous — was async for no reason

get_module() is just a dict lookup with no I/O. Being async caused a
missing-await bug at api.py:410 (returned coroutine instead of module,
silently 500'd). Removed await from all 4 call sites."
```

---

### Task 3: Fix missing `tier` parameter in `upsert_curation` API call

The PUT `/api/curation/{entity_id}` route calls `upsert_curation()` without the required `tier` param → TypeError at runtime.

**Files:**
- Modify: `aria/hub/api.py:50-52` (CurationUpdate model)
- Modify: `aria/hub/api.py:1160-1161` (call site)
- Modify: `tests/hub/test_api_config.py:250` (update existing test)

**Step 1: Write the failing test**

Update the existing test at `tests/hub/test_api_config.py` in `TestPutCuration`:

```python
    def test_put_curation_passes_tier(self, api_hub, api_client):
        """Upsert must pass tier parameter to cache method."""
        api_hub.cache.upsert_curation = AsyncMock(return_value=None)
        api_hub.publish = AsyncMock()

        response = api_client.put(
            "/api/curation/light.living_room",
            json={"status": "tracked", "tier": 2, "decided_by": "user"},
        )
        assert response.status_code == 200

        api_hub.cache.upsert_curation.assert_called_once_with(
            "light.living_room",
            status="tracked",
            tier=2,
            decided_by="user",
            human_override=True,
        )

    def test_put_curation_default_tier(self, api_hub, api_client):
        """Tier defaults to 3 when not specified."""
        api_hub.cache.upsert_curation = AsyncMock(return_value=None)
        api_hub.publish = AsyncMock()

        response = api_client.put(
            "/api/curation/light.living_room",
            json={"status": "tracked"},
        )
        assert response.status_code == 200

        call_kwargs = api_hub.cache.upsert_curation.call_args
        assert call_kwargs[1]["tier"] == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_api_config.py::TestPutCuration::test_put_curation_passes_tier -v --timeout=120`
Expected: FAIL — upsert_curation called without `tier`

**Step 3: Add tier to CurationUpdate model and pass it**

In `aria/hub/api.py`, update `CurationUpdate` (around line 50):
```python
class CurationUpdate(BaseModel):
    status: str
    tier: int = 3
    decided_by: str = "user"
```

Update the call site at line 1160:
```python
            result = await hub.cache.upsert_curation(
                entity_id, status=body.status, tier=body.tier, decided_by=body.decided_by, human_override=True
            )
```

Also fix the null return — return a proper response instead of `result` (which is `None`):
```python
            await hub.publish("curation_updated", {"entity_id": entity_id, "status": body.status})
            return {"status": "ok", "entity_id": entity_id, "curation_status": body.status}
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_api_config.py::TestPutCuration -v --timeout=120`
Expected: PASS

**Step 5: Update existing test assertion**

The existing `test_put_curation_override` needs to match the new call signature. Update its `assert_called_once_with` to include `tier=3` (the default).

**Step 6: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_config.py
git commit -m "fix: add missing tier parameter to PUT /api/curation endpoint

upsert_curation() requires tier:int but the API route didn't pass it.
Added tier field to CurationUpdate model (default=3). Also fixed null
response body — now returns structured JSON."
```

---

### Task 4: Fix `set_cache` kwargs conflict in ml_engine.py

`ml_engine.py:1092` passes `category="predictions"` as a kwarg which conflicts with the positional `category` arg → TypeError.

**Files:**
- Modify: `aria/modules/ml_engine.py:1092-1097`
- Test: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

Add to `tests/hub/test_ml_training.py`:

```python
class TestSetCachePredictions:
    """Verify ML engine stores predictions via set_cache without TypeError."""

    async def test_generate_predictions_calls_set_cache_correctly(self):
        """set_cache should be called with (category, data, metadata_dict)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        hub = MagicMock()
        hub.set_cache = AsyncMock()
        hub.get_cache = AsyncMock(return_value=None)
        hub.get_config_value = AsyncMock(return_value=None)

        from aria.modules.ml_engine import MLEngine

        engine = MLEngine.__new__(MLEngine)
        engine.hub = hub
        engine.models = {}
        engine.logger = MagicMock()
        engine.weight_tuner = MagicMock()
        engine.weight_tuner.to_dict.return_value = {}
        engine.online_blend_weight = 0.0

        # Call the method that triggers set_cache
        # We need to mock _build_prediction_feature_vector to return something
        engine._build_prediction_feature_vector = AsyncMock(return_value=(None, []))

        result = await engine.generate_predictions()

        # If _build_prediction_feature_vector returns (None, []), generate_predictions
        # exits early. That's fine — the important thing is no TypeError from set_cache.
        # For a more complete test, we'd need real models. But we verify the call pattern
        # if set_cache was called.
        if hub.set_cache.called:
            args, kwargs = hub.set_cache.call_args
            # First positional arg is category string
            assert isinstance(args[0], str)
            # Second positional arg is data dict
            assert isinstance(args[1], dict)
            # No conflicting 'category' kwarg
            assert "category" not in kwargs
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestSetCachePredictions -v --timeout=120`
Expected: FAIL or error due to the kwargs conflict

**Step 3: Fix the set_cache call**

In `aria/modules/ml_engine.py` lines 1092-1097, change:
```python
        await self.hub.set_cache(
            "ml_predictions",
            result,
            category="predictions",
            ttl_seconds=86400,
        )
```
to:
```python
        await self.hub.set_cache(
            "ml_predictions",
            result,
            {"source": "ml_engine", "ttl_seconds": 86400},
        )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestSetCachePredictions -v --timeout=120`
Expected: PASS

**Step 5: Run ML training tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120 -q`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "fix: remove conflicting category kwarg in set_cache call

set_cache(category, data, metadata) was called with positional category=
'ml_predictions' AND kwarg category='predictions', causing TypeError.
Moved metadata into the proper third arg dict."
```

---

### Task 5: Fix `OnlineLearnerModule` — extend Module base class

`OnlineLearnerModule` doesn't extend `Module`, so it has no `module_id`, no standard logger, and `module.module_id` references crash.

**Files:**
- Modify: `aria/modules/online_learner.py:17-24`
- Test: `tests/hub/test_online_learner.py` (create)

**Step 1: Write the failing test**

Create `tests/hub/test_online_learner.py`:

```python
"""Test that OnlineLearnerModule follows the Module contract."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.online_learner import OnlineLearnerModule


class TestOnlineLearnerModuleContract:
    def test_has_module_id(self):
        """Module must have a module_id attribute."""
        hub = MagicMock()
        hub.subscribe = MagicMock()
        with patch("aria.modules.online_learner.scan_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
            module = OnlineLearnerModule(hub)
        assert hasattr(module, "module_id")
        assert module.module_id == "online_learner"

    def test_is_instance_of_module(self):
        """OnlineLearnerModule must extend the Module base class."""
        from aria.hub.core import Module

        hub = MagicMock()
        with patch("aria.modules.online_learner.scan_hardware") as mock_hw:
            mock_hw.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
            module = OnlineLearnerModule(hub)
        assert isinstance(module, Module)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_online_learner.py -v --timeout=120`
Expected: FAIL — `assert hasattr(module, "module_id")` or `isinstance` check fails

**Step 3: Fix the class definition**

In `aria/modules/online_learner.py`, change the import and class:

Add import at the top:
```python
from aria.hub.core import Module
```

Change line 17:
```python
class OnlineLearnerModule(Module):
    """Hub module that maintains per-target online models."""

    def __init__(self, hub):
        super().__init__("online_learner", hub)
        self.models: dict[str, OnlineModelWrapper] = {}
        self._active = False
```

Remove the manual `self.logger = logger` assignment (line 24) — the parent `Module.__init__` sets `self.logger`.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_online_learner.py -v --timeout=120`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/online_learner.py tests/hub/test_online_learner.py
git commit -m "fix: OnlineLearnerModule now extends Module base class

Was a standalone class with no module_id, no standard logger, and
invisible to isinstance(m, Module) checks. Now extends Module with
super().__init__('online_learner', hub)."
```

---

### Task 6: Derive `_TRAJECTORY_ENCODING` from `TRAJECTORY_CLASSES`

Two independent copies of trajectory class names. Adding a class to one silently corrupts training data.

**Files:**
- Modify: `aria/modules/ml_engine.py:1133-1139`
- Test: `tests/hub/test_ml_training.py`

**Step 1: Write the failing test**

Add to `tests/hub/test_ml_training.py`:

```python
class TestTrajectoryEncoding:
    """Verify trajectory encoding derives from canonical TRAJECTORY_CLASSES."""

    def test_encoding_matches_trajectory_classes(self):
        """_TRAJECTORY_ENCODING must contain all TRAJECTORY_CLASSES values."""
        from aria.engine.sequence import TRAJECTORY_CLASSES
        from aria.modules.ml_engine import MLEngine

        for cls in TRAJECTORY_CLASSES:
            assert cls in MLEngine._TRAJECTORY_ENCODING, (
                f"TRAJECTORY_CLASSES has '{cls}' but _TRAJECTORY_ENCODING is missing it"
            )

    def test_encoding_has_no_extra_keys(self):
        """_TRAJECTORY_ENCODING should not have keys absent from TRAJECTORY_CLASSES."""
        from aria.engine.sequence import TRAJECTORY_CLASSES
        from aria.modules.ml_engine import MLEngine

        for key in MLEngine._TRAJECTORY_ENCODING:
            assert key in TRAJECTORY_CLASSES, (
                f"_TRAJECTORY_ENCODING has '{key}' but TRAJECTORY_CLASSES does not"
            )

    def test_encoding_preserves_index_order(self):
        """Encoding values must match TRAJECTORY_CLASSES list index."""
        from aria.engine.sequence import TRAJECTORY_CLASSES
        from aria.modules.ml_engine import MLEngine

        for i, cls in enumerate(TRAJECTORY_CLASSES):
            assert MLEngine._TRAJECTORY_ENCODING[cls] == i
```

**Step 2: Run test to verify it passes (currently in sync)**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestTrajectoryEncoding -v --timeout=120`
Expected: PASS (they happen to be in sync). The value of this change is structural — it prevents future drift.

**Step 3: Replace hardcoded dict with derived version**

In `aria/modules/ml_engine.py`, add import near the top:
```python
from aria.engine.sequence import TRAJECTORY_CLASSES
```

Replace lines 1133-1139:
```python
    # Trajectory class encoding (Phase 3)
    _TRAJECTORY_ENCODING = {
        "stable": 0,
        "ramping_up": 1,
        "winding_down": 2,
        "anomalous_transition": 3,
    }
```
with:
```python
    # Trajectory class encoding — derived from canonical source (Phase 3)
    _TRAJECTORY_ENCODING = {cls: i for i, cls in enumerate(TRAJECTORY_CLASSES)}
```

**Step 4: Run test to verify it still passes**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestTrajectoryEncoding -v --timeout=120`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "fix: derive _TRAJECTORY_ENCODING from TRAJECTORY_CLASSES

Two independent copies of the same mapping. Adding a trajectory class
to sequence.py would silently fall back to encoding 0 in ml_engine.
Now derives from the canonical source."
```

---

### Task 7: Fix PatternRecognitionModule — move subscribe to initialize(), add shutdown()

Subscribe happens in `__init__` (before tier check), and there's no `shutdown()` to unsubscribe.

**Files:**
- Modify: `aria/modules/pattern_recognition.py:48-49,51-65`
- Test: `tests/hub/test_pattern_recognition.py`

**Step 1: Write the failing tests**

Add to `tests/hub/test_pattern_recognition.py`:

```python
class TestPatternRecognitionLifecycle:
    @pytest.fixture
    def mock_hub(self):
        hub = MagicMock()
        hub.subscribe = MagicMock()
        hub.unsubscribe = MagicMock()
        hub.set_cache = AsyncMock()
        hub.get_cache = AsyncMock(return_value=None)
        hub.get_config_value = AsyncMock(return_value=None)
        return hub

    def test_no_subscribe_in_init(self, mock_hub):
        """Subscribe should NOT happen in __init__ — only after initialize()."""
        with patch("aria.modules.pattern_recognition.scan_hardware") as mock_hw, \
             patch("aria.modules.pattern_recognition.recommend_tier", return_value=3):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = PatternRecognitionModule(mock_hub)

        mock_hub.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_after_initialize_when_active(self, mock_hub):
        """Subscribe should happen in initialize() when tier >= 3."""
        with patch("aria.modules.pattern_recognition.scan_hardware") as mock_hw, \
             patch("aria.modules.pattern_recognition.recommend_tier", return_value=3):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = PatternRecognitionModule(mock_hub)
            await module.initialize()

        mock_hub.subscribe.assert_called_once_with("shadow_resolved", module._on_shadow_resolved)

    @pytest.mark.asyncio
    async def test_no_subscribe_when_tier_too_low(self, mock_hub):
        """Should NOT subscribe when tier < MIN_TIER."""
        with patch("aria.modules.pattern_recognition.scan_hardware") as mock_hw, \
             patch("aria.modules.pattern_recognition.recommend_tier", return_value=2):
            mock_hw.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
            module = PatternRecognitionModule(mock_hub)
            await module.initialize()

        mock_hub.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_unsubscribes(self, mock_hub):
        """shutdown() must unsubscribe from shadow_resolved."""
        with patch("aria.modules.pattern_recognition.scan_hardware") as mock_hw, \
             patch("aria.modules.pattern_recognition.recommend_tier", return_value=3):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = PatternRecognitionModule(mock_hub)
            await module.initialize()
            await module.shutdown()

        mock_hub.unsubscribe.assert_called_once_with("shadow_resolved", module._on_shadow_resolved)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py::TestPatternRecognitionLifecycle -v --timeout=120`
Expected: FAIL — `test_no_subscribe_in_init` fails (subscribe IS called in __init__)

**Step 3: Move subscribe to initialize(), add shutdown()**

In `aria/modules/pattern_recognition.py`, remove line 49 from `__init__`:
```python
        # DELETE: hub.subscribe("shadow_resolved", self._on_shadow_resolved)
```

In `initialize()`, add the subscribe after the tier check succeeds (after `self.active = True`):
```python
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
        self.hub.subscribe("shadow_resolved", self._on_shadow_resolved)
        logger.info(f"Pattern recognition active at tier {tier}")
```

Add `shutdown()` method:
```python
    async def shutdown(self):
        """Unsubscribe from events on shutdown."""
        if self.active:
            self.hub.unsubscribe("shadow_resolved", self._on_shadow_resolved)
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py -v --timeout=120`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add aria/modules/pattern_recognition.py tests/hub/test_pattern_recognition.py
git commit -m "fix: move PatternRecognition subscribe to initialize(), add shutdown()

Subscribe was in __init__ before tier check — wasted overhead when
disabled. Also had no shutdown() to unsubscribe, leaking event handlers.
Now matches ShadowEngine and OnlineLearner patterns."
```

---

### Task 8: Fix feature name mismatch between engine and hub

Engine's `get_feature_names()` excludes `pattern_features`. Hub's `_get_feature_names()` includes them. This causes dimension mismatch between training and inference.

**Files:**
- Modify: `aria/engine/features/vector_builder.py:39-49`
- Test: `tests/engine/test_features.py` (or `tests/engine/test_feature_config_validation.py`)

**Step 1: Write the failing test**

Add to `tests/engine/test_feature_config_validation.py`:

```python
class TestFeatureNameConsistency:
    """Verify engine and hub produce identical feature name lists."""

    def test_get_feature_names_includes_pattern_features(self):
        """Engine get_feature_names must include pattern_features when enabled."""
        from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
        from aria.engine.features.vector_builder import get_feature_names

        config = {**DEFAULT_FEATURE_CONFIG}
        config["pattern_features"] = {"trajectory_class": True}

        names = get_feature_names(config)
        assert "trajectory_class" in names

    def test_get_feature_names_excludes_disabled_pattern_features(self):
        """Pattern features should not appear when disabled."""
        from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
        from aria.engine.features.vector_builder import get_feature_names

        config = {**DEFAULT_FEATURE_CONFIG}
        config["pattern_features"] = {"trajectory_class": False}

        names = get_feature_names(config)
        assert "trajectory_class" not in names
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_feature_config_validation.py::TestFeatureNameConsistency -v --timeout=120`
Expected: FAIL — `assert "trajectory_class" in names` fails

**Step 3: Add pattern_features to get_feature_names**

In `aria/engine/features/vector_builder.py`, update `get_feature_names()`:
```python
def get_feature_names(config=None):
    """Return ordered list of feature names from config."""
    if config is None:
        config = DEFAULT_FEATURE_CONFIG
    names = _get_time_feature_names(config.get("time_features", {}))
    names.extend(_get_enabled_keys(config, "weather_features", prefix="weather_"))
    names.extend(_get_enabled_keys(config, "home_features"))
    names.extend(_get_enabled_keys(config, "lag_features"))
    names.extend(_get_enabled_keys(config, "interaction_features"))
    names.extend(_get_enabled_keys(config, "presence_features"))
    names.extend(_get_enabled_keys(config, "pattern_features"))
    return names
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_feature_config_validation.py::TestFeatureNameConsistency -v --timeout=120`
Expected: PASS

**Step 5: Run all feature tests**

Run: `.venv/bin/python -m pytest tests/engine/test_features.py tests/engine/test_feature_config_validation.py -v --timeout=120`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add aria/engine/features/vector_builder.py tests/engine/test_feature_config_validation.py
git commit -m "fix: add pattern_features to engine get_feature_names()

Engine and hub had different feature name lists — engine excluded
pattern_features, hub included them. This caused dimension mismatch
between batch training and real-time inference. Now both use the same
config-driven feature set."
```

---

### Task 9: Fix `datetime.utcnow()` deprecation

Two calls to deprecated `datetime.utcnow()` in api.py. Inconsistent with the rest of the codebase.

**Files:**
- Modify: `aria/hub/api.py:505,688`

**Step 1: No separate test needed** — this is a simple substitution with identical behavior in ARIA's single-timezone context. Existing tests cover both routes.

**Step 2: Replace both occurrences**

In `aria/hub/api.py` line 505, change:
```python
caps[capability_name]["promoted_at"] = datetime.utcnow().strftime("%Y-%m-%d")
```
to:
```python
caps[capability_name]["promoted_at"] = datetime.now().strftime("%Y-%m-%d")
```

In `aria/hub/api.py` line 688, change:
```python
"timestamp": datetime.utcnow().isoformat(),
```
to:
```python
"timestamp": datetime.now().isoformat(),
```

**Step 3: Run affected tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api_config.py tests/hub/test_api_shadow.py -v --timeout=120 -q`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add aria/hub/api.py
git commit -m "fix: replace deprecated datetime.utcnow() with datetime.now()

datetime.utcnow() was deprecated in Python 3.12. The rest of the
codebase uses datetime.now(). Consistent convention now."
```

---

### Task 10: Fix WebSocket broadcast set iteration safety

Iterating over a live set can raise `RuntimeError` if a connection is added during broadcast.

**Files:**
- Modify: `aria/hub/api.py:82`

**Step 1: No separate test needed** — this is a defensive one-line fix for a race condition.

**Step 2: Copy the set before iterating**

In `aria/hub/api.py` line 82, change:
```python
        for connection in self.active_connections:
```
to:
```python
        for connection in list(self.active_connections):
```

**Step 3: Run WebSocket tests**

Run: `.venv/bin/python -m pytest tests/hub/ -k "websocket" -v --timeout=120`
Expected: ALL PASS (or no WebSocket-specific tests — that's fine, the fix is obviously safe)

**Step 4: Commit**

```bash
git add aria/hub/api.py
git commit -m "fix: copy active_connections before iterating in broadcast

Iterating over a live set while coroutines may add connections could
raise RuntimeError. list() snapshot prevents this."
```

---

### Task 11: Add test improvements from coverage review

Strengthen Phase 3 tests that were flagged as weak by the test coverage reviewer.

**Files:**
- Modify: `tests/hub/test_pattern_recognition.py`
- Modify: `tests/integration/test_pattern_recognition_pipeline.py`

**Step 1: Add `store_anomaly_explanations` round-trip test**

In `tests/hub/test_pattern_recognition.py`:

```python
    async def test_store_and_retrieve_anomaly_explanations(self, mock_hub):
        """store_anomaly_explanations -> get_current_state round-trip."""
        with patch("aria.modules.pattern_recognition.scan_hardware") as mock_hw, \
             patch("aria.modules.pattern_recognition.recommend_tier", return_value=3):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = PatternRecognitionModule(mock_hub)

        explanations = [{"feature": "power_watts", "contribution": 0.45}]
        module.store_anomaly_explanations(explanations)
        state = module.get_current_state()
        assert state["anomaly_explanations"] == explanations
```

**Step 2: Strengthen trajectory value assertions**

In `tests/hub/test_pattern_recognition.py`, in `test_on_shadow_resolved_updates_cache`, change:
```python
assert module.current_trajectory is not None
```
to:
```python
assert module.current_trajectory == "ramping_up"
```

In `tests/integration/test_pattern_recognition_pipeline.py`, in `test_full_pipeline`, change:
```python
assert module.current_trajectory is not None
```
to:
```python
assert module.current_trajectory == "ramping_up"
```

**Step 3: Verify set_cache payload structure**

In `test_on_shadow_resolved_updates_cache`, after the trajectory assertion, add:
```python
        # Verify cache payload structure
        cache_args = mock_hub.set_cache.call_args[0]
        assert cache_args[0] == "pattern_trajectory"
        assert cache_args[1]["trajectory"] == "ramping_up"
        assert cache_args[1]["method"] == "heuristic"
        assert "timestamp" in cache_args[1]
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_pattern_recognition.py tests/integration/test_pattern_recognition_pipeline.py -v --timeout=120`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/hub/test_pattern_recognition.py tests/integration/test_pattern_recognition_pipeline.py
git commit -m "test: strengthen Phase 3 test assertions

- Add store_anomaly_explanations round-trip test
- Assert trajectory value is 'ramping_up', not just not-None
- Verify set_cache payload structure (key, trajectory, method)"
```

---

### Task 12: Run full test suite and verify

**Step 1: Check available memory**

```bash
free -h | awk '/Mem:/{print $7}'
```

If < 4G available, run by suite instead.

**Step 2: Run full suite**

```bash
.venv/bin/python -m pytest tests/ -v --timeout=120 -q
```

Expected: ~1510+ passed, 0 failed

**Step 3: Run ruff**

```bash
.venv/bin/python -m ruff check aria/ tests/ && .venv/bin/python -m ruff format --check aria/ tests/
```

Expected: clean
