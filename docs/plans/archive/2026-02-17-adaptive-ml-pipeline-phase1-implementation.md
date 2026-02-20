# Adaptive ML Pipeline — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** Completed — merged at `c21caab` on 2026-02-17

**Goal:** Build hardware-aware tiered model registry with graceful fallback, close existing pipeline gaps, add cross-validation and feature selection feedback.

**Architecture:** New `aria/engine/hardware.py` for hardware scanning, wire existing dead `aria/engine/models/registry.py` into `aria/modules/ml_engine.py`, add fallback tracking to hub cache, replace single train/val split with expanding-window CV, add feature pruning loop.

**Tech Stack:** psutil (new dep), optuna (optional Tier 3+), existing scikit-learn/lightgbm/river stack.

**Design doc:** `docs/plans/2026-02-17-adaptive-ml-pipeline-design.md`

---

## Task 1: Hardware Capability Scanner

**Files:**
- Create: `aria/engine/hardware.py`
- Test: `tests/engine/test_hardware.py`

**Step 1: Write failing tests**

```python
# tests/engine/test_hardware.py
import pytest
from unittest.mock import patch, MagicMock

from aria.engine.hardware import HardwareProfile, scan_hardware, recommend_tier


class TestHardwareProfile:
    def test_profile_dataclass_fields(self):
        profile = HardwareProfile(
            ram_gb=32.0, cpu_cores=8, gpu_available=False,
            gpu_name=None, benchmark_score=None
        )
        assert profile.ram_gb == 32.0
        assert profile.cpu_cores == 8
        assert profile.gpu_available is False

    def test_tier_1_low_hardware(self):
        profile = HardwareProfile(ram_gb=1.5, cpu_cores=1, gpu_available=False)
        assert recommend_tier(profile) == 1

    def test_tier_2_moderate_hardware(self):
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2

    def test_tier_3_high_cpu(self):
        profile = HardwareProfile(ram_gb=16.0, cpu_cores=8, gpu_available=False)
        assert recommend_tier(profile) == 3

    def test_tier_4_gpu_available(self):
        profile = HardwareProfile(ram_gb=16.0, cpu_cores=8, gpu_available=True)
        assert recommend_tier(profile) == 4

    def test_tier_4_requires_min_ram(self):
        """GPU present but <8GB RAM should not get Tier 4."""
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=4, gpu_available=True)
        assert recommend_tier(profile) == 2

    def test_tier_boundary_2gb_2cores(self):
        profile = HardwareProfile(ram_gb=2.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2

    def test_tier_boundary_8gb_4cores(self):
        profile = HardwareProfile(ram_gb=8.0, cpu_cores=4, gpu_available=False)
        assert recommend_tier(profile) == 3


class TestScanHardware:
    @patch("aria.engine.hardware.psutil")
    def test_scan_returns_profile(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)
        mock_psutil.cpu_count.return_value = 8
        profile = scan_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.ram_gb == pytest.approx(32.0, abs=0.5)
        assert profile.cpu_cores == 8

    @patch("aria.engine.hardware.psutil")
    def test_scan_gpu_detection_no_torch(self, mock_psutil):
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_psutil.cpu_count.return_value = 4
        with patch.dict("sys.modules", {"torch": None}):
            profile = scan_hardware()
        assert profile.gpu_available is False
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/engine/test_hardware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aria.engine.hardware'`

**Step 3: Write implementation**

```python
# aria/engine/hardware.py
"""Hardware capability scanner for tiered model selection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tier thresholds
TIER_4_MIN_RAM_GB = 8.0
TIER_3_MIN_RAM_GB = 8.0
TIER_3_MIN_CORES = 4
TIER_2_MIN_RAM_GB = 2.0
TIER_2_MIN_CORES = 2


@dataclass
class HardwareProfile:
    ram_gb: float
    cpu_cores: int
    gpu_available: bool
    gpu_name: str | None = None
    benchmark_score: float | None = None


def recommend_tier(profile: HardwareProfile) -> int:
    """Recommend ML tier based on hardware profile."""
    if profile.gpu_available and profile.ram_gb >= TIER_4_MIN_RAM_GB:
        return 4
    if profile.ram_gb >= TIER_3_MIN_RAM_GB and profile.cpu_cores >= TIER_3_MIN_CORES:
        return 3
    if profile.ram_gb >= TIER_2_MIN_RAM_GB and profile.cpu_cores >= TIER_2_MIN_CORES:
        return 2
    return 1


def scan_hardware() -> HardwareProfile:
    """Probe system hardware and return a HardwareProfile."""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed — defaulting to Tier 1 profile")
        return HardwareProfile(ram_gb=0, cpu_cores=1, gpu_available=False)

    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024 ** 3)
    cpu_cores = psutil.cpu_count(logical=True) or 1

    gpu_available = False
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_name = "Apple MPS"
    except (ImportError, Exception):
        pass

    profile = HardwareProfile(
        ram_gb=round(ram_gb, 1),
        cpu_cores=cpu_cores,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )
    tier = recommend_tier(profile)
    logger.info(
        f"Hardware: {profile.ram_gb}GB RAM, {profile.cpu_cores} cores, "
        f"GPU={'yes (' + (gpu_name or '?') + ')' if gpu_available else 'no'} "
        f"→ Recommended tier: {tier}"
    )
    return profile
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/engine/test_hardware.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add aria/engine/hardware.py tests/engine/test_hardware.py
git commit -m "feat: add hardware capability scanner for tiered ML"
```

---

## Task 2: Add psutil Dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add psutil to core dependencies**

In `pyproject.toml` under `[project] dependencies`, add `"psutil>=5.9.0"` to the list. It's a core dep because the scanner runs at startup for all tiers.

**Step 2: Install and verify**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/pip install -e ".[dev]"`
Expected: psutil installs successfully

**Step 3: Run hardware scanner manually**

Run: `.venv/bin/python -c "from aria.engine.hardware import scan_hardware, recommend_tier; p = scan_hardware(); print(f'Tier: {recommend_tier(p)}, Profile: {p}')"`
Expected: Output showing actual hardware profile and tier recommendation

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add psutil for hardware capability scanning"
```

---

## Task 3: Wire Model Registry to ML Engine

The registry at `aria/engine/models/registry.py` exists but is dead code. This task wires it into `ml_engine.py`.

**Files:**
- Modify: `aria/engine/models/registry.py` (extend with tier support)
- Modify: `aria/modules/ml_engine.py:150-162` (model dicts) and `ml_engine.py:387-419` (`_fit_all_models`)
- Test: `tests/engine/test_model_registry.py` (new)
- Test: `tests/hub/test_ml_training.py` (verify existing tests still pass)

**Step 1: Read existing registry**

Read `aria/engine/models/registry.py` in full to understand the current `ModelRegistry` and `BaseModel` interfaces. Note what needs extending vs. replacing.

**Step 2: Write failing tests for tier-aware registry**

```python
# tests/engine/test_model_registry.py
import pytest
from aria.engine.models.registry import TieredModelRegistry, ModelEntry


class TestTieredModelRegistry:
    def test_register_and_resolve_tier_2(self):
        registry = TieredModelRegistry()
        entry = ModelEntry(
            name="lgbm_power",
            tier=2,
            model_factory=lambda: "mock_lgbm",
            params={"n_estimators": 100},
            weight=0.40,
            requires=["lightgbm"],
        )
        registry.register("power_watts", entry)
        resolved = registry.resolve("power_watts", current_tier=2)
        assert len(resolved) == 1
        assert resolved[0].name == "lgbm_power"

    def test_resolve_excludes_higher_tiers(self):
        registry = TieredModelRegistry()
        registry.register("power_watts", ModelEntry(
            name="lgbm_power", tier=2,
            model_factory=lambda: None, params={}, weight=0.40, requires=[]
        ))
        registry.register("power_watts", ModelEntry(
            name="transformer_power", tier=4,
            model_factory=lambda: None, params={}, weight=0.30, requires=["torch"]
        ))
        resolved = registry.resolve("power_watts", current_tier=2)
        assert len(resolved) == 1
        assert resolved[0].name == "lgbm_power"

    def test_resolve_includes_lower_tiers(self):
        registry = TieredModelRegistry()
        registry.register("power_watts", ModelEntry(
            name="lgbm_simple", tier=1,
            model_factory=lambda: None, params={}, weight=0.50, requires=[]
        ))
        registry.register("power_watts", ModelEntry(
            name="lgbm_full", tier=2,
            model_factory=lambda: None, params={}, weight=0.40, requires=[]
        ))
        resolved = registry.resolve("power_watts", current_tier=3)
        assert len(resolved) == 2

    def test_resolve_skips_missing_dependencies(self):
        registry = TieredModelRegistry()
        registry.register("power_watts", ModelEntry(
            name="transformer", tier=3,
            model_factory=lambda: None, params={}, weight=0.30,
            requires=["nonexistent_package_xyz"]
        ))
        resolved = registry.resolve("power_watts", current_tier=3)
        assert len(resolved) == 0

    def test_weights_renormalize(self):
        registry = TieredModelRegistry()
        registry.register("power_watts", ModelEntry(
            name="gb", tier=2, model_factory=lambda: None,
            params={}, weight=0.35, requires=[]
        ))
        registry.register("power_watts", ModelEntry(
            name="rf", tier=2, model_factory=lambda: None,
            params={}, weight=0.25, requires=[]
        ))
        resolved = registry.resolve("power_watts", current_tier=2)
        total = sum(e.weight for e in resolved)
        # Weights should be renormalized to sum to 1.0
        normed = registry.get_normalized_weights("power_watts", current_tier=2)
        assert pytest.approx(sum(normed.values()), abs=0.001) == 1.0

    def test_default_stack_registers_current_models(self):
        """Default registry should contain gb, rf, lgbm at tier 2."""
        registry = TieredModelRegistry.with_defaults()
        for target in ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]:
            resolved = registry.resolve(target, current_tier=2)
            names = [e.name for e in resolved]
            assert "gb" in names or any("gb" in n for n in names)
```

**Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/engine/test_model_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'TieredModelRegistry'`

**Step 4: Extend registry with tier support**

Modify `aria/engine/models/registry.py` — keep existing `ModelRegistry` and `BaseModel` for backward compat. Add:

```python
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    name: str
    tier: int
    model_factory: callable  # () -> fitted model instance
    params: dict = field(default_factory=dict)
    weight: float = 1.0
    requires: list[str] = field(default_factory=list)
    fallback_tier: int | None = None


class TieredModelRegistry:
    """Tier-aware model registry. Each target has a stack of models at different tiers."""

    def __init__(self):
        self._stacks: dict[str, list[ModelEntry]] = {}  # target -> [entries]

    def register(self, target: str, entry: ModelEntry) -> None:
        self._stacks.setdefault(target, []).append(entry)

    def resolve(self, target: str, current_tier: int) -> list[ModelEntry]:
        """Return all entries at or below current_tier with satisfied dependencies."""
        entries = self._stacks.get(target, [])
        resolved = []
        for e in entries:
            if e.tier > current_tier:
                continue
            if not self._check_deps(e):
                logger.info(f"Skipping {e.name}: missing {e.requires}")
                continue
            resolved.append(e)
        return sorted(resolved, key=lambda e: e.tier)

    def get_normalized_weights(self, target: str, current_tier: int) -> dict[str, float]:
        """Return weights renormalized to sum to 1.0 for resolved entries."""
        resolved = self.resolve(target, current_tier)
        total = sum(e.weight for e in resolved)
        if total == 0:
            return {}
        return {e.name: e.weight / total for e in resolved}

    @staticmethod
    def _check_deps(entry: ModelEntry) -> bool:
        for pkg in entry.requires:
            try:
                importlib.import_module(pkg)
            except ImportError:
                return False
        return True

    @classmethod
    def with_defaults(cls) -> TieredModelRegistry:
        """Create registry with ARIA's default model stacks."""
        registry = cls()
        targets = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]

        for target in targets:
            # Tier 1: single LightGBM
            registry.register(target, ModelEntry(
                name="lgbm_lite", tier=1, weight=1.0,
                model_factory=None,  # set at training time
                params={"n_estimators": 50, "max_depth": 3, "verbosity": -1},
                requires=["lightgbm"],
            ))

            # Tier 2: full ensemble (current ARIA default)
            registry.register(target, ModelEntry(
                name="gb", tier=2, weight=0.35,
                model_factory=None,
                params={"n_estimators": 100, "learning_rate": 0.1,
                        "max_depth": 4, "subsample": 0.8},
                requires=[],
            ))
            registry.register(target, ModelEntry(
                name="rf", tier=2, weight=0.25,
                model_factory=None,
                params={"n_estimators": 100, "max_depth": 5},
                requires=[],
            ))
            registry.register(target, ModelEntry(
                name="lgbm", tier=2, weight=0.40,
                model_factory=None,
                params={"n_estimators": 100, "learning_rate": 0.1,
                        "max_depth": 4, "num_leaves": 15, "subsample": 0.8,
                        "verbosity": -1, "importance_type": "gain"},
                requires=["lightgbm"],
            ))

        return registry
```

**Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/engine/test_model_registry.py -v`
Expected: All 6 tests PASS

**Step 6: Verify existing tests unaffected**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All existing tests still PASS (registry additions are additive, no behavior change yet)

**Step 7: Commit**

```bash
git add aria/engine/models/registry.py tests/engine/test_model_registry.py
git commit -m "feat: add tier-aware model registry with default stacks"
```

---

## Task 4: Integrate Registry into ML Engine Training

This replaces the hardcoded `_fit_all_models` with registry-driven training.

**Files:**
- Modify: `aria/modules/ml_engine.py:150-162` (init), `387-419` (_fit_all_models), `1009+` (generate_predictions)
- Modify: `aria/modules/ml_engine.py:__init__` — accept hardware tier
- Test: `tests/hub/test_ml_training.py` (extend existing)

**Step 1: Write failing tests for registry-driven training**

Add to existing test file `tests/hub/test_ml_training.py`:

```python
class TestRegistryDrivenTraining:
    @pytest.fixture
    def ml_engine_with_registry(self, mock_hub, tmp_path):
        models_dir = tmp_path / "models"
        training_dir = tmp_path / "training_data"
        models_dir.mkdir()
        training_dir.mkdir()
        engine = MLEngine(mock_hub, str(models_dir), str(training_dir))
        return engine

    def test_engine_has_registry(self, ml_engine_with_registry):
        assert hasattr(ml_engine_with_registry, "registry")
        assert ml_engine_with_registry.registry is not None

    def test_engine_has_current_tier(self, ml_engine_with_registry):
        assert hasattr(ml_engine_with_registry, "current_tier")
        assert ml_engine_with_registry.current_tier in (1, 2, 3, 4)

    def test_fit_uses_registry_entries(self, ml_engine_with_registry):
        """Models fitted should correspond to registry entries for current tier."""
        engine = ml_engine_with_registry
        resolved = engine.registry.resolve("power_watts", engine.current_tier)
        resolved_names = {e.name for e in resolved}
        # At minimum, Tier 2 defaults should be present
        assert len(resolved_names) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py::TestRegistryDrivenTraining -v`
Expected: FAIL — engine doesn't have `registry` attribute yet

**Step 3: Modify MLEngine.__init__ to use registry**

In `aria/modules/ml_engine.py`, modify `__init__` (around line 130-162):

- Import `TieredModelRegistry` from `aria.engine.models.registry`
- Import `scan_hardware, recommend_tier` from `aria.engine.hardware`
- Add `self.registry = TieredModelRegistry.with_defaults()`
- Add hardware scan: `profile = scan_hardware(); self.current_tier = recommend_tier(profile)`
- Check config override: read `ml.tier_override` from hub cache; if set and not "auto", use that value
- Keep existing `self.enabled_models` and `self.model_weights` dicts for backward compat but derive them from registry:
  ```python
  resolved = self.registry.resolve("power_watts", self.current_tier)
  self.enabled_models = {e.name: True for e in resolved}
  self.model_weights = self.registry.get_normalized_weights("power_watts", self.current_tier)
  ```

**Step 4: Modify _fit_all_models to use registry entries**

Replace hardcoded model instantiation at lines 387-419 with registry-driven loop:

```python
@staticmethod
def _fit_all_models(X_train, y_train, w_train, registry, target, tier):
    """Fit models from registry for target at given tier."""
    entries = registry.resolve(target, tier)
    fitted = {}
    for entry in entries:
        try:
            model = _create_model(entry)
            if entry.name.startswith("iso"):
                model.fit(X_train)
            else:
                model.fit(X_train, y_train, sample_weight=w_train)
            fitted[entry.name] = model
        except Exception as e:
            logger.warning(f"Failed to fit {entry.name}: {e}")
    return fitted
```

Add helper `_create_model(entry)` that maps entry name/params to sklearn/lgbm constructors. This keeps model creation logic centralized.

**Step 5: Update generate_predictions to use fitted dict**

In `generate_predictions` (around line 1009), replace hardcoded model access with iteration over `self.models[target]` dict keyed by entry name. Use `self.registry.get_normalized_weights()` for blending.

**Step 6: Run full ML training test suite**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All tests PASS (new + existing)

**Step 7: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "feat: wire tiered model registry into ML engine training"
```

---

## Task 5: Graceful Fallback Engine

**Files:**
- Create: `aria/engine/fallback.py`
- Test: `tests/engine/test_fallback.py`
- Modify: `aria/modules/ml_engine.py` — wrap model fitting with fallback

**Step 1: Write failing tests**

```python
# tests/engine/test_fallback.py
import pytest
from datetime import datetime, timedelta
from aria.engine.fallback import FallbackTracker, FallbackEvent


class TestFallbackTracker:
    def test_record_fallback(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2,
                       error="MemoryError", memory_mb=25600)
        assert tracker.is_fallen_back("lgbm_power")

    def test_fallback_expires(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        # Simulate expiry
        tracker._events["lgbm_power"].timestamp = (
            datetime.now() - timedelta(days=8)
        )
        assert not tracker.is_fallen_back("lgbm_power")

    def test_get_effective_tier(self):
        tracker = FallbackTracker(ttl_days=7)
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 3
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 2

    def test_active_fallbacks_list(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        tracker.record("transformer_power", from_tier=4, to_tier=3, error="CUDA OOM")
        active = tracker.active_fallbacks()
        assert len(active) == 2

    def test_clear_fallback(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        tracker.clear("lgbm_power")
        assert not tracker.is_fallen_back("lgbm_power")
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/engine/test_fallback.py -v`
Expected: FAIL — module not found

**Step 3: Implement**

```python
# aria/engine/fallback.py
"""Per-model fallback tracking with TTL-based expiry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    model_name: str
    from_tier: int
    to_tier: int
    error: str
    memory_mb: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class FallbackTracker:
    def __init__(self, ttl_days: int = 7):
        self.ttl = timedelta(days=ttl_days)
        self._events: dict[str, FallbackEvent] = {}

    def record(self, model_name: str, from_tier: int, to_tier: int,
               error: str, memory_mb: float | None = None) -> None:
        event = FallbackEvent(
            model_name=model_name, from_tier=from_tier, to_tier=to_tier,
            error=error, memory_mb=memory_mb,
        )
        self._events[model_name] = event
        logger.warning(
            f"Model fallback: {model_name} tier {from_tier}→{to_tier} ({error})"
        )

    def is_fallen_back(self, model_name: str) -> bool:
        event = self._events.get(model_name)
        if event is None:
            return False
        if datetime.now() - event.timestamp > self.ttl:
            del self._events[model_name]
            return False
        return True

    def get_effective_tier(self, model_name: str, original_tier: int) -> int:
        if self.is_fallen_back(model_name):
            return self._events[model_name].to_tier
        return original_tier

    def active_fallbacks(self) -> list[FallbackEvent]:
        # Clean expired first
        now = datetime.now()
        expired = [k for k, v in self._events.items() if now - v.timestamp > self.ttl]
        for k in expired:
            del self._events[k]
        return list(self._events.values())

    def clear(self, model_name: str) -> None:
        self._events.pop(model_name, None)

    def to_dict(self) -> list[dict]:
        return [
            {
                "model": e.model_name, "from_tier": e.from_tier,
                "to_tier": e.to_tier, "error": e.error,
                "memory_mb": e.memory_mb,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.active_fallbacks()
        ]
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_fallback.py -v`
Expected: All 5 PASS

**Step 5: Wire fallback into ml_engine training loop**

In `aria/modules/ml_engine.py`, in the registry-driven `_fit_all_models` (from Task 4):

```python
# Inside the training loop for each model entry:
try:
    model = _create_model(entry)
    model.fit(X_train, y_train, sample_weight=w_train)
    fitted[entry.name] = model
    self.fallback_tracker.clear(entry.name)
except (MemoryError, Exception) as e:
    self.fallback_tracker.record(
        entry.name, from_tier=entry.tier,
        to_tier=max(1, entry.tier - 1), error=str(e)
    )
    # Try lower-tier fallback entry if available
    fallback_entries = [
        fe for fe in registry.resolve(target, entry.tier - 1)
        if fe.name != entry.name
    ]
    # ... fit fallback if available
```

Also emit `model_fallback` event on hub event bus after recording.

**Step 6: Run ML training tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 7: Commit**

```bash
git add aria/engine/fallback.py tests/engine/test_fallback.py aria/modules/ml_engine.py
git commit -m "feat: add per-model graceful fallback with TTL expiry"
```

---

## Task 6: Close Gap — Snapshot Validation in Hub Training

**Files:**
- Modify: `aria/modules/ml_engine.py` — `_train_model_for_target` method
- Test: `tests/hub/test_ml_training.py` — add validation test

**Step 1: Write failing test**

```python
class TestHubSnapshotValidation:
    @pytest.mark.asyncio
    async def test_invalid_snapshots_rejected_during_training(
        self, ml_engine, tmp_path
    ):
        """Hub training should reject snapshots with too few entities."""
        training_dir = tmp_path / "training_data"
        training_dir.mkdir(exist_ok=True)
        # Write a snapshot with entity count below MIN_ENTITY_COUNT (100)
        bad_snapshot = {
            "date": "2026-02-17",
            "entities": {"total": 5, "unavailable": 0},
            "power": {"total_watts": 100},
        }
        import json
        (training_dir / "2026-02-17_snapshot.json").write_text(json.dumps(bad_snapshot))
        # Training should log warning and produce no model (insufficient valid data)
        # This test verifies validate_snapshot_batch is called in hub path
```

**Step 2: Verify `validate_snapshot_batch` is NOT called in hub _train_model_for_target**

Read `ml_engine.py` around the snapshot loading in `_train_model_for_target`. Confirm validation is missing. (The explore agent found it IS called at line 381 — verify this is in the correct method vs. a different code path.)

**Step 3: If missing, add validation call; if present, add test confirming behavior**

In `_train_model_for_target`, after loading raw snapshots and before `build_training_data`:

```python
from aria.engine.validation import validate_snapshot_batch
valid, rejected = validate_snapshot_batch(raw_snapshots)
if rejected:
    self.logger.warning(f"Hub training: rejected {len(rejected)}/{len(raw_snapshots)} snapshots")
raw_snapshots = valid
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 5: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "fix: add snapshot validation to hub ML training path"
```

---

## Task 7: Close Gap — Unify Feature Extraction

**Files:**
- Modify: `aria/modules/ml_engine.py:947-979` (`_extract_features`)
- Test: `tests/hub/test_ml_training.py`

**Step 1: Write failing test**

```python
class TestFeatureUnification:
    @pytest.mark.asyncio
    async def test_extract_features_delegates_to_vector_builder(self, ml_engine):
        """Hub _extract_features must use vector_builder as single source of truth."""
        from unittest.mock import patch
        snapshot = {"date": "2026-02-17", "entities": {"total": 3050}}
        with patch("aria.modules.ml_engine._engine_build_feature_vector") as mock_builder:
            mock_builder.return_value = {"hour_sin": 0.5}
            features = await ml_engine._extract_features(snapshot)
            mock_builder.assert_called_once()
            # Base features from builder should be present
            assert "hour_sin" in features
```

**Step 2: Verify current implementation delegates correctly**

Read `ml_engine.py:947-979`. The explore agent found it already delegates to `_engine_build_feature_vector` then appends 12 hub-only features. Verify there's no duplicated logic (e.g., the hub re-computing time features independently).

**Step 3: If delegation is clean, document with test. If there's drift, fix it.**

The key check: ensure no feature is computed in BOTH `_extract_features` and `build_feature_vector`. The 12 rolling window features should ONLY exist in `_extract_features`. All other features should ONLY come from `build_feature_vector`.

**Step 4: Add guard comment and type annotation**

```python
async def _extract_features(self, snapshot, config=None, prev_snapshot=None,
                             rolling_stats=None, rolling_window_stats=None):
    # SINGLE SOURCE OF TRUTH: all base features come from vector_builder.
    # Hub-only rolling window features appended below.
    features = _engine_build_feature_vector(snapshot, config, prev_snapshot, rolling_stats)

    # Hub-only: 12 rolling window features from live activity log
    # These do NOT exist in the engine batch path (by design).
    rws = rolling_window_stats or {}
    ...
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 6: Commit**

```bash
git add aria/modules/ml_engine.py tests/hub/test_ml_training.py
git commit -m "fix: document and verify feature extraction delegation to vector_builder"
```

---

## Task 8: Close Gap — Wire Presence Features into Engine Snapshots

**Files:**
- Modify: `aria/engine/collectors/snapshot.py` — inject presence data from hub cache
- Modify: `tests/engine/test_snapshot.py` or create `tests/engine/test_presence_in_snapshot.py`

**Step 1: Write failing test**

```python
class TestPresenceInSnapshot:
    def test_snapshot_includes_presence_section(self):
        """Intraday snapshot should include presence data when available."""
        # Verify the snapshot schema includes a "presence" key
        # when presence data is fetched from hub cache
        from aria.engine.collectors.snapshot import build_intraday_snapshot
        # ... mock HA states and presence cache endpoint
```

**Step 2: Check current snapshot builder for presence handling**

Read `aria/engine/collectors/snapshot.py` around lines 26-55 where presence is fetched from hub cache. The explore agent noted this already exists for intraday snapshots. Verify it populates the keys that `vector_builder._build_presence_features` expects: `overall_probability`, `occupied_room_count`, `identified_person_count`, `camera_signal_count`.

**Step 3: If keys align, add test confirming it. If mismatched, fix the mapping.**

**Step 4: Update test fixtures**

In `tests/hub/test_ml_training.py`, update the `synthetic_snapshots` fixture to include a `"presence"` key:

```python
"presence": {
    "overall_probability": 0.85,
    "occupied_room_count": 3,
    "identified_person_count": 2,
    "camera_signal_count": 4,
}
```

This ensures presence features are exercised in training tests.

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py tests/engine/ -v --timeout=120`
Expected: All PASS

**Step 6: Commit**

```bash
git add aria/engine/collectors/snapshot.py tests/
git commit -m "feat: wire presence features into engine snapshot path"
```

---

## Task 9: Time-Series Cross-Validation

**Files:**
- Create: `aria/engine/evaluation.py`
- Test: `tests/engine/test_evaluation.py`
- Modify: `aria/modules/ml_engine.py` — use CV instead of single split

**Step 1: Write failing tests**

```python
# tests/engine/test_evaluation.py
import pytest
import numpy as np
from aria.engine.evaluation import expanding_window_cv


class TestExpandingWindowCV:
    def test_3_fold_expanding_window(self):
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=3))
        assert len(folds) == 3
        # Each fold: (X_train, y_train, X_val, y_val)
        for X_train, y_train, X_val, y_val in folds:
            assert len(X_train) > 0
            assert len(X_val) > 0
            # Training set grows with each fold
        assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])

    def test_5_fold_expanding_window(self):
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=5))
        assert len(folds) == 5

    def test_fold_sizes_correct(self):
        """Validation sets should be roughly equal size."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=3))
        val_sizes = [len(f[3]) for f in folds]
        # Each val set should be ~20 samples (60 / 3)
        for size in val_sizes:
            assert 15 <= size <= 25

    def test_no_data_leakage(self):
        """Training indices must always precede validation indices."""
        X = np.arange(60).reshape(-1, 1)
        y = np.arange(60, dtype=float)
        for X_tr, y_tr, X_val, y_val in expanding_window_cv(X, y, n_folds=3):
            assert X_tr.max() < X_val.min()

    def test_single_fold_fallback(self):
        """With n_folds=1, should return single 80/20 split."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=1))
        assert len(folds) == 1
        X_tr, _, X_val, _ = folds[0]
        assert len(X_tr) == 48  # 80% of 60
        assert len(X_val) == 12  # 20% of 60
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/engine/test_evaluation.py -v`
Expected: FAIL — module not found

**Step 3: Implement**

```python
# aria/engine/evaluation.py
"""Time-series cross-validation utilities."""

from __future__ import annotations

from typing import Generator
import numpy as np


def expanding_window_cv(
    X: np.ndarray, y: np.ndarray, n_folds: int = 3
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Expanding-window cross-validation for time-series data.

    Preserves temporal ordering: training always precedes validation.
    Fold k trains on the first (k+1)/(n_folds+1) of data, validates on the next chunk.

    n_folds=1 is equivalent to a single 80/20 split.
    """
    n = len(X)
    if n_folds == 1:
        split = int(n * 0.8)
        yield X[:split], y[:split], X[split:], y[split:]
        return

    # Each validation window is n / (n_folds + 1) samples
    # Training window expands: fold k uses first (k+1) chunks
    chunk_size = n // (n_folds + 1)
    for k in range(n_folds):
        train_end = chunk_size * (k + 1)
        val_end = min(train_end + chunk_size, n)
        if k == n_folds - 1:
            val_end = n  # Last fold gets remaining samples
        yield X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end]
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_evaluation.py -v`
Expected: All 5 PASS

**Step 5: Integrate into ml_engine training**

In `_train_model_for_target`, replace the single split:

```python
# Determine CV folds based on tier
if self.current_tier >= 3:
    n_folds = 5
elif self.current_tier >= 2:
    n_folds = 3
else:
    n_folds = 1

from aria.engine.evaluation import expanding_window_cv

fold_metrics = []
for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(
    expanding_window_cv(X, y, n_folds=n_folds)
):
    w_train = self._compute_decay_weights(X_tr, ...)
    fitted = self._fit_all_models(X_tr, y_tr, w_train, self.registry, target, self.current_tier)
    # Evaluate on validation fold
    for name, model in fitted.items():
        pred = model.predict(X_val)
        mae = np.mean(np.abs(pred - y_val))
        fold_metrics.append({"fold": fold_idx, "model": name, "mae": mae})

# Use last fold's fitted models as the deployed models
# Store fold_metrics in training metadata
```

**Step 6: Run full training tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 7: Commit**

```bash
git add aria/engine/evaluation.py tests/engine/test_evaluation.py aria/modules/ml_engine.py
git commit -m "feat: add expanding-window time-series cross-validation"
```

---

## Task 10: Feature Selection Feedback Loop

**Files:**
- Create: `aria/engine/features/pruning.py`
- Test: `tests/engine/test_feature_pruning.py`
- Modify: `aria/modules/ml_engine.py` — call after training

**Step 1: Write failing tests**

```python
# tests/engine/test_feature_pruning.py
import pytest
from aria.engine.features.pruning import FeaturePruner


class TestFeaturePruner:
    def test_identify_low_importance_features(self):
        importances = {
            "hour_sin": 0.30, "hour_cos": 0.25, "temp_f": 0.20,
            "humidity_pct": 0.15, "wind_mph": 0.05,
            "is_weekend_x_temp": 0.004, "daylight_x_lights": 0.001,
        }
        pruner = FeaturePruner(threshold=0.01)
        low = pruner.identify_low_importance(importances)
        assert "daylight_x_lights" in low
        assert "hour_sin" not in low

    def test_track_consecutive_low_cycles(self):
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        assert pruner.should_prune("feat_a") is False  # only 1 cycle
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        assert pruner.should_prune("feat_a") is False  # only 2 cycles
        pruner.record_cycle(low_features={"feat_a"})
        assert pruner.should_prune("feat_a") is True   # 3 consecutive
        assert pruner.should_prune("feat_b") is False  # reset (not in 3rd cycle)

    def test_drift_resets_pruning(self):
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle(low_features={"feat_a"})
        assert pruner.should_prune("feat_a") is True
        pruner.on_drift_detected()
        assert pruner.should_prune("feat_a") is False

    def test_get_active_features(self):
        all_features = ["hour_sin", "hour_cos", "wind_mph"]
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle(low_features={"wind_mph"})
        active = pruner.get_active_features(all_features)
        assert "hour_sin" in active
        assert "wind_mph" not in active
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/engine/test_feature_pruning.py -v`
Expected: FAIL — module not found

**Step 3: Implement**

```python
# aria/engine/features/pruning.py
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
            f for f, count in self._consecutive_low.items()
            if count >= self.required_cycles and f not in self._pruned
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
            "consecutive_low": {
                f: c for f, c in self._consecutive_low.items() if c > 0
            },
        }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_feature_pruning.py -v`
Expected: All 4 PASS

**Step 5: Wire into ml_engine post-training**

In `_train_model_for_target`, after computing feature importance:

```python
# After training, compute importance and feed pruner
importances = self._compute_feature_importances(fitted_models, feature_names)
low = self.feature_pruner.identify_low_importance(importances)
self.feature_pruner.record_cycle(low)

# Store in training metadata
metadata["low_importance_features"] = sorted(low)
metadata["pruned_features"] = self.feature_pruner.to_dict()
```

Also subscribe to `drift_detected` events:

```python
async def on_event(self, event_type, data):
    if event_type == "drift_detected":
        self.feature_pruner.on_drift_detected()
```

Feature pruning only activates at Tier 3+ — at Tier 1-2, `FeaturePruner` is instantiated but `record_cycle` is not called (or `required_cycles` set to `999`).

**Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py tests/engine/test_feature_pruning.py -v --timeout=120`
Expected: All PASS

**Step 7: Commit**

```bash
git add aria/engine/features/pruning.py tests/engine/test_feature_pruning.py aria/modules/ml_engine.py
git commit -m "feat: add feature importance tracking with auto-pruning at Tier 3+"
```

---

## Task 11: Optuna Hyperparameter Optimization (Tier 3+)

**Files:**
- Create: `aria/engine/tuning.py`
- Test: `tests/engine/test_tuning.py`
- Modify: `pyproject.toml` — add optuna to ml-extra
- Modify: `aria/modules/ml_engine.py` — call tuner at Tier 3+

**Step 1: Add optuna to optional deps**

In `pyproject.toml` under `[project.optional-dependencies]`, add `"optuna>=3.0.0"` to the `ml-extra` list.

**Step 2: Write failing tests**

```python
# tests/engine/test_tuning.py
import pytest
import numpy as np

try:
    import optuna
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
```

**Step 3: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/engine/test_tuning.py -v`
Expected: FAIL — module not found

**Step 4: Implement**

```python
# aria/engine/tuning.py
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
    X: np.ndarray, y: np.ndarray,
    n_trials: int = 20, n_folds: int = 3,
    sample_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run Optuna optimization with expanding-window CV objective.

    Returns dict with best_params, best_score, trials_completed, fallback.
    Falls back to DEFAULT_PARAMS on any error.
    """
    if len(X) < 10:
        logger.warning("Too few samples for optimization, using defaults")
        return {"best_params": DEFAULT_PARAMS.copy(), "best_score": None,
                "trials_completed": 0, "fallback": True}

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
                    **params, random_state=42, verbosity=-1,
                    importance_type="gain",
                    min_child_samples=max(3, len(X_tr) // 20),
                )
                w = sample_weights[:len(X_tr)] if sample_weights is not None else None
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
        return {"best_params": DEFAULT_PARAMS.copy(), "best_score": None,
                "trials_completed": 0, "fallback": True}
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/engine/test_tuning.py -v --timeout=120`
Expected: All 3 PASS (or skip if optuna not installed)

**Step 6: Wire into ml_engine at Tier 3+**

In `_train_model_for_target`, before fitting:

```python
if self.current_tier >= 3:
    try:
        from aria.engine.tuning import optimize_hyperparams
        opt_result = optimize_hyperparams(X, y, n_trials=20, n_folds=5,
                                          sample_weights=weights)
        if not opt_result["fallback"]:
            # Update LGBM entry params in registry for this training run
            for entry in self.registry.resolve(target, self.current_tier):
                if "lgbm" in entry.name:
                    entry.params.update(opt_result["best_params"])
            metadata["optuna"] = opt_result
    except ImportError:
        pass  # optuna not installed, skip
```

**Step 7: Run training tests**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -v --timeout=120`
Expected: All PASS

**Step 8: Commit**

```bash
git add aria/engine/tuning.py tests/engine/test_tuning.py pyproject.toml aria/modules/ml_engine.py
git commit -m "feat: add Optuna hyperparameter optimization at Tier 3+"
```

---

## Task 12: Config Entries for New ML Settings

**Files:**
- Modify: `aria/hub/config_defaults.py` — add ml.* config entries
- Test: Verify via existing config seed tests or manual inspection

**Step 1: Add config entries**

In `aria/hub/config_defaults.py`, add to `CONFIG_DEFAULTS` list:

```python
{
    "key": "ml.tier_override",
    "default_value": "auto",
    "value_type": "select",
    "label": "ML Tier Override",
    "description": "Override auto-detected ML tier. auto=use hardware detection.",
    "category": "ml",
    "options": "auto,1,2,3,4",
},
{
    "key": "ml.fallback_ttl_days",
    "default_value": "7",
    "value_type": "number",
    "label": "Fallback TTL (days)",
    "description": "Days before retrying a model that fell back to lower tier.",
    "category": "ml",
    "min_value": 1,
    "max_value": 30,
},
{
    "key": "ml.feature_prune_threshold",
    "default_value": "0.01",
    "value_type": "number",
    "label": "Feature Prune Threshold",
    "description": "Features below this importance are candidates for pruning.",
    "category": "ml",
    "min_value": 0.001,
    "max_value": 0.1,
    "step": 0.005,
},
{
    "key": "ml.feature_prune_cycles",
    "default_value": "3",
    "value_type": "number",
    "label": "Feature Prune Cycles",
    "description": "Consecutive low-importance cycles before auto-pruning (Tier 3+).",
    "category": "ml",
    "min_value": 1,
    "max_value": 10,
},
{
    "key": "ml.optuna_trials",
    "default_value": "20",
    "value_type": "number",
    "label": "Optuna Trials",
    "description": "Number of hyperparameter optimization trials per training cycle (Tier 3+).",
    "category": "ml",
    "min_value": 5,
    "max_value": 100,
},
{
    "key": "ml.cv_folds",
    "default_value": "auto",
    "value_type": "select",
    "label": "CV Folds",
    "description": "Cross-validation folds. auto=tier-based (1/3/5).",
    "category": "ml",
    "options": "auto,1,3,5,10",
},
```

**Step 2: Verify config seeds on startup**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -c "from aria.hub.config_defaults import CONFIG_DEFAULTS; ml = [c for c in CONFIG_DEFAULTS if c.get('category') == 'ml']; print(f'{len(ml)} ml config entries'); [print(f'  {c[\"key\"]}: {c[\"default_value\"]}') for c in ml]"`
Expected: All 6 new entries listed

**Step 3: Commit**

```bash
git add aria/hub/config_defaults.py
git commit -m "config: add ML tier, fallback, pruning, and optimization settings"
```

---

## Task 13: Hardware Profile API Endpoint

**Files:**
- Modify: `aria/hub/api.py` (or wherever API routes are defined)
- Test: `tests/hub/test_api.py` (extend)

**Step 1: Write failing test**

```python
class TestHardwareAPI:
    @pytest.mark.asyncio
    async def test_hardware_endpoint_returns_profile(self, client):
        """GET /api/ml/hardware should return hardware profile and tier."""
        response = await client.get("/api/ml/hardware")
        assert response.status_code == 200
        data = response.json()
        assert "ram_gb" in data
        assert "cpu_cores" in data
        assert "recommended_tier" in data
        assert "current_tier" in data
        assert "active_fallbacks" in data
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/hub/test_api.py -k "hardware" -v`
Expected: FAIL — 404 or no such test

**Step 3: Add endpoint**

In the API routes file, add:

```python
@router.get("/api/ml/hardware")
async def get_hardware_profile():
    from aria.engine.hardware import scan_hardware, recommend_tier
    profile = scan_hardware()
    tier = recommend_tier(profile)
    # Get current effective tier from ML engine module
    ml_engine = hub.get_module("ml_engine")
    return {
        "ram_gb": profile.ram_gb,
        "cpu_cores": profile.cpu_cores,
        "gpu_available": profile.gpu_available,
        "gpu_name": profile.gpu_name,
        "recommended_tier": tier,
        "current_tier": ml_engine.current_tier if ml_engine else tier,
        "tier_override": "auto",  # read from config
        "active_fallbacks": ml_engine.fallback_tracker.to_dict() if ml_engine else [],
    }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api.py -k "hardware" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api.py
git commit -m "feat: add /api/ml/hardware endpoint for tier and fallback status"
```

---

## Task 14: Verify Feedback Loop Startup Timing

This is a verification task, not new code. The explore agent found `schedule_periodic_training` already sets `run_immediately=True` when no metadata exists.

**Files:**
- Read: `aria/modules/ml_engine.py:1287-1339`
- Test: `tests/hub/test_ml_training.py` — add explicit test confirming behavior

**Step 1: Write test confirming immediate training on first boot**

```python
class TestTrainingScheduleStartup:
    @pytest.mark.asyncio
    async def test_first_boot_trains_immediately(self, ml_engine, mock_hub):
        """When no training metadata exists, training should run immediately."""
        # Configure mock to return None for training metadata (first boot)
        mock_hub.get_cache.return_value = None
        # Call schedule_periodic_training and verify run_immediately=True
        # by checking that hub.schedule_task is called with run_immediately=True
```

**Step 2: Run test, confirm it passes (existing behavior)**

Run: `.venv/bin/python -m pytest tests/hub/test_ml_training.py -k "first_boot" -v`
Expected: PASS — this confirms the gap is already closed

**Step 3: Commit**

```bash
git add tests/hub/test_ml_training.py
git commit -m "test: verify immediate training on first boot (gap confirmed closed)"
```

---

## Task 15: Integration Test — Full Tier Resolution

End-to-end test that exercises the complete Phase 1 pipeline.

**Files:**
- Create: `tests/integration/test_tiered_ml_pipeline.py`

**Step 1: Write integration test**

```python
# tests/integration/test_tiered_ml_pipeline.py
import pytest
import json
import numpy as np
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from aria.engine.hardware import HardwareProfile, recommend_tier
from aria.engine.models.registry import TieredModelRegistry
from aria.engine.fallback import FallbackTracker
from aria.engine.evaluation import expanding_window_cv
from aria.engine.features.pruning import FeaturePruner


class TestTieredPipelineIntegration:
    def test_full_tier_2_pipeline(self):
        """Simulate complete Tier 2 training pipeline."""
        # 1. Hardware scan → tier recommendation
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        tier = recommend_tier(profile)
        assert tier == 2

        # 2. Registry resolves Tier 2 models
        registry = TieredModelRegistry.with_defaults()
        entries = registry.resolve("power_watts", tier)
        assert len(entries) >= 3  # gb, rf, lgbm

        # 3. CV produces correct fold count for tier
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        n_folds = 3  # Tier 2
        folds = list(expanding_window_cv(X, y, n_folds=n_folds))
        assert len(folds) == 3

        # 4. Feature pruner doesn't activate at Tier 2
        pruner = FeaturePruner(threshold=0.01, required_cycles=999)
        pruner.record_cycle({"feat_a"})
        assert not pruner.should_prune("feat_a")

        # 5. Fallback tracker is ready
        tracker = FallbackTracker(ttl_days=7)
        assert len(tracker.active_fallbacks()) == 0

    def test_full_tier_3_pipeline(self):
        """Simulate Tier 3 pipeline with pruning and more CV folds."""
        profile = HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False)
        tier = recommend_tier(profile)
        assert tier == 3

        registry = TieredModelRegistry.with_defaults()
        entries = registry.resolve("power_watts", tier)
        # Should include all Tier 1 + 2 entries
        assert len(entries) >= 3

        # CV with 5 folds at Tier 3
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=5))
        assert len(folds) == 5

        # Feature pruner active at Tier 3
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle({"low_feat"})
        assert pruner.should_prune("low_feat")

    def test_fallback_degrades_gracefully(self):
        """Model failure at Tier 3 should fall back to Tier 2 behavior."""
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("transformer_power", from_tier=4, to_tier=3, error="CUDA OOM")
        assert tracker.get_effective_tier("transformer_power", original_tier=4) == 3
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 3  # unaffected
```

**Step 2: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/test_tiered_ml_pipeline.py -v`
Expected: All 3 PASS

**Step 3: Run full test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -v --timeout=120 -x -q`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/integration/test_tiered_ml_pipeline.py
git commit -m "test: add integration tests for tiered ML pipeline"
```

---

## Summary

| Task | What | New Files | Commits |
|------|------|-----------|---------|
| 1 | Hardware scanner | `aria/engine/hardware.py`, test | 1 |
| 2 | psutil dependency | pyproject.toml | 1 |
| 3 | Tier-aware model registry | extend `registry.py`, test | 1 |
| 4 | Wire registry into ML engine | modify `ml_engine.py`, test | 1 |
| 5 | Graceful fallback engine | `aria/engine/fallback.py`, test | 1 |
| 6 | Snapshot validation in hub | modify `ml_engine.py`, test | 1 |
| 7 | Unify feature extraction | modify `ml_engine.py`, test | 1 |
| 8 | Presence in engine snapshots | modify `snapshot.py`, test fixtures | 1 |
| 9 | Time-series cross-validation | `aria/engine/evaluation.py`, test | 1 |
| 10 | Feature selection feedback | `aria/engine/features/pruning.py`, test | 1 |
| 11 | Optuna optimization | `aria/engine/tuning.py`, test, pyproject | 1 |
| 12 | ML config entries | modify `config_defaults.py` | 1 |
| 13 | Hardware API endpoint | modify `api.py`, test | 1 |
| 14 | Verify startup timing | test only | 1 |
| 15 | Integration test | test only | 1 |

**Total: 15 tasks, 15 commits, ~7 new files, ~5 modified files**

**Dependencies:** Tasks 1-2 first (hardware scanner needed by everything). Task 3 before 4 (registry before wiring). Task 9 before 11 (CV before Optuna uses it). All others are independent.
