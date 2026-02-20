# Capability Propagation Framework — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a code-defined capability registry to ARIA that inventories all 22 stable capabilities, validates them against tests/config/deps, and exposes them via CLI + API + dashboard.

**Architecture:** Each hub module and engine command declares its capabilities via a `CAPABILITIES` class/module attribute using a `Capability` dataclass. A `CapabilityRegistry` collects all declarations, validates cross-references (config keys, test paths, deps), and serves them through `aria capabilities` CLI commands and `/api/capabilities` REST endpoints.

**Tech Stack:** Python dataclasses, pytest markers, FastAPI endpoints, Preact JSX (dashboard page)

---

## Task 1: Capability Dataclass + Registry Core

**Files:**
- Create: `aria/capabilities.py`
- Test: `tests/test_capabilities.py`

**Step 1: Write the failing test**

```python
# tests/test_capabilities.py
"""Tests for the capability registry."""

import pytest
from aria.capabilities import Capability, CapabilityRegistry


class TestCapabilityDataclass:
    def test_capability_creation(self):
        cap = Capability(
            id="test_cap",
            name="Test Capability",
            description="A test capability",
            module="test_module",
            layer="hub",
            config_keys=["test.key1"],
            test_paths=["tests/test_something.py"],
            test_markers=["test"],
            runtime_deps=[],
            optional_deps=[],
            data_paths=[],
            systemd_units=[],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=[],
        )
        assert cap.id == "test_cap"
        assert cap.layer == "hub"
        assert cap.status == "stable"

    def test_capability_invalid_layer(self):
        with pytest.raises(ValueError, match="layer"):
            Capability(
                id="bad", name="Bad", description="bad", module="x",
                layer="invalid", config_keys=[], test_paths=[], test_markers=[],
                runtime_deps=[], optional_deps=[], data_paths=[],
                systemd_units=[], pipeline_stage=None, status="stable",
                added_version="1.0.0", depends_on=[],
            )

    def test_capability_invalid_status(self):
        with pytest.raises(ValueError, match="status"):
            Capability(
                id="bad", name="Bad", description="bad", module="x",
                layer="hub", config_keys=[], test_paths=[], test_markers=[],
                runtime_deps=[], optional_deps=[], data_paths=[],
                systemd_units=[], pipeline_stage=None, status="invalid",
                added_version="1.0.0", depends_on=[],
            )


class TestCapabilityRegistry:
    def test_register_and_list(self):
        registry = CapabilityRegistry()
        cap = Capability(
            id="test_cap", name="Test", description="test", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        assert "test_cap" in registry.list_ids()
        assert registry.get("test_cap") is cap

    def test_register_duplicate_raises(self):
        registry = CapabilityRegistry()
        cap = Capability(
            id="dup", name="Dup", description="dup", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(cap)

    def test_list_by_layer(self):
        registry = CapabilityRegistry()
        hub_cap = Capability(
            id="hub1", name="Hub", description="hub", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        engine_cap = Capability(
            id="eng1", name="Engine", description="engine", module="mod",
            layer="engine", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(hub_cap)
        registry.register(engine_cap)
        hub_caps = registry.list_by_layer("hub")
        assert len(hub_caps) == 1
        assert hub_caps[0].id == "hub1"

    def test_dependency_graph(self):
        registry = CapabilityRegistry()
        cap_a = Capability(
            id="a", name="A", description="a", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        cap_b = Capability(
            id="b", name="B", description="b", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=["a"],
        )
        registry.register(cap_a)
        registry.register(cap_b)
        graph = registry.dependency_graph()
        assert graph == {"a": [], "b": ["a"]}

    def test_cycle_detection(self):
        registry = CapabilityRegistry()
        cap_a = Capability(
            id="a", name="A", description="a", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=["b"],
        )
        cap_b = Capability(
            id="b", name="B", description="b", module="mod",
            layer="hub", config_keys=[], test_paths=[], test_markers=[],
            runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=["a"],
        )
        registry.register(cap_a)
        registry.register(cap_b)
        issues = registry.validate_deps()
        assert any("cycle" in issue.lower() for issue in issues)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v --timeout=30`
Expected: FAIL with `ModuleNotFoundError: No module named 'aria.capabilities'`

**Step 3: Write minimal implementation**

```python
# aria/capabilities.py
"""ARIA Capability Registry — declares, collects, and validates capabilities."""

from dataclasses import dataclass, field
from typing import Optional


VALID_LAYERS = {"hub", "engine", "dashboard", "cross-cutting"}
VALID_STATUSES = {"stable", "experimental", "planned"}
VALID_PIPELINE_STAGES = {"backtest", "shadow", "suggest", "autonomous", None}


@dataclass(frozen=True)
class Capability:
    """A declared ARIA capability with metadata for testing, config, and deployment."""

    id: str
    name: str
    description: str
    module: str
    layer: str
    config_keys: list[str] = field(default_factory=list)
    test_paths: list[str] = field(default_factory=list)
    test_markers: list[str] = field(default_factory=list)
    runtime_deps: list[str] = field(default_factory=list)
    optional_deps: list[str] = field(default_factory=list)
    data_paths: list[str] = field(default_factory=list)
    systemd_units: list[str] = field(default_factory=list)
    pipeline_stage: Optional[str] = None
    status: str = "stable"
    added_version: str = "1.0.0"
    depends_on: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layer not in VALID_LAYERS:
            raise ValueError(f"Invalid layer '{self.layer}'. Must be one of: {VALID_LAYERS}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of: {VALID_STATUSES}")
        if self.pipeline_stage not in VALID_PIPELINE_STAGES:
            raise ValueError(
                f"Invalid pipeline_stage '{self.pipeline_stage}'. Must be one of: {VALID_PIPELINE_STAGES}"
            )


class CapabilityRegistry:
    """Collects and validates capability declarations."""

    def __init__(self):
        self._capabilities: dict[str, Capability] = {}

    def register(self, capability: Capability):
        """Register a capability. Raises ValueError on duplicate id."""
        if capability.id in self._capabilities:
            raise ValueError(f"Capability '{capability.id}' already registered")
        self._capabilities[capability.id] = capability

    def get(self, capability_id: str) -> Optional[Capability]:
        """Get a capability by id."""
        return self._capabilities.get(capability_id)

    def list_ids(self) -> list[str]:
        """Return all registered capability ids."""
        return list(self._capabilities.keys())

    def list_all(self) -> list[Capability]:
        """Return all registered capabilities."""
        return list(self._capabilities.values())

    def list_by_layer(self, layer: str) -> list[Capability]:
        """Return capabilities filtered by layer."""
        return [c for c in self._capabilities.values() if c.layer == layer]

    def list_by_status(self, status: str) -> list[Capability]:
        """Return capabilities filtered by status."""
        return [c for c in self._capabilities.values() if c.status == status]

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return {id: [depends_on_ids]} for all capabilities."""
        return {c.id: list(c.depends_on) for c in self._capabilities.values()}

    def validate_deps(self) -> list[str]:
        """Validate dependency references and detect cycles.

        Returns list of issue descriptions. Empty = valid.
        """
        issues = []
        # Check all deps reference registered capabilities
        for cap in self._capabilities.values():
            for dep_id in cap.depends_on:
                if dep_id not in self._capabilities:
                    issues.append(
                        f"Capability '{cap.id}' depends on '{dep_id}' which is not registered"
                    )

        # Detect cycles via DFS
        visited = set()
        rec_stack = set()

        def _has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            cap = self._capabilities.get(node_id)
            if cap:
                for dep_id in cap.depends_on:
                    if dep_id not in visited:
                        if _has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        issues.append(f"Dependency cycle detected involving '{node_id}' and '{dep_id}'")
                        return True
            rec_stack.discard(node_id)
            return False

        for cap_id in self._capabilities:
            if cap_id not in visited:
                _has_cycle(cap_id)

        return issues
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v --timeout=30`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add aria/capabilities.py tests/test_capabilities.py
git commit -m "feat: add Capability dataclass and CapabilityRegistry core"
```

---

## Task 2: Validation Engine (config keys, test paths, optional deps)

**Files:**
- Modify: `aria/capabilities.py`
- Test: `tests/test_capabilities.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_capabilities.py`:

```python
class TestRegistryValidation:
    def test_validate_config_keys_missing(self, tmp_path):
        """Config key not in config_defaults should be flagged."""
        registry = CapabilityRegistry()
        cap = Capability(
            id="test", name="Test", description="test", module="mod",
            layer="hub", config_keys=["nonexistent.key"], test_paths=[],
            test_markers=[], runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        issues = registry.validate_config_keys()
        assert any("nonexistent.key" in issue for issue in issues)

    def test_validate_config_keys_present(self):
        """Real config key should pass validation."""
        registry = CapabilityRegistry()
        cap = Capability(
            id="test", name="Test", description="test", module="mod",
            layer="hub", config_keys=["shadow.min_confidence"], test_paths=[],
            test_markers=[], runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        issues = registry.validate_config_keys()
        assert len(issues) == 0

    def test_validate_test_paths_missing(self):
        """Non-existent test path should be flagged."""
        registry = CapabilityRegistry()
        cap = Capability(
            id="test", name="Test", description="test", module="mod",
            layer="hub", config_keys=[], test_paths=["tests/does_not_exist.py"],
            test_markers=[], runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        issues = registry.validate_test_paths()
        assert any("does_not_exist" in issue for issue in issues)

    def test_validate_test_paths_present(self):
        """Real test path should pass validation."""
        registry = CapabilityRegistry()
        cap = Capability(
            id="test", name="Test", description="test", module="mod",
            layer="hub", config_keys=[],
            test_paths=["tests/hub/test_shadow_engine.py"],
            test_markers=[], runtime_deps=[], optional_deps=[], data_paths=[],
            systemd_units=[], pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=[],
        )
        registry.register(cap)
        issues = registry.validate_test_paths()
        assert len(issues) == 0

    def test_validate_all_combines_checks(self):
        """validate_all() should run all validation checks."""
        registry = CapabilityRegistry()
        cap = Capability(
            id="test", name="Test", description="test", module="mod",
            layer="hub", config_keys=["fake.key"],
            test_paths=["tests/fake.py"], test_markers=[], runtime_deps=[],
            optional_deps=[], data_paths=[], systemd_units=[],
            pipeline_stage=None, status="stable",
            added_version="1.0.0", depends_on=["nonexistent"],
        )
        registry.register(cap)
        issues = registry.validate_all()
        assert len(issues) >= 3  # config key + test path + missing dep
```

**Step 2: Run test to verify failures**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py::TestRegistryValidation -v --timeout=30`
Expected: FAIL — `validate_config_keys`, `validate_test_paths`, `validate_all` not defined

**Step 3: Implement validation methods**

Add to `CapabilityRegistry` in `aria/capabilities.py`:

```python
    def validate_config_keys(self) -> list[str]:
        """Check that every declared config_key exists in config_defaults.py."""
        from aria.hub.config_defaults import CONFIG_DEFAULTS
        known_keys = {d["key"] for d in CONFIG_DEFAULTS}
        issues = []
        for cap in self._capabilities.values():
            for key in cap.config_keys:
                if key not in known_keys:
                    issues.append(
                        f"Capability '{cap.id}': config key '{key}' not found in config_defaults.py"
                    )
        return issues

    def validate_test_paths(self) -> list[str]:
        """Check that every declared test_path exists on disk."""
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        issues = []
        for cap in self._capabilities.values():
            for path in cap.test_paths:
                full_path = os.path.join(project_root, path)
                if not os.path.exists(full_path):
                    issues.append(
                        f"Capability '{cap.id}': test path '{path}' does not exist"
                    )
        return issues

    def validate_all(self) -> list[str]:
        """Run all validation checks. Returns combined issues list."""
        issues = []
        issues.extend(self.validate_deps())
        issues.extend(self.validate_config_keys())
        issues.extend(self.validate_test_paths())
        return issues
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v --timeout=30`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add aria/capabilities.py tests/test_capabilities.py
git commit -m "feat: add capability validation (config keys, test paths, deps)"
```

---

## Task 3: Collector Function + Hub Module Declarations

**Files:**
- Modify: `aria/capabilities.py` (add `collect_from_modules`)
- Modify: `aria/hub/core.py` (add `CAPABILITIES = []` default to Module base class)
- Modify: `aria/modules/discovery.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/ml_engine.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/patterns.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/orchestrator.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/shadow_engine.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/data_quality.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/organic_discovery/module.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/intelligence.py` (add CAPABILITIES declaration)
- Modify: `aria/modules/activity_monitor.py` (add CAPABILITIES declaration)
- Test: `tests/test_capabilities.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_capabilities.py`:

```python
class TestCollectFromModules:
    def test_collect_finds_hub_modules(self):
        """collect_from_modules() should find all 9 hub module capabilities."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        hub_caps = registry.list_by_layer("hub")
        # All 9 hub modules should declare at least 1 capability
        hub_module_ids = {c.module for c in hub_caps}
        expected_modules = {
            "discovery", "ml_engine", "pattern_recognition", "orchestrator",
            "shadow_engine", "data_quality", "organic_discovery",
            "intelligence", "activity_monitor",
        }
        assert expected_modules.issubset(hub_module_ids), (
            f"Missing modules: {expected_modules - hub_module_ids}"
        )

    def test_collect_finds_engine_capabilities(self):
        """collect_from_modules() should find engine capabilities."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        engine_caps = registry.list_by_layer("engine")
        assert len(engine_caps) >= 10  # At least 10 engine capabilities

    def test_all_collected_capabilities_validate(self):
        """All real capability declarations should pass validation."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        issues = registry.validate_all()
        assert issues == [], f"Validation issues found:\n" + "\n".join(issues)

    def test_total_capability_count(self):
        """Should find at least 22 stable capabilities."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        stable = registry.list_by_status("stable")
        assert len(stable) >= 22, f"Only found {len(stable)} stable capabilities"
```

**Step 2: Run test to verify failures**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py::TestCollectFromModules -v --timeout=30`
Expected: FAIL — `collect_from_modules` not defined

**Step 3: Add CAPABILITIES to Module base class**

In `aria/hub/core.py`, add to `Module.__init__` area:

```python
class Module:
    """Base class for hub modules."""

    CAPABILITIES: list = []  # Subclasses declare their capabilities here

    def __init__(self, module_id: str, hub: "IntelligenceHub"):
        ...
```

**Step 4: Add declarations to all 9 hub modules**

Each module gets a `CAPABILITIES` list. Here are the declarations (add as class attribute right after the class docstring, before `__init__`):

**`aria/modules/discovery.py`:**
```python
from aria.capabilities import Capability

class DiscoveryModule(Module):
    CAPABILITIES = [
        Capability(
            id="discovery",
            name="HA Entity Discovery",
            description="Scans HA REST + WebSocket for entities, devices, areas, and capabilities",
            module="discovery",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_discover.py"],
            test_markers=["discover"],
            runtime_deps=[],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=[],
        ),
    ]
```

**`aria/modules/ml_engine.py`:**
```python
from aria.capabilities import Capability

class MLEngine(Module):
    CAPABILITIES = [
        Capability(
            id="ml_realtime",
            name="Real-time ML Engine",
            description="Feature engineering, model training (GradientBoosting, RandomForest, LightGBM), periodic retraining",
            module="ml_engine",
            layer="hub",
            config_keys=[
                "features.decay_half_life_days",
                "features.weekday_alignment_bonus",
            ],
            test_paths=["tests/hub/test_ml_training.py", "tests/hub/test_reference_model.py"],
            test_markers=["ml"],
            runtime_deps=["discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]
```

**`aria/modules/patterns.py`:**
```python
from aria.capabilities import Capability

class PatternRecognition(Module):
    CAPABILITIES = [
        Capability(
            id="pattern_recognition",
            name="Event Sequence Patterns",
            description="Detects recurring event sequences from HA logbook data",
            module="pattern_recognition",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_patterns.py"],
            test_markers=["patterns"],
            runtime_deps=["discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]
```

**`aria/modules/orchestrator.py`:**
```python
from aria.capabilities import Capability

class OrchestratorModule(Module):
    CAPABILITIES = [
        Capability(
            id="orchestrator",
            name="Automation Suggestions",
            description="Generates HA automation YAML from detected patterns",
            module="orchestrator",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_orchestrator.py"],
            test_markers=["orchestrator"],
            runtime_deps=["pattern_recognition"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["pattern_recognition"],
        ),
    ]
```

**`aria/modules/shadow_engine.py`:**
```python
from aria.capabilities import Capability

class ShadowEngine(Module):
    CAPABILITIES = [
        Capability(
            id="shadow_predictions",
            name="Shadow Mode Predictions",
            description="Predict-compare-score loop that learns from implicit feedback",
            module="shadow_engine",
            layer="hub",
            config_keys=[
                "shadow.min_confidence",
                "shadow.default_window_seconds",
                "shadow.resolution_interval_s",
                "shadow.prediction_cooldown_s",
            ],
            test_paths=["tests/hub/test_shadow_engine.py", "tests/hub/test_cache_shadow.py", "tests/hub/test_api_shadow.py"],
            test_markers=["shadow"],
            runtime_deps=["activity_monitoring", "discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage="shadow",
            status="stable",
            added_version="1.0.0",
            depends_on=["activity_monitoring", "discovery"],
        ),
    ]
```

**`aria/modules/data_quality.py`:**
```python
from aria.capabilities import Capability

class DataQualityModule(Module):
    CAPABILITIES = [
        Capability(
            id="data_quality",
            name="Entity Curation Pipeline",
            description="Entity classification: auto-exclude, edge cases, default include",
            module="data_quality",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_data_quality.py"],
            test_markers=["data_quality"],
            runtime_deps=["discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]
```

**`aria/modules/organic_discovery/module.py`:**
```python
from aria.capabilities import Capability

class OrganicDiscoveryModule(Module):
    CAPABILITIES = [
        Capability(
            id="organic_discovery",
            name="Organic Capability Discovery",
            description="Two-layer HDBSCAN clustering: domain attributes + temporal co-occurrence",
            module="organic_discovery",
            layer="hub",
            config_keys=[
                "organic.autonomy_mode",
                "organic.naming_backend",
            ],
            test_paths=[
                "tests/hub/test_organic_discovery_module.py",
                "tests/hub/test_organic_clustering.py",
                "tests/hub/test_organic_behavioral.py",
                "tests/hub/test_organic_feature_vectors.py",
                "tests/hub/test_organic_naming.py",
                "tests/hub/test_organic_scoring.py",
                "tests/hub/test_organic_seed_validation.py",
                "tests/hub/test_api_organic_discovery.py",
            ],
            test_markers=["organic"],
            runtime_deps=["discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service", "aria-organic-discovery.timer"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]
```

**`aria/modules/intelligence.py`:**
```python
from aria.capabilities import Capability

class IntelligenceModule(Module):
    CAPABILITIES = [
        Capability(
            id="intelligence_assembly",
            name="Unified Intelligence Cache",
            description="Assembles daily/intraday snapshots, baselines, predictions, ML scores into unified cache",
            module="intelligence",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_intelligence.py"],
            test_markers=["intelligence"],
            runtime_deps=[],
            optional_deps=[],
            data_paths=["~/ha-logs/intelligence/"],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=[],
        ),
    ]
```

**`aria/modules/activity_monitor.py`:**
```python
from aria.capabilities import Capability

class ActivityMonitor(Module):
    CAPABILITIES = [
        Capability(
            id="activity_monitoring",
            name="WebSocket Activity Monitor",
            description="WebSocket state_changed listener, 15-min windows, adaptive snapshots, prediction analytics",
            module="activity_monitor",
            layer="hub",
            config_keys=[
                "activity.daily_snapshot_cap",
                "activity.snapshot_cooldown_s",
                "activity.flush_interval_s",
                "activity.max_window_age_h",
            ],
            test_paths=["tests/hub/test_activity_monitor.py"],
            test_markers=["activity"],
            runtime_deps=["discovery"],
            optional_deps=[],
            data_paths=[],
            systemd_units=["aria-hub.service"],
            pipeline_stage=None,
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]
```

**Step 5: Add engine capability declarations**

Create `aria/engine/capabilities.py`:

```python
"""Engine capability declarations."""

from aria.capabilities import Capability

ENGINE_CAPABILITIES = [
    Capability(id="snapshot", name="Daily State Snapshot", description="Collect current HA state snapshot", module="engine.snapshot", layer="engine", config_keys=[], test_paths=["tests/engine/test_collectors.py"], test_markers=["snapshot"], runtime_deps=[], optional_deps=[], data_paths=["~/ha-logs/intelligence/daily/"], systemd_units=["aria-full.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=[]),
    Capability(id="predictions", name="Prediction Generation", description="Generate predictions from latest snapshot", module="engine.predictions", layer="engine", config_keys=[], test_paths=["tests/engine/test_predictions.py"], test_markers=["predictions"], runtime_deps=[], optional_deps=[], data_paths=["~/ha-logs/intelligence/predictions/"], systemd_units=["aria-full.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["snapshot"]),
    Capability(id="scoring", name="Prediction Scoring", description="Score yesterday's predictions against actuals", module="engine.predictions", layer="engine", config_keys=[], test_paths=["tests/engine/test_predictions.py"], test_markers=["scoring"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-score.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["predictions"]),
    Capability(id="model_training", name="ML Model Training", description="Retrain GradientBoosting, RandomForest, IsolationForest models", module="engine.models", layer="engine", config_keys=[], test_paths=["tests/engine/test_models.py"], test_markers=["training"], runtime_deps=[], optional_deps=[], data_paths=["~/ha-logs/intelligence/models/"], systemd_units=["aria-retrain.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["snapshot"]),
    Capability(id="meta_learning", name="LLM Meta-Learning", description="Ollama-based meta-learning to tune feature config", module="engine.llm", layer="engine", config_keys=[], test_paths=["tests/engine/test_llm.py", "tests/integration/test_meta_learning.py"], test_markers=["meta_learn"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-meta-learn.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["model_training"]),
    Capability(id="drift_detection", name="Concept Drift Detection", description="Page-Hinkley + fixed-window drift detection on prediction accuracy", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_drift.py"], test_markers=["drift"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-check-drift.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["scoring"]),
    Capability(id="correlations", name="Entity Correlations", description="Entity co-occurrence correlation analysis from logbook", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_entity_correlations.py"], test_markers=["correlations"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-correlations.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=[]),
    Capability(id="sequence_training", name="Sequence Model Training", description="Train Markov chain model from logbook event sequences", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_sequence_anomalies.py"], test_markers=["sequences"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-sequences-train.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=[]),
    Capability(id="sequence_anomalies", name="Sequence Anomaly Detection", description="Detect anomalous event sequences using Markov chain model", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_sequence_anomalies.py"], test_markers=["sequences"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-sequences-detect.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["sequence_training"]),
    Capability(id="automation_suggestions", name="LLM Automation Suggestions", description="Generate HA automation YAML via Ollama LLM", module="engine.llm", layer="engine", config_keys=[], test_paths=["tests/engine/test_automation_suggestions.py"], test_markers=["automations"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-suggest-automations.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["correlations"]),
    Capability(id="prophet_forecasting", name="Prophet Seasonal Forecasting", description="Train Prophet models for power, lights, devices, availability", module="engine.models", layer="engine", config_keys=[], test_paths=["tests/engine/test_prophet.py"], test_markers=["prophet"], runtime_deps=[], optional_deps=["prophet"], data_paths=[], systemd_units=["aria-prophet.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["snapshot"]),
    Capability(id="occupancy_estimation", name="Bayesian Occupancy Estimation", description="Occupancy estimation from sensor fusion", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_occupancy.py"], test_markers=["occupancy"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-occupancy.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["snapshot"]),
    Capability(id="power_profiling", name="Appliance Power Profiling", description="Analyze per-outlet power consumption patterns", module="engine.analysis", layer="engine", config_keys=[], test_paths=["tests/engine/test_power_profiles.py"], test_markers=["power"], runtime_deps=[], optional_deps=[], data_paths=[], systemd_units=["aria-power-profiles.timer"], pipeline_stage=None, status="stable", added_version="1.0.0", depends_on=["snapshot"]),
]
```

**Step 6: Add collect_from_modules to CapabilityRegistry**

Add to `aria/capabilities.py`:

```python
    def collect_from_modules(self):
        """Discover and register all capabilities from hub modules and engine."""
        # Hub modules
        from aria.modules.discovery import DiscoveryModule
        from aria.modules.ml_engine import MLEngine
        from aria.modules.patterns import PatternRecognition
        from aria.modules.orchestrator import OrchestratorModule
        from aria.modules.shadow_engine import ShadowEngine
        from aria.modules.data_quality import DataQualityModule
        from aria.modules.organic_discovery.module import OrganicDiscoveryModule
        from aria.modules.intelligence import IntelligenceModule
        from aria.modules.activity_monitor import ActivityMonitor

        hub_modules = [
            DiscoveryModule, MLEngine, PatternRecognition, OrchestratorModule,
            ShadowEngine, DataQualityModule, OrganicDiscoveryModule,
            IntelligenceModule, ActivityMonitor,
        ]

        for module_cls in hub_modules:
            for cap in getattr(module_cls, "CAPABILITIES", []):
                self.register(cap)

        # Engine capabilities
        from aria.engine.capabilities import ENGINE_CAPABILITIES
        for cap in ENGINE_CAPABILITIES:
            self.register(cap)
```

**Step 7: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v --timeout=30`
Expected: All 16 tests PASS

**Step 8: Commit**

```bash
git add aria/capabilities.py aria/engine/capabilities.py aria/hub/core.py aria/modules/discovery.py aria/modules/ml_engine.py aria/modules/patterns.py aria/modules/orchestrator.py aria/modules/shadow_engine.py aria/modules/data_quality.py aria/modules/organic_discovery/module.py aria/modules/intelligence.py aria/modules/activity_monitor.py tests/test_capabilities.py
git commit -m "feat: add capability declarations to all 9 hub modules + 13 engine commands"
```

---

## Task 4: CLI Commands (`aria capabilities list/verify/health`)

**Files:**
- Modify: `aria/cli.py` (add capabilities subcommand group)
- Test: `tests/test_capabilities.py` (append CLI tests)

**Step 1: Write the failing test**

Append to `tests/test_capabilities.py`:

```python
import subprocess
import sys

class TestCapabilitiesCLI:
    def test_list_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "discovery" in result.stdout
        assert "shadow_predictions" in result.stdout

    def test_list_layer_filter(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list", "--layer", "engine"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "snapshot" in result.stdout
        # Hub caps should not appear
        assert "shadow_predictions" not in result.stdout

    def test_verify_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "verify"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "issues" in result.stdout.lower() or "pass" in result.stdout.lower()

    def test_export_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "export"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert "capabilities" in data
        assert len(data["capabilities"]) >= 22
```

**Step 2: Run test to verify failures**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py::TestCapabilitiesCLI -v --timeout=60`
Expected: FAIL — `capabilities` not a recognized command

**Step 3: Add capabilities subcommand to `aria/cli.py`**

In `aria/cli.py`, add to `main()` after the existing subparsers:

```python
    # Capabilities subcommand group
    cap_parser = subparsers.add_parser("capabilities", help="Capability registry management")
    cap_sub = cap_parser.add_subparsers(dest="cap_command")
    cap_list = cap_sub.add_parser("list", help="List all registered capabilities")
    cap_list.add_argument("--layer", choices=["hub", "engine", "dashboard", "cross-cutting"], help="Filter by layer")
    cap_list.add_argument("--status", choices=["stable", "experimental", "planned"], help="Filter by status")
    cap_list.add_argument("--verbose", action="store_true", help="Show full details")
    cap_sub.add_parser("verify", help="Validate capability declarations")
    cap_verify = cap_sub.add_parser("verify", help="Validate capability declarations")
    cap_verify.add_argument("--strict", action="store_true", help="Fail on any issue (for CI)")
    cap_sub.add_parser("export", help="Export capabilities as JSON")
```

Add to `_dispatch()`:

```python
    elif args.command == "capabilities":
        _capabilities(args)
```

Add the handler function:

```python
def _capabilities(args):
    """Handle capabilities subcommands."""
    import json
    from aria.capabilities import CapabilityRegistry

    registry = CapabilityRegistry()
    registry.collect_from_modules()

    if args.cap_command == "list":
        caps = registry.list_all()
        if args.layer:
            caps = [c for c in caps if c.layer == args.layer]
        if hasattr(args, "status") and args.status:
            caps = [c for c in caps if c.status == args.status]

        # Group by layer
        from itertools import groupby
        caps_sorted = sorted(caps, key=lambda c: c.layer)
        for layer, group in groupby(caps_sorted, key=lambda c: c.layer):
            group_list = list(group)
            print(f"\n{layer.title()} Layer ({len(group_list)})")
            print(f"  {'ID':<28} {'Name':<35} {'Status':<14} {'Tests':<6} {'Config'}")
            print(f"  {'─'*28} {'─'*35} {'─'*14} {'─'*6} {'─'*8}")
            for c in group_list:
                test_count = len(c.test_paths)
                config_count = len(c.config_keys)
                print(f"  {c.id:<28} {c.name:<35} {c.status:<14} {test_count:<6} {config_count}")

        total = len(caps)
        print(f"\nTotal: {total} capabilities")

    elif args.cap_command == "verify":
        issues = registry.validate_all()
        if issues:
            print(f"Found {len(issues)} issue(s):\n")
            for issue in issues:
                print(f"  ✗ {issue}")
            if hasattr(args, "strict") and args.strict:
                sys.exit(1)
        else:
            print(f"All {len(registry.list_all())} capabilities pass validation.")

    elif args.cap_command == "export":
        from dataclasses import asdict
        caps = registry.list_all()
        data = {
            "capabilities": [asdict(c) for c in caps],
            "total": len(caps),
            "by_layer": {
                layer: len(registry.list_by_layer(layer))
                for layer in ["hub", "engine", "dashboard", "cross-cutting"]
            },
            "by_status": {
                status: len(registry.list_by_status(status))
                for status in ["stable", "experimental", "planned"]
            },
        }
        print(json.dumps(data, indent=2))

    else:
        print("Usage: aria capabilities {list|verify|export}")
        sys.exit(1)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py -v --timeout=60`
Expected: All 20 tests PASS

**Step 5: Commit**

```bash
git add aria/cli.py tests/test_capabilities.py
git commit -m "feat: add 'aria capabilities' CLI commands (list, verify, export)"
```

---

## Task 5: API Endpoints (`/api/capabilities`)

**Files:**
- Modify: `aria/hub/api.py` (add capability routes)
- Test: `tests/hub/test_api_capabilities.py` (new)

**Step 1: Write the failing test**

Create `tests/hub/test_api_capabilities.py`:

```python
"""Tests for /api/capabilities endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

from aria.hub.core import IntelligenceHub
from aria.hub.api import create_api


@pytest.fixture
def api_hub():
    mock_hub = MagicMock(spec=IntelligenceHub)
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.module_status = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub.get_uptime_seconds = MagicMock(return_value=0)
    return mock_hub


@pytest.fixture
def client(api_hub):
    app = create_api(api_hub)
    return TestClient(app)


class TestCapabilitiesAPI:
    def test_list_capabilities(self, client):
        resp = client.get("/api/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data
        assert "total" in data
        assert data["total"] >= 22

    def test_get_single_capability(self, client):
        resp = client.get("/api/capabilities/discovery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "discovery"
        assert data["layer"] == "hub"

    def test_get_nonexistent_capability(self, client):
        resp = client.get("/api/capabilities/nonexistent")
        assert resp.status_code == 404

    def test_capabilities_graph(self, client):
        resp = client.get("/api/capabilities/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
```

**Step 2: Run test to verify failures**

Run: `.venv/bin/python -m pytest tests/hub/test_api_capabilities.py -v --timeout=30`
Expected: FAIL — 404 for `/api/capabilities`

**Step 3: Add API routes to `aria/hub/api.py`**

Inside `create_api()`, add after existing router routes:

```python
    # --- Capabilities routes ---
    @router.get("/capabilities")
    async def list_capabilities(layer: Optional[str] = None, status: Optional[str] = None):
        from aria.capabilities import CapabilityRegistry
        from dataclasses import asdict
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        caps = registry.list_all()
        if layer:
            caps = [c for c in caps if c.layer == layer]
        if status:
            caps = [c for c in caps if c.status == status]
        return {"capabilities": [asdict(c) for c in caps], "total": len(caps)}

    @router.get("/capabilities/graph")
    async def capabilities_graph():
        from aria.capabilities import CapabilityRegistry
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        graph = registry.dependency_graph()
        nodes = [{"id": cap.id, "name": cap.name, "layer": cap.layer, "status": cap.status}
                 for cap in registry.list_all()]
        edges = []
        for cap_id, deps in graph.items():
            for dep_id in deps:
                edges.append({"from": dep_id, "to": cap_id})
        return {"nodes": nodes, "edges": edges}

    @router.get("/capabilities/{capability_id}")
    async def get_capability(capability_id: str):
        from aria.capabilities import CapabilityRegistry
        from dataclasses import asdict
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        cap = registry.get(capability_id)
        if not cap:
            raise HTTPException(status_code=404, detail=f"Capability '{capability_id}' not found")
        return asdict(cap)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_api_capabilities.py -v --timeout=30`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_capabilities.py
git commit -m "feat: add /api/capabilities REST endpoints"
```

---

## Task 6: Runtime Health Check

**Files:**
- Modify: `aria/capabilities.py` (add `health()` method)
- Modify: `aria/hub/api.py` (add `/api/capabilities/health` endpoint)
- Test: `tests/test_capabilities.py` (append)
- Test: `tests/hub/test_api_capabilities.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_capabilities.py`:

```python
class TestRuntimeHealth:
    def test_health_with_running_modules(self):
        """Health check should report module status from hub."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        # Simulate hub module status
        module_status = {
            "discovery": "running",
            "ml_engine": "running",
            "shadow_engine": "failed",
        }
        health = registry.health(module_status)
        assert health["discovery"]["module_loaded"] is True
        assert health["shadow_predictions"]["module_loaded"] is False

    def test_health_without_hub(self):
        """Health check without hub status should show all as unknown."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        health = registry.health({})
        for cap_id, status in health.items():
            assert status["module_loaded"] is None  # Unknown
```

**Step 2: Run test, verify fail, implement**

Add to `CapabilityRegistry`:

```python
    def health(self, module_status: dict[str, str]) -> dict[str, dict]:
        """Check runtime health of each capability against hub module status.

        Args:
            module_status: {module_id: "running"|"failed"|"registered"} from hub

        Returns:
            {capability_id: {"module_loaded": bool|None, "module_status": str}}
        """
        result = {}
        for cap in self._capabilities.values():
            # Map capability module name to hub module_id
            # Hub modules: module field matches hub registration id
            # Engine capabilities: always None (batch, not runtime)
            if cap.layer == "engine":
                result[cap.id] = {"module_loaded": None, "module_status": "batch"}
            elif cap.module in module_status:
                running = module_status[cap.module] == "running"
                result[cap.id] = {
                    "module_loaded": running,
                    "module_status": module_status[cap.module],
                }
            else:
                result[cap.id] = {"module_loaded": None, "module_status": "unknown"}
        return result
```

Add `/api/capabilities/health` endpoint in `api.py` (inside `create_api`, BEFORE the `/{capability_id}` route to avoid path conflict):

```python
    @router.get("/capabilities/health")
    async def capabilities_health():
        from aria.capabilities import CapabilityRegistry
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        return registry.health(dict(hub.module_status))
```

**Step 3: Run tests, verify pass, commit**

Run: `.venv/bin/python -m pytest tests/test_capabilities.py tests/hub/test_api_capabilities.py -v --timeout=30`

```bash
git add aria/capabilities.py aria/hub/api.py tests/test_capabilities.py tests/hub/test_api_capabilities.py
git commit -m "feat: add capability runtime health checks"
```

---

## Task 7: Dashboard Capabilities Page

**Files:**
- Create: `aria/dashboard/spa/src/pages/Capabilities.jsx`
- Modify: `aria/dashboard/spa/src/app.jsx` (add route)
- Modify: `aria/dashboard/spa/src/components/Nav.jsx` (add nav link)

**Step 1: Check existing dashboard patterns**

Read: `aria/dashboard/spa/src/app.jsx` and one existing page for patterns.

**Step 2: Create the Capabilities page**

Create `aria/dashboard/spa/src/pages/Capabilities.jsx` following the existing page patterns (use `fetch('/api/capabilities')`, `.t-frame` cards, status badges). The page should show:
- Capability count summary by layer
- Table with id, name, status, test count, config key count
- Dependency graph (simple list view — full graph visualization is a future enhancement)
- Health status badges when hub is running

**Step 3: Add route and nav link**

Add `<Route path="/capabilities" component={Capabilities} />` to `app.jsx`.
Add nav link to `Nav.jsx`.

**Step 4: Build the SPA**

Run: `cd aria/dashboard/spa && npm run build`

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/pages/Capabilities.jsx aria/dashboard/spa/src/app.jsx aria/dashboard/spa/src/components/Nav.jsx
git commit -m "feat: add Capabilities dashboard page"
```

---

## Task 8: Integration Test + Final Validation

**Files:**
- Create: `tests/integration/test_capabilities_integration.py`

**Step 1: Write integration test**

```python
"""Integration test — verify the full capability registry against live codebase."""

from aria.capabilities import CapabilityRegistry


class TestCapabilityRegistryIntegration:
    def test_full_collection_and_validation(self):
        """Collect all capabilities and run full validation suite."""
        registry = CapabilityRegistry()
        registry.collect_from_modules()

        # Should find at least 22 stable capabilities
        stable = registry.list_by_status("stable")
        assert len(stable) >= 22, f"Only {len(stable)} stable capabilities"

        # Should have all 9 hub modules represented
        hub_modules = {c.module for c in registry.list_by_layer("hub")}
        assert len(hub_modules) >= 9, f"Only {len(hub_modules)} hub modules"

        # Should have at least 10 engine capabilities
        engine_caps = registry.list_by_layer("engine")
        assert len(engine_caps) >= 10

        # Full validation should pass
        issues = registry.validate_all()
        assert issues == [], f"Validation issues:\n" + "\n".join(issues)

        # Dependency graph should have no cycles
        dep_issues = registry.validate_deps()
        assert dep_issues == [], f"Dep issues:\n" + "\n".join(dep_issues)

    def test_every_hub_module_has_capability(self):
        """No hub module should be missing a capability declaration."""
        from aria.modules.discovery import DiscoveryModule
        from aria.modules.ml_engine import MLEngine
        from aria.modules.patterns import PatternRecognition
        from aria.modules.orchestrator import OrchestratorModule
        from aria.modules.shadow_engine import ShadowEngine
        from aria.modules.data_quality import DataQualityModule
        from aria.modules.organic_discovery.module import OrganicDiscoveryModule
        from aria.modules.intelligence import IntelligenceModule
        from aria.modules.activity_monitor import ActivityMonitor

        for module_cls in [
            DiscoveryModule, MLEngine, PatternRecognition, OrchestratorModule,
            ShadowEngine, DataQualityModule, OrganicDiscoveryModule,
            IntelligenceModule, ActivityMonitor,
        ]:
            caps = getattr(module_cls, "CAPABILITIES", [])
            assert len(caps) > 0, f"{module_cls.__name__} has no CAPABILITIES declared"
```

**Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/integration/test_capabilities_integration.py -v --timeout=60`
Expected: All PASS

**Step 3: Run full test suite to verify no regressions**

Run: `.venv/bin/python -m pytest tests/ -v --timeout=120 -x`
Expected: ~1000+ tests PASS (984 existing + ~25 new)

**Step 4: Commit**

```bash
git add tests/integration/test_capabilities_integration.py
git commit -m "test: add capability registry integration tests"
```

---

## Task 9: Final Cleanup + Docs Update

**Files:**
- Modify: `CLAUDE.md` (update capability count, add capabilities CLI to command table)

**Step 1: Update CLAUDE.md**

Add `aria capabilities list/verify/export/health` to the CLI commands table.
Update the "Hub Modules (9)" section to note they now declare capabilities.

**Step 2: Run verify one final time**

Run: `cd ~/Documents/projects/ha-aria && .venv/bin/python -m aria.cli capabilities verify`
Expected: "All N capabilities pass validation."

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with capability registry commands"
```

---

## Summary

| Task | What | New Tests | Files Touched |
|------|------|-----------|---------------|
| 1 | Capability dataclass + Registry core | 7 | 2 |
| 2 | Validation engine | 5 | 2 |
| 3 | Module declarations (9 hub + 13 engine) | 4 | 12 |
| 4 | CLI commands | 4 | 2 |
| 5 | API endpoints | 4 | 2 |
| 6 | Runtime health | 2+ | 4 |
| 7 | Dashboard page | 0 (visual) | 3 |
| 8 | Integration test | 2 | 1 |
| 9 | Docs cleanup | 0 | 1 |
| **Total** | | **~28** | **~29** |
