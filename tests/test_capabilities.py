"""Tests for aria.capabilities — Capability dataclass, registry, and CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from aria.capabilities import Capability, CapabilityRegistry, DemandSignal

PROJECT_ROOT = str(Path(__file__).parent.parent)


# --- Capability dataclass ---


def _make_cap(**overrides):
    """Create a valid Capability with sensible defaults, overridable per-test."""
    defaults = dict(
        id="lighting",
        name="Lighting Control",
        description="Controls lights across zones",
        module="aria.modules.discovery",
        layer="hub",
        config_keys=["light_threshold"],
        test_paths=["tests/hub/test_discovery.py"],
        test_markers=["lighting"],
        runtime_deps=["aiohttp"],
        optional_deps=[],
        data_paths=["~/ha-logs/intelligence/"],
        systemd_units=["aria-hub.service"],
        pipeline_stage=None,
        status="stable",
        added_version="1.0.0",
        depends_on=[],
    )
    defaults.update(overrides)
    return Capability(**defaults)


class TestCapabilityCreation:
    """Valid Capability creation."""

    def test_basic_creation(self):
        cap = _make_cap()
        assert cap.id == "lighting"
        assert cap.name == "Lighting Control"
        assert cap.layer == "hub"
        assert cap.status == "stable"

    def test_frozen(self):
        cap = _make_cap()
        with pytest.raises(AttributeError):
            cap.id = "something_else"

    def test_all_layers_valid(self):
        for layer in ("hub", "engine", "dashboard", "cross-cutting"):
            cap = _make_cap(id=f"cap_{layer}", layer=layer)
            assert cap.layer == layer

    def test_all_statuses_valid(self):
        for status in ("stable", "experimental", "planned"):
            cap = _make_cap(id=f"cap_{status}", status=status)
            assert cap.status == status

    def test_all_pipeline_stages_valid(self):
        for stage in ("backtest", "shadow", "suggest", "autonomous", None):
            cap = _make_cap(id=f"cap_{stage}", pipeline_stage=stage)
            assert cap.pipeline_stage == stage

    def test_with_depends_on(self):
        cap = _make_cap(depends_on=["discovery", "ml_engine"])
        assert cap.depends_on == ["discovery", "ml_engine"]


class TestDemandSignal:
    """Tests for DemandSignal dataclass and its integration with Capability."""

    def test_demand_signal_defaults(self):
        sig = DemandSignal()
        assert sig.entity_domains == []
        assert sig.device_classes == []
        assert sig.min_entities == 5
        assert sig.description == ""

    def test_demand_signal_custom_fields(self):
        sig = DemandSignal(
            entity_domains=["light", "switch"],
            device_classes=["occupancy"],
            min_entities=10,
            description="Lighting groups for zone control",
        )
        assert sig.entity_domains == ["light", "switch"]
        assert sig.device_classes == ["occupancy"]
        assert sig.min_entities == 10
        assert sig.description == "Lighting groups for zone control"

    def test_demand_signal_frozen(self):
        sig = DemandSignal()
        with pytest.raises(AttributeError):
            sig.min_entities = 10

    def test_capability_with_demand_signals(self):
        signals = [
            DemandSignal(entity_domains=["light"], min_entities=10),
            DemandSignal(entity_domains=["climate"], device_classes=["thermostat"]),
        ]
        cap = _make_cap(demand_signals=signals)
        assert len(cap.demand_signals) == 2
        assert cap.demand_signals[0].entity_domains == ["light"]
        assert cap.demand_signals[1].device_classes == ["thermostat"]

    def test_capability_without_demand_signals_defaults_empty(self):
        cap = _make_cap()
        assert cap.demand_signals == []


class TestCapabilityValidation:
    """Reject invalid field values in __post_init__."""

    def test_invalid_layer_rejected(self):
        with pytest.raises(ValueError, match="layer"):
            _make_cap(layer="invalid_layer")

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError, match="status"):
            _make_cap(status="deprecated")

    def test_invalid_pipeline_stage_rejected(self):
        with pytest.raises(ValueError, match="pipeline_stage"):
            _make_cap(pipeline_stage="production")

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="id"):
            _make_cap(id="")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            _make_cap(name="")


# --- CapabilityRegistry ---


class TestRegistryBasics:
    """Register, get, and list capabilities."""

    def test_register_and_get(self):
        reg = CapabilityRegistry()
        cap = _make_cap()
        reg.register(cap)
        assert reg.get("lighting") is cap

    def test_get_missing_returns_none(self):
        reg = CapabilityRegistry()
        assert reg.get("nonexistent") is None

    def test_list_ids(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="a"))
        reg.register(_make_cap(id="b"))
        assert sorted(reg.list_ids()) == ["a", "b"]

    def test_list_all(self):
        reg = CapabilityRegistry()
        cap_a = _make_cap(id="a")
        cap_b = _make_cap(id="b")
        reg.register(cap_a)
        reg.register(cap_b)
        result = reg.list_all()
        assert len(result) == 2
        assert cap_a in result
        assert cap_b in result

    def test_duplicate_rejected(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="lighting"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_cap(id="lighting"))


class TestRegistryFiltering:
    """Filter by layer and status."""

    @pytest.fixture()
    def populated_registry(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="hub1", layer="hub", status="stable"))
        reg.register(_make_cap(id="hub2", layer="hub", status="experimental"))
        reg.register(_make_cap(id="eng1", layer="engine", status="stable"))
        reg.register(_make_cap(id="dash1", layer="dashboard", status="planned"))
        return reg

    def test_list_by_layer(self, populated_registry):
        hub_caps = populated_registry.list_by_layer("hub")
        assert {c.id for c in hub_caps} == {"hub1", "hub2"}

    def test_list_by_layer_empty(self, populated_registry):
        assert populated_registry.list_by_layer("cross-cutting") == []

    def test_list_by_status(self, populated_registry):
        stable = populated_registry.list_by_status("stable")
        assert {c.id for c in stable} == {"hub1", "eng1"}


class TestRegistryDependencyGraph:
    """dependency_graph() and validate_deps()."""

    def test_dependency_graph_structure(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="base", depends_on=[]))
        reg.register(_make_cap(id="mid", depends_on=["base"]))
        reg.register(_make_cap(id="top", depends_on=["mid"]))

        graph = reg.dependency_graph()
        assert graph == {
            "base": [],
            "mid": ["base"],
            "top": ["mid"],
        }

    def test_validate_deps_no_issues(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="base", depends_on=[]))
        reg.register(_make_cap(id="child", depends_on=["base"]))
        errors = reg.validate_deps()
        assert errors == []

    def test_validate_deps_missing_dependency(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="orphan", depends_on=["nonexistent"]))
        errors = reg.validate_deps()
        assert any("nonexistent" in e for e in errors)

    def test_validate_deps_cycle_detection(self):
        """Two-node cycle: a->b->a."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="a", depends_on=["b"]))
        reg.register(_make_cap(id="b", depends_on=["a"]))
        errors = reg.validate_deps()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_deps_three_node_cycle(self):
        """a->b->c->a."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="a", depends_on=["b"]))
        reg.register(_make_cap(id="b", depends_on=["c"]))
        reg.register(_make_cap(id="c", depends_on=["a"]))
        errors = reg.validate_deps()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_deps_self_reference(self):
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="self_ref", depends_on=["self_ref"]))
        errors = reg.validate_deps()
        assert any("cycle" in e.lower() for e in errors)


# --- Validation Engine (config keys, test paths, validate_all) ---


class TestValidateConfigKeys:
    """validate_config_keys() checks declared config_keys against CONFIG_DEFAULTS."""

    def test_validate_config_keys_missing(self):
        """Cap with a nonexistent config key should produce an issue."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="bad_cfg", config_keys=["totally.fake.key"]))
        issues = reg.validate_config_keys()
        assert len(issues) == 1
        assert "totally.fake.key" in issues[0]

    def test_validate_config_keys_present(self):
        """Cap with a real config key should produce no issue."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="good_cfg", config_keys=["shadow.min_confidence"]))
        issues = reg.validate_config_keys()
        assert issues == []


class TestValidateTestPaths:
    """validate_test_paths() checks declared test_paths exist on disk."""

    def test_validate_test_paths_missing(self):
        """Cap with a nonexistent test path should produce an issue."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="bad_path", test_paths=["tests/nonexistent_file.py"]))
        issues = reg.validate_test_paths()
        assert len(issues) == 1
        assert "tests/nonexistent_file.py" in issues[0]

    def test_validate_test_paths_present(self):
        """Cap with a real test path should produce no issue."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="good_path", test_paths=["tests/hub/test_shadow_engine.py"]))
        issues = reg.validate_test_paths()
        assert issues == []


class TestValidateAll:
    """validate_all() combines all validation checks."""

    def test_validate_all_combines_checks(self):
        """Cap with fake config key + fake test path + missing dep should produce >=3 issues."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(
            id="broken",
            config_keys=["fake.config.key"],
            test_paths=["tests/does_not_exist.py"],
            depends_on=["missing_dep"],
        ))
        issues = reg.validate_all()
        assert len(issues) >= 3
        assert any("fake.config.key" in i for i in issues)
        assert any("tests/does_not_exist.py" in i for i in issues)
        assert any("missing_dep" in i for i in issues)


# --- collect_from_modules ---


class TestCollectFromModules:
    """Tests for CapabilityRegistry.collect_from_modules()."""

    def test_collect_finds_hub_modules(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        hub_caps = registry.list_by_layer("hub")
        hub_module_ids = {c.module for c in hub_caps}
        expected = {
            "discovery", "ml_engine", "pattern_recognition", "orchestrator",
            "shadow_engine", "data_quality", "organic_discovery",
            "intelligence", "activity_monitor",
        }
        assert expected.issubset(hub_module_ids), f"Missing: {expected - hub_module_ids}"

    def test_collect_finds_engine_capabilities(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        engine_caps = registry.list_by_layer("engine")
        assert len(engine_caps) >= 10

    def test_all_collected_capabilities_validate(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        issues = registry.validate_all()
        assert issues == [], f"Validation issues:\n" + "\n".join(issues)

    def test_total_capability_count(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        stable = registry.list_by_status("stable")
        assert len(stable) >= 22


# --- CLI Commands ---


class TestCapabilitiesCLI:
    """Tests for the `aria capabilities` CLI subcommand."""

    def test_list_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "discovery" in result.stdout
        assert "shadow_predictions" in result.stdout

    def test_list_layer_filter(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list", "--layer", "engine"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "snapshot" in result.stdout
        assert "shadow_predictions" not in result.stdout

    def test_list_status_filter(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list", "--status", "stable"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "stable" in result.stdout

    def test_list_verbose(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "list", "--verbose"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert len(result.stdout) > 500

    def test_verify_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "verify"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_export_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "export"],
            capture_output=True, text=True, timeout=30,
            cwd=PROJECT_ROOT,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "capabilities" in data
        assert data["total"] >= 22
        assert "by_layer" in data
        assert "by_status" in data


class TestRuntimeHealth:
    """health() maps module status to capability health."""

    def test_health_with_running_modules(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        module_status = {
            "discovery": "running",
            "ml_engine": "running",
            "shadow_engine": "failed",
        }
        health = registry.health(module_status)
        assert health["discovery"]["module_loaded"] is True
        assert health["ml_realtime"]["module_loaded"] is True
        assert health["shadow_predictions"]["module_loaded"] is False
        assert health["shadow_predictions"]["module_status"] == "failed"

    def test_health_engine_capabilities_are_batch(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        health = registry.health({})
        assert health["snapshot"]["module_loaded"] is None
        assert health["snapshot"]["module_status"] == "batch"

    def test_health_unknown_modules(self):
        registry = CapabilityRegistry()
        registry.collect_from_modules()
        health = registry.health({})
        assert health["discovery"]["module_loaded"] is None
        assert health["discovery"]["module_status"] == "unknown"


class TestDemandSignalDeclarations:
    """Verify ML engine and shadow engine declare demand signals."""

    def test_ml_engine_declares_demand_signals(self):
        from aria.modules.ml_engine import MLEngine
        caps = MLEngine.CAPABILITIES
        assert any(len(c.demand_signals) > 0 for c in caps)
        # Check the first demand signal has required fields
        ds = caps[0].demand_signals[0]
        assert ds.entity_domains
        assert ds.min_entities >= 1

    def test_shadow_engine_declares_demand_signals(self):
        from aria.modules.shadow_engine import ShadowEngine
        caps = ShadowEngine.CAPABILITIES
        assert any(len(c.demand_signals) > 0 for c in caps)

    def test_demand_signals_are_frozen(self):
        ds = DemandSignal(entity_domains=["sensor"], min_entities=5)
        with pytest.raises(Exception):  # FrozenInstanceError
            ds.min_entities = 10
