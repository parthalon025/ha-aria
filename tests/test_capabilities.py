"""Tests for aria.capabilities — Capability dataclass and registry."""

import pytest

from aria.capabilities import Capability, CapabilityRegistry


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
        """Two-node cycle: a→b→a."""
        reg = CapabilityRegistry()
        reg.register(_make_cap(id="a", depends_on=["b"]))
        reg.register(_make_cap(id="b", depends_on=["a"]))
        errors = reg.validate_deps()
        assert any("cycle" in e.lower() for e in errors)

    def test_validate_deps_three_node_cycle(self):
        """a→b→c→a."""
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
