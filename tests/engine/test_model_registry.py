import pytest

from aria.engine.models.registry import ModelEntry, TieredModelRegistry


class TestTieredModelRegistry:
    def test_register_and_resolve_tier_2(self):
        registry = TieredModelRegistry()
        entry = ModelEntry(
            name="gb_power",
            tier=2,
            model_factory=lambda: "mock_gb",
            params={"n_estimators": 100},
            weight=0.40,
            requires=[],
        )
        registry.register("power_watts", entry)
        resolved = registry.resolve("power_watts", current_tier=2)
        assert len(resolved) == 1
        assert resolved[0].name == "gb_power"

    def test_resolve_excludes_higher_tiers(self):
        registry = TieredModelRegistry()
        registry.register(
            "power_watts",
            ModelEntry(name="lgbm_power", tier=2, model_factory=lambda: None, params={}, weight=0.40, requires=[]),
        )
        registry.register(
            "power_watts",
            ModelEntry(
                name="transformer_power",
                tier=4,
                model_factory=lambda: None,
                params={},
                weight=0.30,
                requires=["torch"],
            ),
        )
        resolved = registry.resolve("power_watts", current_tier=2)
        assert len(resolved) == 1
        assert resolved[0].name == "lgbm_power"

    def test_resolve_includes_lower_tiers(self):
        registry = TieredModelRegistry()
        registry.register(
            "power_watts",
            ModelEntry(name="lgbm_simple", tier=1, model_factory=lambda: None, params={}, weight=0.50, requires=[]),
        )
        registry.register(
            "power_watts",
            ModelEntry(name="lgbm_full", tier=2, model_factory=lambda: None, params={}, weight=0.40, requires=[]),
        )
        resolved = registry.resolve("power_watts", current_tier=3)
        assert len(resolved) == 2

    def test_resolve_skips_missing_dependencies(self):
        registry = TieredModelRegistry()
        registry.register(
            "power_watts",
            ModelEntry(
                name="transformer",
                tier=3,
                model_factory=lambda: None,
                params={},
                weight=0.30,
                requires=["nonexistent_package_xyz"],
            ),
        )
        resolved = registry.resolve("power_watts", current_tier=3)
        assert len(resolved) == 0

    def test_weights_renormalize(self):
        registry = TieredModelRegistry()
        registry.register(
            "power_watts",
            ModelEntry(name="gb", tier=2, model_factory=lambda: None, params={}, weight=0.35, requires=[]),
        )
        registry.register(
            "power_watts",
            ModelEntry(name="rf", tier=2, model_factory=lambda: None, params={}, weight=0.25, requires=[]),
        )
        registry.resolve("power_watts", current_tier=2)
        normed = registry.get_normalized_weights("power_watts", current_tier=2)
        assert pytest.approx(sum(normed.values()), abs=0.001) == 1.0

    def test_default_stack_registers_current_models(self):
        """Default registry should contain gb, rf, lgbm at tier 2."""
        registry = TieredModelRegistry.with_defaults()
        for target in ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]:
            resolved = registry.resolve(target, current_tier=2)
            names = [e.name for e in resolved]
            assert "gb" in names or any("gb" in n for n in names)
