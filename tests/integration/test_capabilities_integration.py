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

    def test_cli_verify_passes(self):
        """The CLI verify command should exit 0 with no issues."""
        import subprocess
        import sys
        from pathlib import Path

        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "verify"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0, f"verify failed: {result.stderr}"

    def test_cli_export_matches_registry(self):
        """CLI export JSON should match registry count."""
        import subprocess
        import sys
        import json
        from pathlib import Path

        result = subprocess.run(
            [sys.executable, "-m", "aria.cli", "capabilities", "export"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(__file__).parent.parent.parent),
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)

        registry = CapabilityRegistry()
        registry.collect_from_modules()
        assert data["total"] == len(registry.list_all())
