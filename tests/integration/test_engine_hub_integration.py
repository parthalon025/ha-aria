"""Integration tests: verify engine and hub can interoperate within the aria namespace."""


def test_engine_imports_accessible_from_hub():
    """Verify hub code can import engine modules."""
    from aria.engine.analysis.entity_correlations import summarize_entity_correlations
    from aria.engine.analysis.sequence_anomalies import MarkovChainDetector
    from aria.engine.config import AppConfig
    from aria.engine.storage.data_store import DataStore

    assert AppConfig is not None
    assert DataStore is not None
    assert summarize_entity_correlations is not None
    assert MarkovChainDetector is not None


def test_hub_imports_accessible():
    """Verify hub core can be imported."""
    from aria.hub.cache import CacheManager
    from aria.hub.constants import CACHE_INTELLIGENCE
    from aria.hub.core import IntelligenceHub, Module

    assert IntelligenceHub is not None
    assert Module is not None
    assert CacheManager is not None
    assert isinstance(CACHE_INTELLIGENCE, str)


def test_module_imports_accessible():
    """Verify all hub modules can be imported."""
    from aria.modules.activity_monitor import ActivityMonitor
    from aria.modules.discovery import DiscoveryModule
    from aria.modules.intelligence import IntelligenceModule
    from aria.modules.ml_engine import MLEngine
    from aria.modules.orchestrator import OrchestratorModule
    from aria.modules.patterns import PatternRecognition
    from aria.modules.shadow_engine import ShadowEngine

    assert IntelligenceModule is not None
    assert DiscoveryModule is not None
    assert ShadowEngine is not None
    assert ActivityMonitor is not None
    assert MLEngine is not None
    assert PatternRecognition is not None
    assert OrchestratorModule is not None


def test_engine_and_hub_share_namespace():
    """Verify engine and hub live under the same aria package."""
    import aria

    assert hasattr(aria, "__version__")

    import aria.engine
    import aria.hub
    import aria.modules

    # Both are subpackages of the same top-level
    assert aria.engine.__name__.startswith("aria.")
    assert aria.hub.__name__.startswith("aria.")
    assert aria.modules.__name__.startswith("aria.")


def test_cli_entry_point_importable():
    """Verify the CLI entry point can be imported."""
    from aria.cli import main

    assert callable(main)
