"""Data collectors — fetch HA states, extract domain data, build snapshots."""

# Import extractors to trigger @CollectorRegistry.register() decorators.
# Without this, the registry is empty and all snapshots produce zeros.
import aria.engine.collectors.extractors  # noqa: F401
from aria.engine.collectors.registry import BaseCollector, CollectorRegistry

__all__ = ["CollectorRegistry", "BaseCollector"]
