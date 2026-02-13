"""Data collectors — fetch HA states, extract domain data, build snapshots."""

from ha_intelligence.collectors.registry import CollectorRegistry, BaseCollector

__all__ = ["CollectorRegistry", "BaseCollector"]
