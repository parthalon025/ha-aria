"""Collector registry — decorator-based registration for data extractors.

Adding a new device type:
    @CollectorRegistry.register("my_device")
    class MyDeviceCollector(BaseCollector):
        def extract(self, snapshot, states):
            # populate snapshot with device data
            ...
"""

from abc import ABC, abstractmethod


class CollectorRegistry:
    """Registry for data collectors using decorator-based registration."""
    _collectors: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """Decorator to register a collector class."""
        def decorator(collector_class):
            key = name or collector_class.__name__.lower().replace("collector", "")
            cls._collectors[key] = collector_class
            return collector_class
        return decorator

    @classmethod
    def get(cls, name: str):
        """Get a collector class by name."""
        if name not in cls._collectors:
            raise KeyError(
                f"Collector '{name}' not registered. "
                f"Available: {list(cls._collectors.keys())}"
            )
        return cls._collectors[name]

    @classmethod
    def all(cls) -> dict[str, type]:
        """Get all registered collectors."""
        return dict(cls._collectors)

    @classmethod
    def clear(cls):
        """Clear registry (for testing)."""
        cls._collectors.clear()


class BaseCollector(ABC):
    """Abstract base class for all data collectors."""

    @abstractmethod
    def extract(self, snapshot: dict, states: list[dict]) -> None:
        """Extract data from HA states and populate snapshot in-place.

        Args:
            snapshot: The snapshot dict to populate (mutated in-place).
            states: List of HA entity state dicts from the REST API.
        """
        ...
