"""Model registry — decorator-based registration for ML models.

Adding a new model type:
    @ModelRegistry.register("my_model")
    class MyModel(BaseModel):
        def train(self, feature_names, X, y, model_dir, config):
            ...
        def predict(self, feature_vector):
            ...
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for ML model classes using decorator-based registration."""

    _models: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):
        """Decorator to register a model class."""

        def decorator(model_class):
            key = name or model_class.__name__.lower()
            cls._models[key] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str):
        """Get a model class by name."""
        if name not in cls._models:
            raise KeyError(f"Model '{name}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def all(cls) -> dict[str, type]:
        """Get all registered models."""
        return dict(cls._models)

    @classmethod
    def available(cls) -> list[str]:
        """List available model names."""
        return list(cls._models.keys())

    @classmethod
    def clear(cls):
        """Clear registry (for testing)."""
        cls._models.clear()


class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def train(self, **kwargs):
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, **kwargs):
        """Generate predictions."""
        ...


# --- Tier-aware model registry (Phase 1) ---


@dataclass
class ModelEntry:
    name: str
    tier: int
    model_factory: object  # callable or None — set at training time
    params: dict = field(default_factory=dict)
    weight: float = 1.0
    requires: list[str] = field(default_factory=list)
    fallback_tier: int | None = None


class TieredModelRegistry:
    """Tier-aware model registry. Each target has a stack of models at different tiers."""

    def __init__(self):
        self._stacks: dict[str, list[ModelEntry]] = {}

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
            registry.register(
                target,
                ModelEntry(
                    name="lgbm_lite",
                    tier=1,
                    weight=1.0,
                    model_factory=None,
                    params={"n_estimators": 50, "max_depth": 3, "verbosity": -1},
                    requires=["lightgbm"],
                ),
            )

            # Tier 2: full ensemble (current ARIA default)
            registry.register(
                target,
                ModelEntry(
                    name="gb",
                    tier=2,
                    weight=0.35,
                    model_factory=None,
                    params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
                    requires=[],
                ),
            )
            registry.register(
                target,
                ModelEntry(
                    name="rf",
                    tier=2,
                    weight=0.25,
                    model_factory=None,
                    params={"n_estimators": 100, "max_depth": 5},
                    requires=[],
                ),
            )
            registry.register(
                target,
                ModelEntry(
                    name="lgbm",
                    tier=2,
                    weight=0.40,
                    model_factory=None,
                    params={
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 4,
                        "num_leaves": 15,
                        "subsample": 0.8,
                        "verbosity": -1,
                        "importance_type": "gain",
                    },
                    requires=["lightgbm"],
                ),
            )

        return registry
