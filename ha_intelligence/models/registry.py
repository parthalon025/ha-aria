"""Model registry — decorator-based registration for ML models.

Adding a new model type:
    @ModelRegistry.register("my_model")
    class MyModel(BaseModel):
        def train(self, feature_names, X, y, model_dir, config):
            ...
        def predict(self, feature_vector):
            ...
"""

from abc import ABC, abstractmethod


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
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {list(cls._models.keys())}"
            )
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
