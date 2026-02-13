"""Model serialization — pickle save/load for sklearn models."""

import os
import pickle
from pathlib import Path


class ModelIO:
    """Pickle-based model persistence."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def save_model(self, model, name: str, metadata: dict | None = None) -> Path:
        """Save a trained model to disk."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata or {}}, f)
        return path

    def load_model(self, name: str):
        """Load a saved model. Returns (model, metadata) or (None, None) if not found."""
        path = self.models_dir / f"{name}.pkl"
        if not path.is_file():
            return None, None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "model" in data:
                return data["model"], data.get("metadata", {})
            # Legacy format: just the model object
            return data, {}
        except Exception:
            return None, None

    def list_models(self) -> list[str]:
        """List available model names."""
        if not self.models_dir.is_dir():
            return []
        return [f.stem for f in sorted(self.models_dir.glob("*.pkl"))]

    def model_exists(self, name: str) -> bool:
        """Check if a model file exists."""
        return (self.models_dir / f"{name}.pkl").is_file()
