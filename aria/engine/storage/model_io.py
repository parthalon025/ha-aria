"""Model serialization — pickle save/load for sklearn models."""

import pickle
import shutil
from datetime import datetime
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

    def save_model_versioned(self, model, name: str, metadata: dict | None = None,
                             keep: int = 3) -> Path:
        """Save model with date-based versioning.

        Creates a timestamped copy alongside a _latest pointer.
        Prunes old versions, keeping the most recent `keep` copies.

        Not yet wired into production model saving — stub for future use.

        Args:
            model: Trained model object (picklable).
            name: Base model name (e.g. "power_watts").
            metadata: Optional metadata dict stored alongside model.
            keep: Number of old versioned copies to retain (default 3).

        Returns:
            Path to the versioned model file.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = self.models_dir / f"{name}_{timestamp}.pkl"
        latest_path = self.models_dir / f"{name}_latest.pkl"

        payload = {"model": model, "metadata": metadata or {}}

        # Save versioned copy
        with open(versioned_path, "wb") as f:
            pickle.dump(payload, f)

        # Update latest pointer (copy is atomic-enough on Linux ext4/btrfs)
        shutil.copy2(versioned_path, latest_path)

        # Prune old versions (keep last N timestamped copies)
        versions = sorted(self.models_dir.glob(f"{name}_2*.pkl"))
        for old in versions[:-keep]:
            old.unlink()

        return versioned_path
