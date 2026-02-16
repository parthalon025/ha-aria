"""Unit tests for ModelIO — pickle-based model persistence.

Tests save/load round-trips, missing files, corrupt data,
directory creation, listing, and existence checks.
"""

import pickle

import pytest

from aria.engine.storage.model_io import ModelIO

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_io(tmp_path):
    """Create a ModelIO pointing at a temp directory."""
    return ModelIO(tmp_path / "models")


@pytest.fixture
def model_io_with_dir(tmp_path):
    """Create a ModelIO with the models directory pre-existing."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return ModelIO(models_dir)


# ============================================================================
# Save Model
# ============================================================================


class TestSaveModel:
    """Test save_model writes pickle files correctly."""

    def test_save_creates_file(self, model_io):
        """save_model creates a .pkl file."""
        path = model_io.save_model({"type": "dummy"}, "test_model")
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_save_creates_directory(self, model_io):
        """save_model creates models directory if it doesn't exist."""
        assert not model_io.models_dir.exists()
        model_io.save_model({"type": "dummy"}, "test_model")
        assert model_io.models_dir.is_dir()

    def test_save_returns_correct_path(self, model_io):
        """Returned path matches expected name."""
        path = model_io.save_model("model", "my_model")
        assert path == model_io.models_dir / "my_model.pkl"

    def test_save_with_metadata(self, model_io):
        """Metadata is stored alongside the model."""
        model_io.save_model("model", "test", metadata={"version": "1.0"})
        path = model_io.models_dir / "test.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert data["metadata"]["version"] == "1.0"

    def test_save_without_metadata(self, model_io):
        """Without metadata, empty dict is stored."""
        model_io.save_model("model", "test")
        path = model_io.models_dir / "test.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        assert data["metadata"] == {}

    def test_save_overwrites_existing(self, model_io):
        """Saving with same name overwrites the existing file."""
        model_io.save_model("v1", "test")
        model_io.save_model("v2", "test")

        model, meta = model_io.load_model("test")
        assert model == "v2"


# ============================================================================
# Load Model
# ============================================================================


class TestLoadModel:
    """Test load_model reads pickle files correctly."""

    def test_load_round_trip(self, model_io):
        """Save then load returns the same model."""
        original = {"weights": [1, 2, 3], "bias": 0.5}
        model_io.save_model(original, "my_model")

        model, metadata = model_io.load_model("my_model")
        assert model == original
        assert metadata == {}

    def test_load_with_metadata_round_trip(self, model_io):
        """Save with metadata then load returns both."""
        original = [1, 2, 3]
        meta = {"trained_at": "2026-02-13", "accuracy": 0.95}
        model_io.save_model(original, "scored_model", metadata=meta)

        model, metadata = model_io.load_model("scored_model")
        assert model == original
        assert metadata["accuracy"] == 0.95

    def test_load_missing_file(self, model_io):
        """Loading a nonexistent model returns (None, None)."""
        model, metadata = model_io.load_model("nonexistent")
        assert model is None
        assert metadata is None

    def test_load_corrupt_file(self, model_io):
        """Loading a corrupt pickle file returns (None, None)."""
        model_io.models_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = model_io.models_dir / "corrupt.pkl"
        corrupt_path.write_bytes(b"this is not a valid pickle file")

        model, metadata = model_io.load_model("corrupt")
        assert model is None
        assert metadata is None

    def test_load_legacy_format(self, model_io):
        """Loading a pickle that's just a raw object (no dict wrapper) works."""
        model_io.models_dir.mkdir(parents=True, exist_ok=True)
        path = model_io.models_dir / "legacy.pkl"
        with open(path, "wb") as f:
            pickle.dump("raw_model_object", f)

        model, metadata = model_io.load_model("legacy")
        assert model == "raw_model_object"
        assert metadata == {}

    def test_load_complex_model(self, model_io):
        """Round-trip with a complex nested object."""
        original = {
            "layers": [[0.1, 0.2], [0.3, 0.4]],
            "config": {"learning_rate": 0.01, "epochs": 100},
        }
        model_io.save_model(original, "complex")

        model, _ = model_io.load_model("complex")
        assert model["layers"][1] == [0.3, 0.4]
        assert model["config"]["epochs"] == 100


# ============================================================================
# List Models
# ============================================================================


class TestListModels:
    """Test list_models enumerates available models."""

    def test_list_empty_directory(self, model_io_with_dir):
        """Empty models directory returns empty list."""
        assert model_io_with_dir.list_models() == []

    def test_list_nonexistent_directory(self, tmp_path):
        """Nonexistent directory returns empty list."""
        io = ModelIO(tmp_path / "does_not_exist")
        assert io.list_models() == []

    def test_list_multiple_models(self, model_io):
        """Lists all saved model names (without .pkl extension)."""
        model_io.save_model("a", "alpha")
        model_io.save_model("b", "beta")
        model_io.save_model("c", "gamma")

        names = model_io.list_models()
        assert sorted(names) == ["alpha", "beta", "gamma"]

    def test_list_ignores_non_pkl_files(self, model_io_with_dir):
        """Only .pkl files are listed."""
        (model_io_with_dir.models_dir / "readme.txt").write_text("not a model")
        model_io_with_dir.save_model("m", "real_model")

        names = model_io_with_dir.list_models()
        assert names == ["real_model"]

    def test_list_sorted(self, model_io):
        """Model names are returned sorted."""
        model_io.save_model("z", "zebra")
        model_io.save_model("a", "alpha")

        names = model_io.list_models()
        assert names == ["alpha", "zebra"]


# ============================================================================
# Model Exists
# ============================================================================


class TestModelExists:
    """Test model_exists checks."""

    def test_exists_true(self, model_io):
        """Returns True for saved model."""
        model_io.save_model("m", "test_model")
        assert model_io.model_exists("test_model") is True

    def test_exists_false(self, model_io):
        """Returns False for nonexistent model."""
        assert model_io.model_exists("nonexistent") is False

    def test_exists_false_nonexistent_dir(self, tmp_path):
        """Returns False when models directory doesn't exist."""
        io = ModelIO(tmp_path / "no_dir")
        assert io.model_exists("anything") is False
