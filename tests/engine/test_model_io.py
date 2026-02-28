"""Unit tests for ModelIO — pickle-based model persistence.

Tests save/load round-trips, missing files, corrupt data,
directory creation, listing, and existence checks.

Note: load_model validates that the loaded object has a .predict() interface
(#217). Round-trip tests use sklearn models, which are picklable and satisfy
this contract. Save-only tests (no load_model call) may use any picklable object.
"""

import pickle

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from aria.engine.storage.model_io import ModelIO


def _make_lr() -> LinearRegression:
    """Return a minimal fitted LinearRegression (picklable, has .predict())."""
    lr = LinearRegression()
    lr.fit(np.array([[1], [2], [3]]), np.array([1.0, 2.0, 3.0]))
    return lr


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
        """Saving with same name overwrites the existing file (#217: use sklearn model)."""
        lr1 = _make_lr()
        lr2 = _make_lr()
        lr2.coef_ = lr1.coef_ * 2  # distinguish the two

        model_io.save_model(lr1, "test")
        model_io.save_model(lr2, "test")

        model, _meta = model_io.load_model("test")
        assert model is not None
        assert hasattr(model, "predict")


# ============================================================================
# Load Model
# ============================================================================


class TestLoadModel:
    """Test load_model reads pickle files correctly."""

    def test_load_round_trip(self, model_io):
        """Save then load returns an sklearn model (#217: must have .predict())."""
        original = _make_lr()
        model_io.save_model(original, "my_model")

        model, metadata = model_io.load_model("my_model")
        assert model is not None
        assert hasattr(model, "predict")
        assert metadata == {}

    def test_load_with_metadata_round_trip(self, model_io):
        """Save with metadata then load returns model + metadata (#217)."""
        original = _make_lr()
        meta = {"trained_at": "2026-02-13", "accuracy": 0.95}
        model_io.save_model(original, "scored_model", metadata=meta)

        model, metadata = model_io.load_model("scored_model")
        assert model is not None
        assert hasattr(model, "predict")
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
        """Loading a legacy pickle (raw sklearn object, no dict wrapper) works (#217)."""
        model_io.models_dir.mkdir(parents=True, exist_ok=True)
        path = model_io.models_dir / "legacy.pkl"
        lr = _make_lr()
        with open(path, "wb") as f:
            pickle.dump(lr, f)

        model, metadata = model_io.load_model("legacy")
        assert model is not None
        assert hasattr(model, "predict")
        assert metadata == {}

    def test_load_complex_model(self, model_io):
        """Round-trip with a real sklearn model verifies predict() interface (#217)."""
        original = _make_lr()
        model_io.save_model(original, "complex")

        model, _ = model_io.load_model("complex")
        assert model is not None
        assert hasattr(model, "predict")
        # Verify it actually predicts correctly
        pred = model.predict(np.array([[4]]))
        assert abs(pred[0] - 4.0) < 0.1


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


# =============================================================================
# #217 — model_io validates loaded object has .predict() interface
# =============================================================================


class TestLoadModelValidation:
    """load_model must validate the loaded object has a .predict() interface."""

    def test_load_dict_without_predict_returns_none_and_logs_error(self, model_io_with_dir, caplog):
        """#217: pickle.load of a dict (no .predict) returns None and logs ERROR."""
        import logging

        path = model_io_with_dir.models_dir / "bad_model.pkl"
        # Write a raw dict — no "model" key, no .predict() method
        with open(path, "wb") as f:
            pickle.dump({"not_a_model": True}, f)

        with caplog.at_level(logging.ERROR):
            model, _metadata = model_io_with_dir.load_model("bad_model")

        assert model is None, f"Expected None for unpredictable object, got: {model}"
        error_msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("predict" in m.lower() or "interface" in m.lower() or "model_io" in m.lower() for m in error_msgs), (
            f"Expected ERROR log about missing .predict(), got: {caplog.records}"
        )

    def test_load_valid_model_with_predict_succeeds(self, model_io_with_dir):
        """#217: sklearn-like object with .predict() loads successfully."""
        import numpy as np
        from sklearn.linear_model import LinearRegression

        # Use a real sklearn model (picklable and has .predict())
        lr = LinearRegression()
        lr.fit(np.array([[1], [2], [3]]), np.array([1.0, 2.0, 3.0]))
        model_io_with_dir.save_model(lr, "valid_model")
        loaded_model, _ = model_io_with_dir.load_model("valid_model")
        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")
