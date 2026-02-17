import numpy as np
import pytest

from aria.engine.evaluation import expanding_window_cv


class TestExpandingWindowCV:
    def test_3_fold_expanding_window(self):
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=3))
        assert len(folds) == 3
        for X_train, _y_train, X_val, _y_val in folds:
            assert len(X_train) > 0
            assert len(X_val) > 0
        # Training set grows with each fold
        assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])

    def test_5_fold_expanding_window(self):
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=5))
        assert len(folds) == 5

    def test_fold_sizes_correct(self):
        """Validation sets should be roughly equal size."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=3))
        val_sizes = [len(f[3]) for f in folds]
        for size in val_sizes:
            assert 10 <= size <= 25

    def test_no_data_leakage(self):
        """Training indices must always precede validation indices."""
        X = np.arange(60).reshape(-1, 1)
        y = np.arange(60, dtype=float)
        for X_tr, _y_tr, X_val, _y_val in expanding_window_cv(X, y, n_folds=3):
            assert X_tr.max() < X_val.min()

    def test_single_fold_fallback(self):
        """With n_folds=1, should return single 80/20 split."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=1))
        assert len(folds) == 1
        X_tr, _, X_val, _ = folds[0]
        assert len(X_tr) == 48  # 80% of 60
        assert len(X_val) == 12  # 20% of 60

    def test_zero_folds_raises(self):
        """n_folds=0 should raise ValueError, not silently produce nothing."""
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        with pytest.raises(ValueError, match="n_folds must be >= 1"):
            list(expanding_window_cv(X, y, n_folds=0))

    def test_too_few_samples_raises(self):
        """More folds than samples should raise ValueError."""
        X = np.random.randn(3, 2)
        y = np.random.randn(3)
        with pytest.raises(ValueError, match="Not enough samples"):
            list(expanding_window_cv(X, y, n_folds=3))
