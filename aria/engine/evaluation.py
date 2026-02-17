"""Time-series cross-validation utilities."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np


def expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 3,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Expanding-window cross-validation for time-series data.

    Preserves temporal ordering: training always precedes validation.
    Fold k trains on the first (k+1)/(n_folds+1) of data, validates on the next chunk.

    n_folds=1 is equivalent to a single 80/20 split.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    n = len(X)
    if n_folds == 1:
        split = int(n * 0.8)
        yield X[:split], y[:split], X[split:], y[split:]
        return

    chunk_size = n // (n_folds + 1)
    if chunk_size == 0:
        raise ValueError(f"Not enough samples ({n}) for {n_folds} folds — need at least {n_folds + 1}")
    for k in range(n_folds):
        train_end = chunk_size * (k + 1)
        val_end = min(train_end + chunk_size, n)
        if k == n_folds - 1:
            val_end = n  # Last fold gets remaining samples
        yield X[:train_end], y[:train_end], X[train_end:val_end], y[train_end:val_end]
