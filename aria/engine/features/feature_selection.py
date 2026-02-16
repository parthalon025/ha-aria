"""mRMR (minimum Redundancy Maximum Relevance) feature selection.

Selects a subset of features that are individually relevant to the target
while being minimally redundant with each other.  Used to prune the 48-dim
feature vector before model training so gradient-boosting and random-forest
models focus on the most informative signals.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 30,
) -> list[str]:
    """Select features using mRMR (minimum Redundancy Maximum Relevance).

    Algorithm:
        1. Compute mutual information between each feature and the target
           (relevance scores).
        2. Greedily select features that maximize:
              relevance(f) - mean(|corr(f, already_selected)|)
           where redundancy is approximated by the mean absolute Pearson
           correlation with features already in the selected set.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        feature_names: Names corresponding to columns of X.
        max_features: Maximum number of features to select.

    Returns:
        List of selected feature names, ordered by selection round.
    """
    n_samples, n_features = X.shape

    # Edge case: fewer features than requested — return all
    if n_features <= max_features:
        logger.info(
            "mRMR: %d features <= max_features=%d, returning all",
            n_features,
            max_features,
        )
        return list(feature_names)

    # Step 1: Compute relevance (mutual information with target)
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        logger.warning("sklearn not available; returning all features unselected")
        return list(feature_names)

    relevance = mutual_info_regression(X, y, random_state=42)

    # Pre-compute full absolute correlation matrix once (avoids O(k*n) per-pair calls)
    corr_matrix = np.abs(np.corrcoef(X.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Step 2: Greedy forward selection
    selected_indices: list[int] = []
    remaining = set(range(n_features))

    for _round_num in range(max_features):
        best_score: float | None = None
        best_idx: int | None = None

        for idx in remaining:
            rel = relevance[idx]

            # Compute redundancy: mean |corr| with already-selected features
            redundancy = corr_matrix[idx, selected_indices].mean() if selected_indices else 0.0

            score = rel - redundancy

            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.discard(best_idx)

    selected_names = [feature_names[i] for i in selected_indices]
    logger.info(
        "mRMR selected %d/%d features: %s",
        len(selected_names),
        n_features,
        selected_names,
    )
    return selected_names
