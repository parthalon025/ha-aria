"""HDBSCAN clustering engine for organic entity discovery.

Takes a feature matrix and entity IDs, runs HDBSCAN clustering,
and returns cluster information with per-cluster silhouette scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler


def cluster_entities(
    matrix: np.ndarray,
    entity_ids: list[str],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[dict]:
    """Cluster entities using HDBSCAN on a feature matrix.

    Args:
        matrix: Feature matrix of shape (n_entities, n_features).
        entity_ids: Entity ID strings, one per row in matrix.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples: Minimum samples for HDBSCAN core point definition.

    Returns:
        List of cluster dicts with keys: cluster_id, entity_ids, silhouette.
        Noise points (label -1) are excluded.

    Raises:
        ValueError: If matrix row count doesn't match entity_ids length.
    """
    if matrix.shape[0] != len(entity_ids):
        raise ValueError(f"Matrix has {matrix.shape[0]} rows but got {len(entity_ids)} entity_ids")

    # Edge case: empty or too-small input
    if matrix.shape[0] < min_cluster_size:
        return []

    # 1. Standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    # 2. Run HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(scaled)

    # 3. Identify unique clusters (exclude noise = -1)
    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        return []

    # 4. Compute per-sample silhouette scores (needs >= 2 clusters)
    n_clusters = len(unique_labels)
    has_silhouette = n_clusters >= 2
    sample_scores = silhouette_samples(scaled, labels) if has_silhouette else np.zeros(len(labels))

    # 5. Build cluster dicts
    entity_arr = np.array(entity_ids)
    clusters = []
    for label in unique_labels:
        mask = labels == label
        member_ids = entity_arr[mask].tolist()
        # Single cluster — silhouette is undefined, report 0.0
        avg_silhouette = float(np.mean(sample_scores[mask])) if has_silhouette else 0.0

        clusters.append(
            {
                "cluster_id": int(label),
                "entity_ids": member_ids,
                "silhouette": avg_silhouette,
            }
        )

    return clusters
