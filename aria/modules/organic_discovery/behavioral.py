"""Behavioral clustering — Layer 2 of organic capability discovery.

Discovers entity groups based on temporal co-occurrence: entities that change
state together within time windows, regardless of their type or domain.

Layer 1 clusters by what entities ARE (attributes).
Layer 2 clusters by what entities DO (temporal co-occurrence).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler


def _parse_when(when_str: str) -> datetime:
    """Parse ISO 8601 timestamp from logbook entry."""
    return datetime.fromisoformat(when_str)


def _window_key(ts: datetime, window_minutes: int) -> tuple:
    """Compute the time-window bucket key for a timestamp.

    Groups by (date, hour, minute_bin) where minute_bin = minute // window_minutes.
    """
    return (ts.date(), ts.hour, ts.minute // window_minutes)


def build_cooccurrence_matrix(
    logbook_entries: list[dict],
    window_minutes: int = 15,
) -> tuple[np.ndarray, list[str]]:
    """Build entity co-occurrence matrix from logbook state changes.

    Groups events into time windows. For each window, every pair of entities
    that changed state within that window gets a co-occurrence count += 1.

    Args:
        logbook_entries: List of dicts with entity_id, state, when keys.
        window_minutes: Size of time window in minutes for grouping.

    Returns:
        (matrix, entity_ids) where matrix[i][j] = co-occurrence count.
        Matrix is symmetric. Diagonal[i][i] = number of windows entity i appeared in.
    """
    if not logbook_entries:
        return np.empty((0, 0)), []

    # 1. Group entity_ids by time window (skip system entries without entity_id)
    windows: dict[tuple, set[str]] = defaultdict(set)
    for entry in logbook_entries:
        eid = entry.get("entity_id")
        if not eid:
            continue
        ts = _parse_when(entry["when"])
        key = _window_key(ts, window_minutes)
        windows[key].add(eid)

    # 2. Collect all unique entity IDs in stable sorted order
    all_entities: set[str] = set()
    for entities in windows.values():
        all_entities.update(entities)
    entity_ids = sorted(all_entities)
    entity_index = {eid: i for i, eid in enumerate(entity_ids)}

    n = len(entity_ids)
    matrix = np.zeros((n, n), dtype=np.float64)

    # 3. For each window, increment co-occurrence for all pairs
    for entities in windows.values():
        indices = [entity_index[eid] for eid in entities]
        for a in indices:
            for b in indices:
                matrix[a][b] += 1

    return matrix, entity_ids


def extract_temporal_pattern(
    entity_ids: list[str],
    logbook_entries: list[dict],
) -> dict:
    """Extract when a group of entities is most active.

    Args:
        entity_ids: Entity IDs to analyze.
        logbook_entries: Full logbook to filter from.

    Returns:
        {
            "peak_hours": list of hours where activity > 1.5x average,
            "weekday_bias": fraction of events on weekdays (Mon-Fri), 0.0 if no events
        }
    """
    entity_set = set(entity_ids)
    timestamps: list[datetime] = []
    for entry in logbook_entries:
        if entry.get("entity_id") in entity_set:
            timestamps.append(_parse_when(entry["when"]))

    if not timestamps:
        return {"peak_hours": [], "weekday_bias": 0.0}

    # Hour distribution
    hour_counts = np.zeros(24, dtype=np.float64)
    for ts in timestamps:
        hour_counts[ts.hour] += 1

    avg = hour_counts.mean()
    peak_hours = [h for h in range(24) if hour_counts[h] > 1.5 * avg] if avg > 0 else []

    # Weekday bias: Mon=0..Fri=4 are weekdays
    weekday_count = sum(1 for ts in timestamps if ts.weekday() < 5)
    weekday_bias = weekday_count / len(timestamps)

    return {
        "peak_hours": peak_hours,
        "weekday_bias": round(weekday_bias, 4),
    }


def cluster_behavioral(
    logbook_entries: list[dict],
    min_cluster_size: int = 3,
    window_minutes: int = 15,
) -> list[dict]:
    """Full pipeline: build co-occurrence -> HDBSCAN -> extract temporal patterns.

    Args:
        logbook_entries: Logbook state-change entries.
        min_cluster_size: Minimum entities per cluster for HDBSCAN.
        window_minutes: Co-occurrence time window in minutes.

    Returns:
        List of cluster dicts with keys:
            cluster_id, entity_ids, silhouette, temporal_pattern
    """
    if not logbook_entries:
        return []

    # 1. Build co-occurrence matrix
    matrix, entity_ids = build_cooccurrence_matrix(logbook_entries, window_minutes)

    if len(entity_ids) < min_cluster_size:
        return []

    # 2. Standardize and cluster
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size - 1),
    )
    labels = clusterer.fit_predict(scaled)

    # 3. Identify clusters (exclude noise = -1)
    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        return []

    # 4. Compute silhouette scores
    n_clusters = len(unique_labels)
    sample_scores = silhouette_samples(scaled, labels) if n_clusters >= 2 else np.zeros(len(labels))

    # 5. Build cluster results with temporal patterns
    entity_arr = np.array(entity_ids)
    clusters = []
    for label in unique_labels:
        mask = labels == label
        member_ids = entity_arr[mask].tolist()

        avg_silhouette = float(np.mean(sample_scores[mask])) if n_clusters >= 2 else 0.0

        temporal_pattern = extract_temporal_pattern(member_ids, logbook_entries)

        clusters.append(
            {
                "cluster_id": int(label),
                "entity_ids": member_ids,
                "silhouette": avg_silhouette,
                "temporal_pattern": temporal_pattern,
            }
        )

    return clusters
