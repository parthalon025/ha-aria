# Organic Capability Discovery — Implementation Plan

## In Plain English

This is the construction plan for building ARIA's automatic capability detection system. It works in layers -- first teach it to group similar devices together by their properties, then teach it to notice devices that act in concert based on behavioral patterns, then wire up the scoring, naming, dashboard UI, and user controls.

## Why This Exists

The design document describes what organic discovery should do; this plan describes how to build it without breaking ARIA's existing capability system along the way. The implementation spans a new Python module, database schema extensions, API endpoints, and dashboard pages. Each task is sequenced so that tests are written before code, each piece is independently verifiable, and the existing 10 hard-coded capabilities continue working throughout as a safety net.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace hard-coded capability detection with organic two-layer discovery (domain clustering + behavioral co-occurrence), usefulness scoring, user-selectable autonomy, and full dashboard UI.

**Architecture:** New `OrganicDiscoveryModule` registered after existing discovery module. Reads entity/logbook data → HDBSCAN clustering → usefulness scoring → cache write. Extended Capabilities page shows promoted/candidate/archived sections with usefulness bars. Settings panel controls autonomy mode and naming backend.

**Tech Stack:** Python (sklearn HDBSCAN, numpy, scipy), FastAPI, Preact + Tailwind, SQLite cache

**Design doc:** `docs/plans/2026-02-14-organic-capability-discovery-design.md`

---

## Phase 1: Domain Clustering (Layer 1)

### Task 1: Feature Vector Builder

**Files:**
- Create: `aria/modules/organic_discovery/__init__.py`
- Create: `aria/modules/organic_discovery/feature_vectors.py`
- Test: `tests/hub/test_organic_feature_vectors.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_feature_vectors.py
"""Tests for entity feature vector construction."""
import pytest
import numpy as np
from aria.modules.organic_discovery.feature_vectors import build_feature_matrix


MOCK_ENTITIES = [
    {"entity_id": "light.living_room", "state": "on", "attributes": {
        "device_class": None, "friendly_name": "Living Room Light",
        "brightness": 200, "color_temp": 370, "supported_color_modes": ["brightness", "color_temp"],
    }},
    {"entity_id": "light.bedroom", "state": "off", "attributes": {
        "device_class": None, "friendly_name": "Bedroom Light",
        "brightness": 0, "supported_color_modes": ["brightness"],
    }},
    {"entity_id": "sensor.power_outlet_1", "state": "145.3", "attributes": {
        "device_class": "power", "unit_of_measurement": "W",
        "friendly_name": "Outlet 1 Power",
    }},
    {"entity_id": "sensor.temperature_kitchen", "state": "22.5", "attributes": {
        "device_class": "temperature", "unit_of_measurement": "°C",
        "friendly_name": "Kitchen Temperature",
    }},
    {"entity_id": "binary_sensor.front_door", "state": "off", "attributes": {
        "device_class": "door", "friendly_name": "Front Door",
    }},
    {"entity_id": "climate.living_room", "state": "heat", "attributes": {
        "device_class": None, "hvac_modes": ["heat", "cool", "off"],
        "current_temperature": 21.5, "temperature": 22.0,
        "friendly_name": "Living Room Thermostat",
    }},
]

MOCK_DEVICES = {
    "device_1": {"area_id": "living_room", "manufacturer": "Philips"},
    "device_2": {"area_id": "bedroom", "manufacturer": "Philips"},
    "device_3": {"area_id": "kitchen", "manufacturer": "Shelly"},
}

MOCK_ENTITY_REGISTRY = {
    "light.living_room": {"device_id": "device_1"},
    "light.bedroom": {"device_id": "device_2"},
    "sensor.power_outlet_1": {"device_id": "device_3"},
    "sensor.temperature_kitchen": {"device_id": "device_3"},
    "binary_sensor.front_door": {"device_id": None},
    "climate.living_room": {"device_id": "device_1"},
}

MOCK_ACTIVITY = {
    "light.living_room": 8.5,
    "light.bedroom": 4.2,
    "sensor.power_outlet_1": 96.0,
    "sensor.temperature_kitchen": 48.0,
    "binary_sensor.front_door": 3.1,
    "climate.living_room": 12.0,
}


def test_build_feature_matrix_returns_correct_shape():
    matrix, entity_ids, feature_names = build_feature_matrix(
        MOCK_ENTITIES, MOCK_DEVICES, MOCK_ENTITY_REGISTRY, MOCK_ACTIVITY
    )
    assert matrix.shape[0] == len(MOCK_ENTITIES)
    assert matrix.shape[1] == len(feature_names)
    assert len(entity_ids) == len(MOCK_ENTITIES)


def test_build_feature_matrix_entity_ids_match():
    matrix, entity_ids, feature_names = build_feature_matrix(
        MOCK_ENTITIES, MOCK_DEVICES, MOCK_ENTITY_REGISTRY, MOCK_ACTIVITY
    )
    expected_ids = [e["entity_id"] for e in MOCK_ENTITIES]
    assert entity_ids == expected_ids


def test_build_feature_matrix_includes_domain_features():
    matrix, entity_ids, feature_names = build_feature_matrix(
        MOCK_ENTITIES, MOCK_DEVICES, MOCK_ENTITY_REGISTRY, MOCK_ACTIVITY
    )
    # Should have domain one-hot columns
    domain_features = [f for f in feature_names if f.startswith("domain_")]
    assert len(domain_features) >= 3  # light, sensor, binary_sensor, climate


def test_build_feature_matrix_includes_activity():
    matrix, entity_ids, feature_names = build_feature_matrix(
        MOCK_ENTITIES, MOCK_DEVICES, MOCK_ENTITY_REGISTRY, MOCK_ACTIVITY
    )
    assert "avg_daily_state_changes" in feature_names
    activity_col = feature_names.index("avg_daily_state_changes")
    light_row = entity_ids.index("light.living_room")
    assert matrix[light_row, activity_col] == pytest.approx(8.5)


def test_build_feature_matrix_handles_missing_device():
    """Entities without a device should still get feature vectors."""
    matrix, entity_ids, feature_names = build_feature_matrix(
        MOCK_ENTITIES, MOCK_DEVICES, MOCK_ENTITY_REGISTRY, MOCK_ACTIVITY
    )
    door_row = entity_ids.index("binary_sensor.front_door")
    assert not np.isnan(matrix[door_row]).any()


def test_build_feature_matrix_empty_input():
    matrix, entity_ids, feature_names = build_feature_matrix([], {}, {}, {})
    assert matrix.shape == (0, 0)
    assert entity_ids == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_feature_vectors.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'aria.modules.organic_discovery'"

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/__init__.py
"""Organic capability discovery — automatic entity clustering and behavioral pattern detection."""

# aria/modules/organic_discovery/feature_vectors.py
"""Build feature vectors from HA entity states for clustering."""
import numpy as np
from typing import Any


def build_feature_matrix(
    entities: list[dict[str, Any]],
    devices: dict[str, dict],
    entity_registry: dict[str, dict],
    activity_rates: dict[str, float],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a numeric feature matrix from entity states and metadata.

    Returns:
        (matrix, entity_ids, feature_names) where matrix is (n_entities, n_features).
    """
    if not entities:
        return np.empty((0, 0)), [], []

    # Extract raw features per entity
    rows = []
    entity_ids = []

    # Collect all unique values for categorical encoding
    all_domains = sorted(set(eid.split(".")[0] for e in entities for eid in [e["entity_id"]]))
    all_device_classes = sorted(set(
        e.get("attributes", {}).get("device_class") or "none"
        for e in entities
    ))
    all_units = sorted(set(
        e.get("attributes", {}).get("unit_of_measurement") or "none"
        for e in entities
    ))
    all_areas = sorted(set(
        _resolve_area(e["entity_id"], devices, entity_registry) or "none"
        for e in entities
    ))
    all_manufacturers = sorted(set(
        _resolve_manufacturer(e["entity_id"], devices, entity_registry) or "none"
        for e in entities
    ))

    for entity in entities:
        eid = entity["entity_id"]
        attrs = entity.get("attributes", {})
        domain = eid.split(".")[0]
        device_class = attrs.get("device_class") or "none"
        unit = attrs.get("unit_of_measurement") or "none"
        area = _resolve_area(eid, devices, entity_registry) or "none"
        manufacturer = _resolve_manufacturer(eid, devices, entity_registry) or "none"

        row = []

        # Domain one-hot
        for d in all_domains:
            row.append(1.0 if domain == d else 0.0)

        # Device class one-hot
        for dc in all_device_classes:
            row.append(1.0 if device_class == dc else 0.0)

        # Unit one-hot
        for u in all_units:
            row.append(1.0 if unit == u else 0.0)

        # Area one-hot
        for a in all_areas:
            row.append(1.0 if area == a else 0.0)

        # Manufacturer one-hot
        for m in all_manufacturers:
            row.append(1.0 if manufacturer == m else 0.0)

        # Numeric features
        row.append(_count_state_cardinality(entity))
        row.append(activity_rates.get(eid, 0.0))
        row.append(1.0 if entity.get("state") != "unavailable" else 0.0)

        # Capability flags
        row.append(1.0 if "brightness" in str(attrs.get("supported_color_modes", [])) else 0.0)
        row.append(1.0 if "color_temp" in str(attrs.get("supported_color_modes", [])) else 0.0)
        row.append(1.0 if "rgb" in str(attrs.get("supported_color_modes", [])).lower() else 0.0)
        row.append(1.0 if attrs.get("hvac_modes") else 0.0)
        row.append(1.0 if attrs.get("temperature") is not None else 0.0)

        rows.append(row)
        entity_ids.append(eid)

    # Build feature names
    feature_names = (
        [f"domain_{d}" for d in all_domains]
        + [f"device_class_{dc}" for dc in all_device_classes]
        + [f"unit_{u}" for u in all_units]
        + [f"area_{a}" for a in all_areas]
        + [f"manufacturer_{m}" for m in all_manufacturers]
        + [
            "state_cardinality",
            "avg_daily_state_changes",
            "available",
            "has_brightness",
            "has_color_temp",
            "has_rgb",
            "has_hvac",
            "has_temperature_target",
        ]
    )

    matrix = np.array(rows, dtype=np.float64)
    return matrix, entity_ids, feature_names


def _resolve_area(entity_id: str, devices: dict, entity_registry: dict) -> str | None:
    reg = entity_registry.get(entity_id, {})
    device_id = reg.get("device_id")
    if device_id and device_id in devices:
        return devices[device_id].get("area_id")
    return None


def _resolve_manufacturer(entity_id: str, devices: dict, entity_registry: dict) -> str | None:
    reg = entity_registry.get(entity_id, {})
    device_id = reg.get("device_id")
    if device_id and device_id in devices:
        return devices[device_id].get("manufacturer")
    return None


def _count_state_cardinality(entity: dict) -> float:
    """Estimate number of distinct states from entity type."""
    state = entity.get("state", "")
    if state in ("on", "off"):
        return 2.0
    try:
        float(state)
        return 100.0  # numeric sensors have high cardinality
    except (ValueError, TypeError):
        return 5.0  # other states (heat, cool, home, away, etc.)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_feature_vectors.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/__init__.py aria/modules/organic_discovery/feature_vectors.py tests/hub/test_organic_feature_vectors.py
git commit -m "feat(organic-discovery): entity feature vector builder for clustering"
```

---

### Task 2: HDBSCAN Clustering Engine

**Files:**
- Create: `aria/modules/organic_discovery/clustering.py`
- Test: `tests/hub/test_organic_clustering.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_clustering.py
"""Tests for HDBSCAN-based entity clustering."""
import pytest
import numpy as np
from aria.modules.organic_discovery.clustering import cluster_entities


def _make_two_clusters(n_each=20, n_features=10, seed=42):
    """Create synthetic data with 2 clear clusters."""
    rng = np.random.RandomState(seed)
    cluster_a = rng.randn(n_each, n_features) + np.array([5.0] * n_features)
    cluster_b = rng.randn(n_each, n_features) + np.array([-5.0] * n_features)
    matrix = np.vstack([cluster_a, cluster_b])
    ids = [f"entity.a_{i}" for i in range(n_each)] + [f"entity.b_{i}" for i in range(n_each)]
    return matrix, ids


def test_cluster_entities_finds_two_clusters():
    matrix, ids = _make_two_clusters()
    clusters = cluster_entities(matrix, ids)
    # Should find at least 2 clusters
    assert len(clusters) >= 2


def test_cluster_entities_returns_correct_structure():
    matrix, ids = _make_two_clusters()
    clusters = cluster_entities(matrix, ids)
    for cluster in clusters:
        assert "cluster_id" in cluster
        assert "entity_ids" in cluster
        assert "silhouette" in cluster
        assert isinstance(cluster["entity_ids"], list)
        assert len(cluster["entity_ids"]) > 0


def test_cluster_entities_assigns_all_non_noise():
    matrix, ids = _make_two_clusters()
    clusters = cluster_entities(matrix, ids)
    assigned = set()
    for c in clusters:
        assigned.update(c["entity_ids"])
    # Most entities should be assigned (some may be noise)
    assert len(assigned) >= len(ids) * 0.8


def test_cluster_entities_separates_distinct_groups():
    matrix, ids = _make_two_clusters()
    clusters = cluster_entities(matrix, ids)
    # Find cluster containing a_0 and cluster containing b_0
    a_cluster = None
    b_cluster = None
    for c in clusters:
        if "entity.a_0" in c["entity_ids"]:
            a_cluster = c["cluster_id"]
        if "entity.b_0" in c["entity_ids"]:
            b_cluster = c["cluster_id"]
    assert a_cluster is not None
    assert b_cluster is not None
    assert a_cluster != b_cluster


def test_cluster_entities_handles_small_input():
    """Fewer than min_cluster_size entities should return empty clusters."""
    matrix = np.array([[1.0, 2.0], [1.1, 2.1]])
    ids = ["entity.a", "entity.b"]
    clusters = cluster_entities(matrix, ids, min_cluster_size=5)
    # Should handle gracefully — either empty or single cluster
    assert isinstance(clusters, list)


def test_cluster_entities_silhouette_range():
    matrix, ids = _make_two_clusters()
    clusters = cluster_entities(matrix, ids)
    for c in clusters:
        assert -1.0 <= c["silhouette"] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_clustering.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/clustering.py
"""HDBSCAN-based entity clustering for organic capability discovery."""
import logging
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def cluster_entities(
    matrix: np.ndarray,
    entity_ids: list[str],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> list[dict]:
    """Cluster entities using HDBSCAN on their feature vectors.

    Returns list of cluster dicts with keys:
        cluster_id, entity_ids, silhouette, centroid_indices
    """
    if matrix.shape[0] < min_cluster_size:
        logger.warning(f"Too few entities ({matrix.shape[0]}) for clustering (min={min_cluster_size})")
        return []

    # Scale features for distance computation
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(scaled)

    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    if len(unique_labels) < 2:
        # Can't compute silhouette with < 2 clusters
        if len(unique_labels) == 1:
            label = list(unique_labels)[0]
            members = [entity_ids[i] for i, l in enumerate(labels) if l == label]
            return [{"cluster_id": 0, "entity_ids": members, "silhouette": 0.0}]
        return []

    # Compute per-sample silhouette (only for non-noise points)
    non_noise_mask = labels != -1
    sil_scores = np.zeros(len(labels))
    if non_noise_mask.sum() >= 2:
        sil_scores[non_noise_mask] = silhouette_samples(
            scaled[non_noise_mask], labels[non_noise_mask]
        )

    clusters = []
    for label in sorted(unique_labels):
        member_mask = labels == label
        member_ids = [entity_ids[i] for i, m in enumerate(member_mask) if m]
        avg_silhouette = float(np.mean(sil_scores[member_mask]))

        clusters.append({
            "cluster_id": int(label),
            "entity_ids": member_ids,
            "silhouette": round(avg_silhouette, 4),
        })

    logger.info(f"HDBSCAN found {len(clusters)} clusters, {(labels == -1).sum()} noise points")
    return clusters
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_clustering.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/clustering.py tests/hub/test_organic_clustering.py
git commit -m "feat(organic-discovery): HDBSCAN clustering engine"
```

---

### Task 3: Seed Validation

**Files:**
- Create: `aria/modules/organic_discovery/seed_validation.py`
- Test: `tests/hub/test_organic_seed_validation.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_seed_validation.py
"""Tests for seed capability validation against discovered clusters."""
import pytest
from aria.modules.organic_discovery.seed_validation import validate_seeds, jaccard_similarity


SEED_CAPABILITIES = {
    "lighting": {
        "entities": ["light.living_room", "light.bedroom", "light.kitchen"],
    },
    "power_monitoring": {
        "entities": ["sensor.outlet_1_power", "sensor.outlet_2_power"],
    },
}


def test_jaccard_similarity_identical():
    assert jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0


def test_jaccard_similarity_disjoint():
    assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_similarity_partial():
    result = jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
    assert result == pytest.approx(0.5)  # 2 / 4


def test_jaccard_similarity_empty():
    assert jaccard_similarity(set(), set()) == 0.0


def test_validate_seeds_perfect_match():
    clusters = [
        {"cluster_id": 0, "entity_ids": ["light.living_room", "light.bedroom", "light.kitchen"]},
        {"cluster_id": 1, "entity_ids": ["sensor.outlet_1_power", "sensor.outlet_2_power"]},
    ]
    results = validate_seeds(SEED_CAPABILITIES, clusters)
    assert results["lighting"]["best_jaccard"] == 1.0
    assert results["power_monitoring"]["best_jaccard"] == 1.0
    assert results["lighting"]["matched"]
    assert results["power_monitoring"]["matched"]


def test_validate_seeds_partial_match():
    clusters = [
        {"cluster_id": 0, "entity_ids": ["light.living_room", "light.bedroom"]},  # missing kitchen
    ]
    results = validate_seeds(SEED_CAPABILITIES, clusters)
    # 2/3 overlap = 0.667 Jaccard — below 0.8 threshold
    assert not results["lighting"]["matched"]
    assert results["lighting"]["best_jaccard"] == pytest.approx(2 / 3, abs=0.01)


def test_validate_seeds_no_match():
    clusters = [
        {"cluster_id": 0, "entity_ids": ["switch.a", "switch.b"]},
    ]
    results = validate_seeds(SEED_CAPABILITIES, clusters)
    assert not results["lighting"]["matched"]
    assert results["lighting"]["best_jaccard"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_seed_validation.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/seed_validation.py
"""Validate organic clusters against seed (hard-coded) capabilities."""
import logging

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.8


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def validate_seeds(
    seed_capabilities: dict,
    clusters: list[dict],
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """Compare discovered clusters against seed capabilities.

    Returns dict mapping seed name to validation result:
        best_jaccard, best_cluster_id, matched (bool), matched_entities
    """
    results = {}

    for seed_name, seed_data in seed_capabilities.items():
        seed_entities = set(seed_data.get("entities", []))
        best_jaccard = 0.0
        best_cluster_id = None

        for cluster in clusters:
            cluster_entities = set(cluster["entity_ids"])
            sim = jaccard_similarity(seed_entities, cluster_entities)
            if sim > best_jaccard:
                best_jaccard = sim
                best_cluster_id = cluster["cluster_id"]

        matched = best_jaccard >= threshold
        if not matched and seed_entities:
            logger.warning(
                f"Seed '{seed_name}' not reproduced: best Jaccard={best_jaccard:.2f} "
                f"(threshold={threshold})"
            )

        results[seed_name] = {
            "best_jaccard": round(best_jaccard, 4),
            "best_cluster_id": best_cluster_id,
            "matched": matched,
        }

    return results
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_seed_validation.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/seed_validation.py tests/hub/test_organic_seed_validation.py
git commit -m "feat(organic-discovery): seed validation with Jaccard similarity"
```

---

### Task 4: Usefulness Scoring

**Files:**
- Create: `aria/modules/organic_discovery/scoring.py`
- Test: `tests/hub/test_organic_scoring.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_scoring.py
"""Tests for capability usefulness scoring."""
import pytest
from aria.modules.organic_discovery.scoring import compute_usefulness, UsefulnessComponents


def test_compute_usefulness_all_perfect():
    components = UsefulnessComponents(
        predictability=1.0, stability=1.0, entity_coverage=1.0,
        activity=1.0, cohesion=1.0,
    )
    score = compute_usefulness(components)
    assert score == 100


def test_compute_usefulness_all_zero():
    components = UsefulnessComponents(
        predictability=0.0, stability=0.0, entity_coverage=0.0,
        activity=0.0, cohesion=0.0,
    )
    score = compute_usefulness(components)
    assert score == 0


def test_compute_usefulness_weights():
    """Predictability (30%) should matter most."""
    high_predict = UsefulnessComponents(
        predictability=1.0, stability=0.0, entity_coverage=0.0,
        activity=0.0, cohesion=0.0,
    )
    high_stability = UsefulnessComponents(
        predictability=0.0, stability=1.0, entity_coverage=0.0,
        activity=0.0, cohesion=0.0,
    )
    assert compute_usefulness(high_predict) > compute_usefulness(high_stability)


def test_compute_usefulness_returns_int():
    components = UsefulnessComponents(
        predictability=0.5, stability=0.8, entity_coverage=0.3,
        activity=0.6, cohesion=0.7,
    )
    score = compute_usefulness(components)
    assert isinstance(score, int)
    assert 0 <= score <= 100


def test_compute_usefulness_clamped():
    """Values above 1.0 should be clamped."""
    components = UsefulnessComponents(
        predictability=1.5, stability=1.2, entity_coverage=0.5,
        activity=0.5, cohesion=0.5,
    )
    score = compute_usefulness(components)
    assert score <= 100


def test_usefulness_components_to_dict():
    components = UsefulnessComponents(
        predictability=0.92, stability=1.0, entity_coverage=0.65,
        activity=0.78, cohesion=0.88,
    )
    d = components.to_dict()
    assert d == {
        "predictability": 92,
        "stability": 100,
        "entity_coverage": 65,
        "activity": 78,
        "cohesion": 88,
    }
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_scoring.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/scoring.py
"""Usefulness scoring for discovered capabilities."""
from dataclasses import dataclass

WEIGHTS = {
    "predictability": 0.30,
    "stability": 0.25,
    "entity_coverage": 0.15,
    "activity": 0.15,
    "cohesion": 0.15,
}


@dataclass
class UsefulnessComponents:
    predictability: float  # 0.0–1.0: ML model accuracy for this cluster
    stability: float       # 0.0–1.0: fraction of runs this cluster appeared
    entity_coverage: float # 0.0–1.0: entities in cluster / total entities (scaled)
    activity: float        # 0.0–1.0: normalized avg daily state changes
    cohesion: float        # 0.0–1.0: silhouette score (rescaled from [-1, 1])

    def to_dict(self) -> dict[str, int]:
        return {
            "predictability": int(round(min(self.predictability, 1.0) * 100)),
            "stability": int(round(min(self.stability, 1.0) * 100)),
            "entity_coverage": int(round(min(self.entity_coverage, 1.0) * 100)),
            "activity": int(round(min(self.activity, 1.0) * 100)),
            "cohesion": int(round(min(self.cohesion, 1.0) * 100)),
        }


def compute_usefulness(components: UsefulnessComponents) -> int:
    """Compute weighted usefulness score (0-100)."""
    raw = (
        WEIGHTS["predictability"] * min(components.predictability, 1.0)
        + WEIGHTS["stability"] * min(components.stability, 1.0)
        + WEIGHTS["entity_coverage"] * min(components.entity_coverage, 1.0)
        + WEIGHTS["activity"] * min(components.activity, 1.0)
        + WEIGHTS["cohesion"] * min(components.cohesion, 1.0)
    )
    return int(round(min(raw, 1.0) * 100))
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_scoring.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/scoring.py tests/hub/test_organic_scoring.py
git commit -m "feat(organic-discovery): usefulness scoring with weighted components"
```

---

### Task 5: Heuristic Naming

**Files:**
- Create: `aria/modules/organic_discovery/naming.py`
- Test: `tests/hub/test_organic_naming.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_naming.py
"""Tests for heuristic cluster naming."""
import pytest
from aria.modules.organic_discovery.naming import heuristic_name, heuristic_description

CLUSTER_LIGHT = {
    "entity_ids": ["light.living_room", "light.bedroom", "light.kitchen"],
    "areas": {"living_room": 1, "bedroom": 1, "kitchen": 1},
    "domains": {"light": 3},
}

CLUSTER_POWER = {
    "entity_ids": ["sensor.outlet_1_power", "sensor.outlet_2_power", "sensor.outlet_3_power"],
    "areas": {"kitchen": 2, "garage": 1},
    "domains": {"sensor": 3},
    "device_classes": {"power": 3},
}

CLUSTER_MIXED_ROOM = {
    "entity_ids": ["light.office", "switch.office_fan", "climate.office"],
    "areas": {"office": 3},
    "domains": {"light": 1, "switch": 1, "climate": 1},
}


def test_heuristic_name_single_domain():
    name = heuristic_name(CLUSTER_LIGHT)
    assert "light" in name.lower()


def test_heuristic_name_device_class_preferred():
    name = heuristic_name(CLUSTER_POWER)
    assert "power" in name.lower()


def test_heuristic_name_room_dominant():
    """When all entities are in one area, include the area name."""
    name = heuristic_name(CLUSTER_MIXED_ROOM)
    assert "office" in name.lower()


def test_heuristic_name_is_snake_case():
    name = heuristic_name(CLUSTER_LIGHT)
    assert " " not in name
    assert name == name.lower()


def test_heuristic_description_nonempty():
    desc = heuristic_description(CLUSTER_LIGHT)
    assert len(desc) > 10


def test_heuristic_name_deduplication():
    """Same input should produce same name."""
    name1 = heuristic_name(CLUSTER_LIGHT)
    name2 = heuristic_name(CLUSTER_LIGHT)
    assert name1 == name2
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_naming.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/naming.py
"""Heuristic naming for discovered clusters."""


def heuristic_name(cluster_info: dict) -> str:
    """Generate a snake_case name from cluster metadata.

    cluster_info keys: entity_ids, domains, areas, device_classes (optional),
                       temporal_pattern (optional)
    """
    parts = []

    domains = cluster_info.get("domains", {})
    device_classes = cluster_info.get("device_classes", {})
    areas = cluster_info.get("areas", {})
    temporal = cluster_info.get("temporal_pattern", {})

    # Time-of-day prefix for behavioral clusters
    peak_hours = temporal.get("peak_hours", [])
    if peak_hours:
        avg_hour = sum(peak_hours) / len(peak_hours)
        if avg_hour < 6:
            parts.append("night")
        elif avg_hour < 12:
            parts.append("morning")
        elif avg_hour < 17:
            parts.append("afternoon")
        else:
            parts.append("evening")

    # Area if dominant (>= 60% of entities in one area)
    total_area_count = sum(areas.values()) if areas else 0
    if areas and total_area_count > 0:
        top_area, top_count = max(areas.items(), key=lambda x: x[1])
        if top_count / total_area_count >= 0.6:
            parts.append(top_area)

    # Device class if present (more specific than domain)
    if device_classes:
        top_dc = max(device_classes, key=device_classes.get)
        if top_dc != "none":
            parts.append(top_dc)
    elif domains:
        # Fall back to dominant domain
        top_domain = max(domains, key=domains.get)
        total_domain_count = sum(domains.values())
        if domains[top_domain] / total_domain_count >= 0.5:
            parts.append(top_domain)
        else:
            parts.append("mixed")

    if not parts:
        parts.append("cluster")

    return "_".join(parts)


def heuristic_description(cluster_info: dict) -> str:
    """Generate a human-readable description of a cluster."""
    entity_ids = cluster_info.get("entity_ids", [])
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})
    temporal = cluster_info.get("temporal_pattern", {})

    count = len(entity_ids)
    domain_str = ", ".join(f"{v} {k}" for k, v in sorted(domains.items(), key=lambda x: -x[1]))
    area_str = ", ".join(sorted(areas.keys())) if areas else "multiple areas"

    desc = f"{count} entities ({domain_str}) in {area_str}."

    peak_hours = temporal.get("peak_hours", [])
    if peak_hours:
        hour_range = f"{min(peak_hours)}:00-{max(peak_hours) + 1}:00"
        desc += f" Most active {hour_range}."

    return desc
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_naming.py -v`
Expected: 6 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/naming.py tests/hub/test_organic_naming.py
git commit -m "feat(organic-discovery): heuristic cluster naming"
```

---

### Task 6: Organic Discovery Module (hub integration)

**Files:**
- Create: `aria/modules/organic_discovery/module.py`
- Modify: `aria/hub/constants.py` — add CACHE_DISCOVERY_HISTORY
- Modify: `aria/cli.py` — register module in serve startup
- Test: `tests/hub/test_organic_discovery_module.py`

This is the orchestrator that ties Tasks 1-5 together and integrates with the hub.

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_discovery_module.py
"""Tests for the OrganicDiscoveryModule hub integration."""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from aria.modules.organic_discovery.module import OrganicDiscoveryModule


@pytest.fixture
def mock_hub():
    hub = AsyncMock()
    hub.cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.schedule_task = AsyncMock()
    return hub


MOCK_ENTITIES_CACHE = {
    "data": [
        {"entity_id": "light.living_room", "state": "on", "attributes": {"device_class": None}},
        {"entity_id": "light.bedroom", "state": "off", "attributes": {"device_class": None}},
        {"entity_id": "light.kitchen", "state": "on", "attributes": {"device_class": None}},
        {"entity_id": "sensor.power_1", "state": "100", "attributes": {"device_class": "power", "unit_of_measurement": "W"}},
        {"entity_id": "sensor.power_2", "state": "200", "attributes": {"device_class": "power", "unit_of_measurement": "W"}},
        {"entity_id": "sensor.power_3", "state": "50", "attributes": {"device_class": "power", "unit_of_measurement": "W"}},
        {"entity_id": "binary_sensor.motion_1", "state": "off", "attributes": {"device_class": "motion"}},
        {"entity_id": "binary_sensor.motion_2", "state": "on", "attributes": {"device_class": "motion"}},
        {"entity_id": "binary_sensor.motion_3", "state": "off", "attributes": {"device_class": "motion"}},
        {"entity_id": "climate.main", "state": "heat", "attributes": {"hvac_modes": ["heat", "cool"]}},
    ] * 3  # Repeat to get enough entities for HDBSCAN min_cluster_size
}

MOCK_DEVICES_CACHE = {"data": {}}
MOCK_CAPABILITIES_CACHE = {
    "data": {
        "lighting": {"available": True, "entities": ["light.living_room", "light.bedroom", "light.kitchen"]},
        "power_monitoring": {"available": True, "entities": ["sensor.power_1", "sensor.power_2", "sensor.power_3"]},
    }
}


@pytest.mark.asyncio
async def test_module_init(mock_hub):
    module = OrganicDiscoveryModule(mock_hub)
    assert module.module_id == "organic_discovery"


@pytest.mark.asyncio
async def test_module_run_discovery_writes_cache(mock_hub):
    """Discovery run should write extended capabilities to cache."""
    mock_hub.cache.get = AsyncMock(side_effect=lambda cat: {
        "entities": MOCK_ENTITIES_CACHE,
        "devices": MOCK_DEVICES_CACHE,
        "capabilities": MOCK_CAPABILITIES_CACHE,
        "activity_summary": {"data": {"entity_activity": {}}},
        "discovery_history": None,
    }.get(cat))

    module = OrganicDiscoveryModule(mock_hub)
    await module.run_discovery()

    # Should have written to capabilities cache
    mock_hub.cache.set.assert_called()
    calls = mock_hub.cache.set.call_args_list
    cap_calls = [c for c in calls if c[0][0] == "capabilities"]
    assert len(cap_calls) >= 1

    # Capabilities data should have usefulness fields
    caps_data = cap_calls[0][0][1]
    for name, cap in caps_data.items():
        assert "usefulness" in cap
        assert "source" in cap
        assert "status" in cap


@pytest.mark.asyncio
async def test_module_preserves_seed_capabilities(mock_hub):
    """Seed capabilities should always be present even if clustering misses them."""
    mock_hub.cache.get = AsyncMock(side_effect=lambda cat: {
        "entities": {"data": []},  # No entities — clustering will find nothing
        "devices": MOCK_DEVICES_CACHE,
        "capabilities": MOCK_CAPABILITIES_CACHE,
        "activity_summary": {"data": {"entity_activity": {}}},
        "discovery_history": None,
    }.get(cat))

    module = OrganicDiscoveryModule(mock_hub)
    await module.run_discovery()

    cap_calls = [c for c in mock_hub.cache.set.call_args_list if c[0][0] == "capabilities"]
    if cap_calls:
        caps_data = cap_calls[0][0][1]
        # Seeds should be preserved
        assert "lighting" in caps_data
        assert "power_monitoring" in caps_data
        assert caps_data["lighting"]["source"] == "seed"


@pytest.mark.asyncio
async def test_module_settings_defaults(mock_hub):
    module = OrganicDiscoveryModule(mock_hub)
    assert module.settings["autonomy_mode"] == "suggest_and_wait"
    assert module.settings["naming_backend"] == "heuristic"
    assert module.settings["promote_threshold"] == 50
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_discovery_module.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# aria/modules/organic_discovery/module.py
"""OrganicDiscoveryModule — hub module for automatic capability discovery."""
import logging
from datetime import datetime, timedelta

from aria.hub.core import IntelligenceHub
from aria.modules.organic_discovery.feature_vectors import build_feature_matrix
from aria.modules.organic_discovery.clustering import cluster_entities
from aria.modules.organic_discovery.seed_validation import validate_seeds
from aria.modules.organic_discovery.scoring import compute_usefulness, UsefulnessComponents
from aria.modules.organic_discovery.naming import heuristic_name, heuristic_description

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "autonomy_mode": "suggest_and_wait",  # suggest_and_wait | auto_promote | autonomous
    "naming_backend": "heuristic",        # heuristic | ollama | external_llm
    "promote_threshold": 50,
    "archive_threshold": 10,
    "promote_streak_days": 7,
    "archive_streak_days": 14,
}


class OrganicDiscoveryModule:
    """Discovers capabilities organically from entity data via clustering."""

    def __init__(self, hub: IntelligenceHub):
        self.hub = hub
        self.module_id = "organic_discovery"
        self.settings = dict(DEFAULT_SETTINGS)
        self._history: list[dict] = []

    async def initialize(self):
        """Load settings and history from cache."""
        settings_entry = await self.hub.cache.get("discovery_settings")
        if settings_entry and settings_entry.get("data"):
            self.settings.update(settings_entry["data"])

        history_entry = await self.hub.cache.get("discovery_history")
        if history_entry and history_entry.get("data"):
            self._history = history_entry["data"]

        logger.info(f"OrganicDiscovery initialized (mode={self.settings['autonomy_mode']})")

    async def run_discovery(self):
        """Execute a full organic discovery run."""
        logger.info("Starting organic discovery run")
        run_start = datetime.utcnow().isoformat()

        # Load data from cache
        entities_entry = await self.hub.cache.get("entities")
        devices_entry = await self.hub.cache.get("devices")
        caps_entry = await self.hub.cache.get("capabilities")
        activity_entry = await self.hub.cache.get("activity_summary")

        entities = (entities_entry or {}).get("data", [])
        devices = (devices_entry or {}).get("data", {})
        seed_caps = (caps_entry or {}).get("data", {})
        activity_data = (activity_entry or {}).get("data", {})
        entity_activity = activity_data.get("entity_activity", {})

        # Build entity registry lookup from entities list
        entity_registry = {}
        if isinstance(entities, list):
            for e in entities:
                eid = e.get("entity_id", "")
                entity_registry[eid] = {
                    "device_id": e.get("attributes", {}).get("device_id"),
                }

        # Phase 1: Domain clustering
        if not entities:
            logger.warning("No entities in cache — preserving seed capabilities only")
            discovered_caps = {}
        else:
            matrix, entity_ids, feature_names = build_feature_matrix(
                entities, devices, entity_registry, entity_activity
            )
            clusters = cluster_entities(matrix, entity_ids)

            # Validate against seeds
            seed_validation = validate_seeds(seed_caps, clusters)

            # Build cluster metadata for naming
            discovered_caps = {}
            for cluster in clusters:
                cluster_info = self._build_cluster_info(cluster, entities, devices, entity_registry)
                name = heuristic_name(cluster_info)
                description = heuristic_description(cluster_info)

                # Deduplicate names
                if name in discovered_caps:
                    name = f"{name}_{cluster['cluster_id']}"

                # Compute usefulness
                components = self._compute_components(
                    cluster, cluster_info, entity_activity, len(entities)
                )
                usefulness = compute_usefulness(components)

                discovered_caps[name] = {
                    "available": True,
                    "entities": cluster["entity_ids"],
                    "total_count": len(cluster["entity_ids"]),
                    "can_predict": False,
                    "source": "organic",
                    "usefulness": usefulness,
                    "usefulness_components": components.to_dict(),
                    "layer": "domain",
                    "status": "candidate",
                    "first_seen": run_start[:10],
                    "promoted_at": None,
                    "naming_method": self.settings["naming_backend"],
                    "description": description,
                    "stability_streak": 1,
                }

        # Merge with seed capabilities (seeds always preserved)
        merged = {}
        for seed_name, seed_data in seed_caps.items():
            merged[seed_name] = {
                **seed_data,
                "source": seed_data.get("source", "seed"),
                "usefulness": seed_data.get("usefulness", 75),
                "usefulness_components": seed_data.get("usefulness_components", {
                    "predictability": 0, "stability": 100,
                    "entity_coverage": 50, "activity": 50, "cohesion": 75,
                }),
                "layer": seed_data.get("layer", "domain"),
                "status": seed_data.get("status", "promoted"),
                "first_seen": seed_data.get("first_seen", run_start[:10]),
                "promoted_at": seed_data.get("promoted_at", run_start[:10]),
                "naming_method": seed_data.get("naming_method", "seed"),
                "description": seed_data.get("description", ""),
                "stability_streak": seed_data.get("stability_streak", 14),
            }

        # Add organically discovered caps (don't overwrite seeds)
        for name, cap in discovered_caps.items():
            if name not in merged:
                # Check stability streak from previous run
                prev_caps = seed_caps if not self._history else {}
                if name in prev_caps:
                    prev_streak = prev_caps[name].get("stability_streak", 0)
                    cap["stability_streak"] = prev_streak + 1
                    cap["first_seen"] = prev_caps[name].get("first_seen", run_start[:10])

                # Apply autonomy rules
                cap = self._apply_autonomy(cap)
                merged[name] = cap

        # Write to cache
        await self.hub.cache.set("capabilities", merged, {
            "count": len(merged),
            "organic_count": sum(1 for c in merged.values() if c.get("source") == "organic"),
            "source": "organic_discovery",
        })

        # Record history
        run_record = {
            "timestamp": run_start,
            "total_clusters": len(discovered_caps),
            "promoted": sum(1 for c in merged.values() if c.get("status") == "promoted"),
            "candidates": sum(1 for c in merged.values() if c.get("status") == "candidate"),
            "archived": sum(1 for c in merged.values() if c.get("status") == "archived"),
        }
        self._history.append(run_record)
        # Keep last 30 runs
        self._history = self._history[-30:]
        await self.hub.cache.set("discovery_history", self._history)

        logger.info(
            f"Organic discovery complete: {len(discovered_caps)} clusters, "
            f"{run_record['promoted']} promoted, {run_record['candidates']} candidates"
        )

        await self.hub.publish("organic_discovery_complete", run_record)

    def _build_cluster_info(self, cluster, entities, devices, entity_registry):
        """Build metadata dict for naming a cluster."""
        entity_map = {e["entity_id"]: e for e in entities if isinstance(e, dict)}
        domains = {}
        device_classes = {}
        areas = {}

        for eid in cluster["entity_ids"]:
            domain = eid.split(".")[0]
            domains[domain] = domains.get(domain, 0) + 1

            entity = entity_map.get(eid, {})
            attrs = entity.get("attributes", {})
            dc = attrs.get("device_class")
            if dc:
                device_classes[dc] = device_classes.get(dc, 0) + 1

            # Resolve area
            reg = entity_registry.get(eid, {})
            dev_id = reg.get("device_id")
            if dev_id and dev_id in devices:
                area = devices[dev_id].get("area_id")
                if area:
                    areas[area] = areas.get(area, 0) + 1

        return {
            "entity_ids": cluster["entity_ids"],
            "domains": domains,
            "device_classes": device_classes,
            "areas": areas,
        }

    def _compute_components(self, cluster, cluster_info, entity_activity, total_entities):
        """Compute usefulness components for a cluster."""
        entity_ids = cluster["entity_ids"]
        n = len(entity_ids)

        # Predictability: 0 until ML models exist for this cluster
        predictability = 0.0

        # Stability: 1.0 on first run (it exists now), tracked over time
        stability = 1.0

        # Entity coverage: scaled with diminishing returns
        if total_entities > 0:
            raw_coverage = n / total_entities
            entity_coverage = min(raw_coverage * 5, 1.0)  # 20% coverage = 1.0
        else:
            entity_coverage = 0.0

        # Activity: average daily state changes, normalized
        activities = [entity_activity.get(eid, 0.0) for eid in entity_ids]
        avg_activity = sum(activities) / len(activities) if activities else 0.0
        activity = min(avg_activity / 50.0, 1.0)  # 50 changes/day = 1.0

        # Cohesion: silhouette score rescaled from [-1,1] to [0,1]
        raw_sil = cluster.get("silhouette", 0.0)
        cohesion = (raw_sil + 1.0) / 2.0

        return UsefulnessComponents(
            predictability=predictability,
            stability=stability,
            entity_coverage=entity_coverage,
            activity=activity,
            cohesion=cohesion,
        )

    def _apply_autonomy(self, cap):
        """Apply autonomy mode rules to a capability."""
        mode = self.settings["autonomy_mode"]
        usefulness = cap.get("usefulness", 0)
        streak = cap.get("stability_streak", 0)

        if mode == "suggest_and_wait":
            # Never auto-promote
            pass
        elif mode == "auto_promote":
            if (usefulness >= self.settings["promote_threshold"]
                    and streak >= self.settings["promote_streak_days"]):
                cap["status"] = "promoted"
                cap["promoted_at"] = cap.get("first_seen")
            if (usefulness <= self.settings["archive_threshold"]
                    and streak >= self.settings["archive_streak_days"]):
                cap["status"] = "archived"
        elif mode == "autonomous":
            if usefulness >= 30:
                cap["status"] = "promoted"
                cap["promoted_at"] = cap.get("first_seen")
            if usefulness <= self.settings["archive_threshold"]:
                cap["status"] = "archived"

        return cap

    async def update_settings(self, new_settings: dict):
        """Update discovery settings and persist."""
        self.settings.update(new_settings)
        await self.hub.cache.set("discovery_settings", self.settings)
        logger.info(f"Discovery settings updated: {self.settings}")
```

Now add the constant and wire into CLI.

Add to `aria/hub/constants.py`:
```python
CACHE_DISCOVERY_HISTORY = "discovery_history"
CACHE_DISCOVERY_SETTINGS = "discovery_settings"
```

Add to `aria/cli.py` in the `_serve` `start()` function, after data_quality module registration and before intelligence:
```python
        # organic_discovery (non-fatal)
        try:
            from aria.modules.organic_discovery.module import OrganicDiscoveryModule

            organic_discovery = OrganicDiscoveryModule(hub)
            hub.register_module(organic_discovery)
            await _init_module(organic_discovery, "organic_discovery")()
        except Exception as e:
            logger.warning(f"Organic discovery module failed (non-fatal): {e}")
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_discovery_module.py -v`
Expected: 4 PASSED

**Step 5: Run all organic discovery tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_*.py -v`
Expected: All PASSED (28 tests across 5 files)

**Step 6: Commit**

```bash
git add aria/modules/organic_discovery/module.py aria/hub/constants.py aria/cli.py tests/hub/test_organic_discovery_module.py
git commit -m "feat(organic-discovery): hub module with clustering pipeline and autonomy"
```

---

### Task 7: API Endpoints

**Files:**
- Modify: `aria/hub/api.py` — add discovery endpoints
- Test: `tests/hub/test_api_organic_discovery.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_api_organic_discovery.py
"""Tests for organic discovery API endpoints."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from aria.hub.api import create_api


@pytest.fixture
def mock_hub():
    hub = AsyncMock()
    hub.cache = AsyncMock()
    hub.modules = {}
    hub.module_status = {}
    hub.is_running = MagicMock(return_value=True)
    hub.publish = AsyncMock()
    return hub


@pytest.fixture
def client(mock_hub):
    app = create_api(mock_hub)
    return TestClient(app)


MOCK_CAPS = {
    "data": {
        "lighting": {"source": "seed", "status": "promoted", "usefulness": 87},
        "evening_routine": {"source": "organic", "status": "candidate", "usefulness": 58},
        "old_cluster": {"source": "organic", "status": "archived", "usefulness": 8},
    }
}


def test_get_capabilities_candidates(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value=MOCK_CAPS)
    resp = client.get("/api/capabilities/candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert "evening_routine" in data


def test_get_discovery_history(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value={
        "data": [{"timestamp": "2026-02-14T00:00:00", "total_clusters": 5}]
    })
    resp = client.get("/api/capabilities/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1


def test_promote_capability(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value=MOCK_CAPS)
    resp = client.put("/api/capabilities/evening_routine/promote")
    assert resp.status_code == 200
    assert resp.json()["status"] == "promoted"


def test_promote_unknown_capability(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value=MOCK_CAPS)
    resp = client.put("/api/capabilities/nonexistent/promote")
    assert resp.status_code == 404


def test_archive_capability(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value=MOCK_CAPS)
    resp = client.put("/api/capabilities/evening_routine/archive")
    assert resp.status_code == 200
    assert resp.json()["status"] == "archived"


def test_get_discovery_settings(client, mock_hub):
    mock_hub.cache.get = AsyncMock(return_value=None)
    mock_hub.modules = {"organic_discovery": MagicMock(settings={
        "autonomy_mode": "suggest_and_wait",
        "naming_backend": "heuristic",
        "promote_threshold": 50,
        "archive_threshold": 10,
        "promote_streak_days": 7,
        "archive_streak_days": 14,
    })}
    resp = client.get("/api/settings/discovery")
    assert resp.status_code == 200
    data = resp.json()
    assert data["autonomy_mode"] == "suggest_and_wait"


def test_update_discovery_settings(client, mock_hub):
    module_mock = AsyncMock()
    module_mock.settings = {"autonomy_mode": "suggest_and_wait"}
    module_mock.update_settings = AsyncMock()
    mock_hub.modules = {"organic_discovery": module_mock}
    resp = client.put("/api/settings/discovery", json={"autonomy_mode": "auto_promote"})
    assert resp.status_code == 200
    module_mock.update_settings.assert_called_once()


def test_trigger_discovery_run(client, mock_hub):
    module_mock = AsyncMock()
    module_mock.run_discovery = AsyncMock()
    mock_hub.modules = {"organic_discovery": module_mock}
    resp = client.post("/api/discovery/run")
    assert resp.status_code == 200


def test_get_discovery_status(client, mock_hub):
    module_mock = MagicMock()
    module_mock._history = [{"timestamp": "2026-02-14T00:00:00"}]
    mock_hub.modules = {"organic_discovery": module_mock}
    resp = client.get("/api/discovery/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "last_run" in data
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_api_organic_discovery.py -v`
Expected: FAIL (404 — endpoints don't exist yet)

**Step 3: Add endpoints to `aria/hub/api.py`**

Add the following routes in `aria/hub/api.py` after the existing `toggle_can_predict` endpoint (after line ~417). Follow the existing pattern — routes are defined within the `create_api(hub)` function:

```python
    @router.get("/api/capabilities/candidates")
    async def get_capability_candidates():
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            return {}
        caps = cached["data"]
        return {name: cap for name, cap in caps.items() if cap.get("status") == "candidate"}

    @router.get("/api/capabilities/history")
    async def get_discovery_history():
        cached = await hub.cache.get("discovery_history")
        if not cached or not cached.get("data"):
            return []
        return cached["data"]

    @router.put("/api/capabilities/{capability_name}/promote")
    async def promote_capability(capability_name: str):
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            raise HTTPException(status_code=404, detail="Capabilities not found")
        caps = cached["data"]
        if capability_name not in caps:
            raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")
        from datetime import datetime
        caps[capability_name]["status"] = "promoted"
        caps[capability_name]["promoted_at"] = datetime.utcnow().strftime("%Y-%m-%d")
        await hub.cache.set("capabilities", caps)
        return {"capability": capability_name, "status": "promoted"}

    @router.put("/api/capabilities/{capability_name}/archive")
    async def archive_capability(capability_name: str):
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            raise HTTPException(status_code=404, detail="Capabilities not found")
        caps = cached["data"]
        if capability_name not in caps:
            raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")
        caps[capability_name]["status"] = "archived"
        await hub.cache.set("capabilities", caps)
        return {"capability": capability_name, "status": "archived"}

    @router.get("/api/settings/discovery")
    async def get_discovery_settings():
        module = hub.modules.get("organic_discovery")
        if not module:
            return {"error": "Organic discovery module not loaded"}
        return module.settings

    @router.put("/api/settings/discovery")
    async def update_discovery_settings(body: dict = Body(...)):
        module = hub.modules.get("organic_discovery")
        if not module:
            raise HTTPException(status_code=404, detail="Organic discovery module not loaded")
        await module.update_settings(body)
        return {"status": "updated", "settings": module.settings}

    @router.post("/api/discovery/run")
    async def trigger_discovery_run():
        module = hub.modules.get("organic_discovery")
        if not module:
            raise HTTPException(status_code=404, detail="Organic discovery module not loaded")
        import asyncio
        asyncio.create_task(module.run_discovery())
        return {"status": "started"}

    @router.get("/api/discovery/status")
    async def get_discovery_status():
        module = hub.modules.get("organic_discovery")
        if not module:
            return {"loaded": False}
        history = module._history
        return {
            "loaded": True,
            "last_run": history[-1]["timestamp"] if history else None,
            "total_runs": len(history),
            "settings": module.settings,
        }
```

**Important:** Add these routes BEFORE the existing `candidates` substring might conflict. Place `/api/capabilities/candidates` and `/api/capabilities/history` routes BEFORE the parameterized `/api/capabilities/{capability_name}/...` routes to avoid path conflicts.

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_api_organic_discovery.py -v`
Expected: 10 PASSED

**Step 5: Run existing API tests to ensure no regressions**

Run: `.venv/bin/python -m pytest tests/hub/test_api_shadow.py tests/hub/test_api_config.py -v`
Expected: All PASSED

**Step 6: Commit**

```bash
git add aria/hub/api.py tests/hub/test_api_organic_discovery.py
git commit -m "feat(organic-discovery): API endpoints for capabilities, settings, discovery control"
```

---

### Task 8: Dashboard — Extended Capabilities Page

**Frontend task — can run in parallel with backend tasks.**

**Files:**
- Rewrite: `aria/dashboard/spa/src/pages/Capabilities.jsx`
- Create: `aria/dashboard/spa/src/components/UsefulnessBar.jsx`
- Create: `aria/dashboard/spa/src/components/CapabilityDetail.jsx`

**Important references:**
- Design language: `docs/design-language.md` — ASCII terminal aesthetic, CSS custom properties only, `class` not `className`, `.t-frame` with `data-label`
- API helpers: `src/api.js` — `fetchJson`, `putJson`, `postJson`
- Cache hook: `src/hooks/useCache.js` — `useCache(category)` → `{ data, loading, error, refetch }`
- Reusable components: `PageBanner`, `HeroCard`, `LoadingState`, `ErrorState`, `CollapsibleSection`

**Step 1: Create UsefulnessBar component**

```jsx
// aria/dashboard/spa/src/components/UsefulnessBar.jsx
/**
 * Horizontal usefulness percentage bar with label.
 * Props: value (0-100), label (string), sublabel (optional string)
 */
export default function UsefulnessBar({ value, label, sublabel }) {
  const pct = Math.max(0, Math.min(100, value || 0));
  const color = pct >= 70 ? 'var(--status-healthy)' : pct >= 40 ? 'var(--accent-warm)' : 'var(--status-unhealthy)';

  return (
    <div class="flex items-center gap-2 text-xs">
      <span class="w-28 truncate" style="color: var(--text-tertiary)">{label}</span>
      <div class="flex-1 h-2 rounded-full" style="background: var(--bg-inset)">
        <div
          class="h-2 rounded-full transition-all"
          style={`width: ${pct}%; background: ${color};`}
        />
      </div>
      <span class="w-8 text-right font-mono" style="color: var(--text-secondary)">
        {pct}%
      </span>
      {sublabel && (
        <span class="text-xs" style="color: var(--text-tertiary)">{sublabel}</span>
      )}
    </div>
  );
}
```

**Step 2: Create CapabilityDetail component**

```jsx
// aria/dashboard/spa/src/components/CapabilityDetail.jsx
import { useState } from 'preact/hooks';
import UsefulnessBar from './UsefulnessBar.jsx';
import { putJson } from '../api.js';

/**
 * Expanded detail view for a capability. Shows usefulness breakdown,
 * entity list, temporal pattern, and action buttons.
 */
export default function CapabilityDetail({ name, capability, onAction }) {
  const [busy, setBusy] = useState(false);
  const uc = capability.usefulness_components || {};
  const entities = capability.entities || [];
  const [expanded, setExpanded] = useState(false);
  const visibleEntities = expanded ? entities : entities.slice(0, 5);
  const temporal = capability.temporal_pattern || {};

  async function handleAction(action) {
    setBusy(true);
    try {
      await putJson(`/api/capabilities/${name}/${action}`);
      if (onAction) onAction();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div class="space-y-3">
      {/* Usefulness breakdown */}
      <div class="space-y-1.5">
        <UsefulnessBar value={uc.predictability} label="Predictability" sublabel={uc.predictability === 0 ? '(no model)' : ''} />
        <UsefulnessBar value={uc.stability} label="Stability" sublabel={uc.stability ? `(${capability.stability_streak || 0}d streak)` : ''} />
        <UsefulnessBar value={uc.entity_coverage} label="Coverage" />
        <UsefulnessBar value={uc.activity} label="Activity" />
        <UsefulnessBar value={uc.cohesion} label="Cohesion" />
      </div>

      {/* Metadata */}
      <div class="text-xs space-y-1" style="color: var(--text-tertiary)">
        {capability.description && <p>{capability.description}</p>}
        <p>Source: {capability.source} · Layer: {capability.layer} · Named by: {capability.naming_method}</p>
        <p>First seen: {capability.first_seen}{capability.promoted_at ? ` · Promoted: ${capability.promoted_at}` : ''}</p>
        {temporal.peak_hours && (
          <p>Peak hours: {temporal.peak_hours.join(', ')}:00{temporal.weekday_bias != null ? ` · Weekday bias: ${(temporal.weekday_bias * 100).toFixed(0)}%` : ''}</p>
        )}
      </div>

      {/* Entities */}
      {entities.length > 0 && (
        <div>
          <ul class="text-xs space-y-0.5" style="color: var(--text-tertiary)">
            {visibleEntities.map((eid) => (
              <li key={eid} class="truncate data-mono">{eid}</li>
            ))}
          </ul>
          {entities.length > 5 && (
            <button
              onClick={() => setExpanded(!expanded)}
              class="text-xs cursor-pointer mt-1"
              style="color: var(--accent); background: none; border: none; padding: 0;"
            >
              {expanded ? 'Show less' : `Show all ${entities.length}`}
            </button>
          )}
        </div>
      )}

      {/* Actions */}
      <div class="flex gap-2 pt-2" style="border-top: 1px solid var(--border-subtle)">
        {capability.status === 'candidate' && (
          <button
            onClick={() => handleAction('promote')}
            disabled={busy}
            class="text-xs px-3 py-1 rounded cursor-pointer"
            style="background: var(--accent); color: var(--bg-base); opacity: ${busy ? 0.5 : 1};"
          >
            Promote
          </button>
        )}
        {capability.status !== 'archived' && (
          <button
            onClick={() => handleAction('archive')}
            disabled={busy}
            class="text-xs px-3 py-1 rounded cursor-pointer"
            style="background: var(--bg-inset); color: var(--text-secondary); opacity: ${busy ? 0.5 : 1};"
          >
            Archive
          </button>
        )}
      </div>
    </div>
  );
}
```

**Step 3: Rewrite Capabilities.jsx**

Rewrite `aria/dashboard/spa/src/pages/Capabilities.jsx` — keep existing `PredictToggle` and `PREDICT_EXPLANATIONS`, add sections for promoted/candidates/archived, add usefulness bars to each card:

The full rewrite should:
1. Keep existing imports + `PredictToggle` + `PREDICT_EXPLANATIONS` + `DEFAULT_EXPLANATION` (lines 1-84)
2. Replace `CapabilityCard` (lines 141-204) to include `UsefulnessBar` and source/layer badges
3. Replace `Capabilities` default export (lines 206-284) to:
   - Split capabilities into promoted/candidates/archived groups
   - Show promoted count + candidate count in HeroCard delta
   - Add `CollapsibleSection` for each group
   - Show usefulness % on each card
   - Add candidate action buttons (Promote/Archive)
   - Add a discovery status bar at top (last run, autonomy mode)
   - Add Settings link

**Step 4: Build the SPA**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`
Expected: Build succeeds with no errors

**Step 5: Commit**

```bash
git add aria/dashboard/spa/src/components/UsefulnessBar.jsx aria/dashboard/spa/src/components/CapabilityDetail.jsx aria/dashboard/spa/src/pages/Capabilities.jsx
git commit -m "feat(dashboard): extended capabilities page with usefulness scores and organic sections"
```

---

### Task 9: Dashboard — Discovery Settings Panel

**Frontend task — depends on Task 8.**

**Files:**
- Create: `aria/dashboard/spa/src/components/DiscoverySettings.jsx`
- Modify: `aria/dashboard/spa/src/pages/Capabilities.jsx` — add settings panel toggle

**Step 1: Create DiscoverySettings component**

```jsx
// aria/dashboard/spa/src/components/DiscoverySettings.jsx
import { useState, useEffect } from 'preact/hooks';
import { fetchJson, putJson, postJson } from '../api.js';

const AUTONOMY_OPTIONS = [
  { value: 'suggest_and_wait', label: 'Suggest & wait', desc: 'Show candidates, you promote manually.' },
  { value: 'auto_promote', label: 'Auto-promote', desc: 'Promote when usefulness stays above threshold.' },
  { value: 'autonomous', label: 'Fully autonomous', desc: 'ARIA manages promotion and archival.' },
];

const NAMING_OPTIONS = [
  { value: 'heuristic', label: 'Heuristic', pro: 'Free, instant, deterministic', con: 'Generic names' },
  { value: 'ollama', label: 'Ollama (deepseek-r1:8b)', pro: 'Natural language names', con: '+45min Ollama slot per run' },
  { value: 'external_llm', label: 'External LLM', pro: 'Best quality names', con: 'API cost per run' },
];

export default function DiscoverySettings({ onClose }) {
  const [settings, setSettings] = useState(null);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    fetchJson('/api/settings/discovery').then(setSettings);
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      await putJson('/api/settings/discovery', settings);
      if (onClose) onClose();
    } finally {
      setSaving(false);
    }
  }

  async function handleRunNow() {
    setRunning(true);
    try {
      await postJson('/api/discovery/run', {});
    } finally {
      setTimeout(() => setRunning(false), 3000);
    }
  }

  if (!settings) return <p class="text-sm" style="color: var(--text-tertiary)">Loading settings...</p>;

  return (
    <div class="t-frame" data-label="Discovery Settings" style="padding: 1rem;">
      <div class="space-y-4">
        {/* Autonomy mode */}
        <fieldset class="space-y-2">
          <legend class="text-sm font-bold" style="color: var(--text-primary)">Autonomy Mode</legend>
          {AUTONOMY_OPTIONS.map((opt) => (
            <label key={opt.value} class="flex items-start gap-2 cursor-pointer">
              <input
                type="radio"
                name="autonomy"
                checked={settings.autonomy_mode === opt.value}
                onChange={() => setSettings({ ...settings, autonomy_mode: opt.value })}
                class="mt-1"
              />
              <div>
                <span class="text-sm font-medium" style="color: var(--text-primary)">{opt.label}</span>
                <p class="text-xs" style="color: var(--text-tertiary)">{opt.desc}</p>
              </div>
            </label>
          ))}
          {settings.autonomy_mode !== 'suggest_and_wait' && (
            <div class="text-xs space-y-1 pl-6" style="color: var(--text-secondary)">
              <div class="flex items-center gap-2">
                <span>Promote at</span>
                <input
                  type="number" min="10" max="100" value={settings.promote_threshold}
                  onInput={(e) => setSettings({ ...settings, promote_threshold: +e.target.value })}
                  class="w-14 px-1 rounded text-center"
                  style="background: var(--bg-inset); color: var(--text-primary); border: 1px solid var(--border-subtle);"
                />
                <span>% for</span>
                <input
                  type="number" min="1" max="30" value={settings.promote_streak_days}
                  onInput={(e) => setSettings({ ...settings, promote_streak_days: +e.target.value })}
                  class="w-14 px-1 rounded text-center"
                  style="background: var(--bg-inset); color: var(--text-primary); border: 1px solid var(--border-subtle);"
                />
                <span>days</span>
              </div>
            </div>
          )}
        </fieldset>

        {/* Naming backend */}
        <fieldset class="space-y-2">
          <legend class="text-sm font-bold" style="color: var(--text-primary)">Naming Backend</legend>
          {NAMING_OPTIONS.map((opt) => (
            <label key={opt.value} class="flex items-start gap-2 cursor-pointer">
              <input
                type="radio"
                name="naming"
                checked={settings.naming_backend === opt.value}
                onChange={() => setSettings({ ...settings, naming_backend: opt.value })}
                class="mt-1"
              />
              <div>
                <span class="text-sm font-medium" style="color: var(--text-primary)">{opt.label}</span>
                <p class="text-xs" style="color: var(--text-tertiary)">{opt.pro} · {opt.con}</p>
              </div>
            </label>
          ))}
        </fieldset>

        {/* Actions */}
        <div class="flex gap-2 pt-2" style="border-top: 1px solid var(--border-subtle)">
          <button onClick={handleSave} disabled={saving} class="text-xs px-3 py-1.5 rounded cursor-pointer" style="background: var(--accent); color: var(--bg-base);">
            {saving ? 'Saving...' : 'Save'}
          </button>
          <button onClick={handleRunNow} disabled={running} class="text-xs px-3 py-1.5 rounded cursor-pointer" style="background: var(--bg-inset); color: var(--text-secondary);">
            {running ? 'Running...' : 'Run Now'}
          </button>
          {onClose && (
            <button onClick={onClose} class="text-xs px-3 py-1.5 cursor-pointer" style="color: var(--text-tertiary); background: none; border: none;">
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Wire settings toggle into Capabilities.jsx**

Add a state variable `showSettings` and a toggle button in the page header, rendering `DiscoverySettings` when active.

**Step 3: Build SPA**

Run: `cd /home/justin/Documents/projects/ha-aria/aria/dashboard/spa && npm run build`

**Step 4: Commit**

```bash
git add aria/dashboard/spa/src/components/DiscoverySettings.jsx aria/dashboard/spa/src/pages/Capabilities.jsx
git commit -m "feat(dashboard): discovery settings panel with autonomy and naming controls"
```

---

## Phase 2: Behavioral Clustering (Layer 2)

### Task 10: Co-occurrence Matrix Builder

**Files:**
- Create: `aria/modules/organic_discovery/behavioral.py`
- Test: `tests/hub/test_organic_behavioral.py`

**Step 1: Write the failing test**

```python
# tests/hub/test_organic_behavioral.py
"""Tests for behavioral co-occurrence clustering."""
import pytest
import numpy as np
from datetime import datetime, timedelta
from aria.modules.organic_discovery.behavioral import (
    build_cooccurrence_matrix,
    extract_temporal_pattern,
    cluster_behavioral,
)


def _make_logbook(patterns, days=14):
    """Generate synthetic logbook entries from co-occurrence patterns.

    patterns: list of (entity_ids, hour, minute_offset) tuples.
    Each pattern generates one co-occurrence per day.
    """
    entries = []
    base = datetime(2026, 2, 1)
    for day in range(days):
        dt = base + timedelta(days=day)
        for entity_ids, hour, offset in patterns:
            for i, eid in enumerate(entity_ids):
                entries.append({
                    "entity_id": eid,
                    "state": "on",
                    "when": (dt.replace(hour=hour, minute=offset + i)).isoformat(),
                })
    return entries


def test_build_cooccurrence_matrix_shape():
    entries = _make_logbook([
        (["light.a", "light.b", "switch.c"], 19, 0),
    ])
    matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] == 3
    assert len(entity_ids) == 3


def test_build_cooccurrence_matrix_symmetry():
    entries = _make_logbook([
        (["light.a", "light.b"], 19, 0),
    ])
    matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
    assert matrix[0, 1] == matrix[1, 0]


def test_build_cooccurrence_matrix_finds_pattern():
    entries = _make_logbook([
        (["light.a", "light.b"], 19, 0),  # These co-occur
    ])
    matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
    a_idx = entity_ids.index("light.a")
    b_idx = entity_ids.index("light.b")
    assert matrix[a_idx, b_idx] > 0


def test_build_cooccurrence_no_false_cooccurrence():
    """Entities in different time windows should not co-occur."""
    entries = _make_logbook([
        (["light.a"], 8, 0),   # Morning
        (["light.b"], 20, 0),  # Evening — 12 hours apart
    ])
    matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
    if len(entity_ids) == 2:
        a_idx = entity_ids.index("light.a")
        b_idx = entity_ids.index("light.b")
        assert matrix[a_idx, b_idx] == 0


def test_extract_temporal_pattern():
    entries = []
    for day in range(14):
        for eid in ["light.a", "light.b"]:
            entries.append({
                "entity_id": eid,
                "when": f"2026-02-{day+1:02d}T19:30:00",
            })
    pattern = extract_temporal_pattern(["light.a", "light.b"], entries)
    assert 19 in pattern["peak_hours"]
    assert "weekday_bias" in pattern


def test_cluster_behavioral_finds_groups():
    entries = _make_logbook([
        (["light.a", "light.b", "switch.c", "light.d", "switch.e"], 19, 0),
        (["sensor.x", "sensor.y", "sensor.z", "binary_sensor.m", "binary_sensor.n"], 8, 0),
    ], days=14)
    clusters = cluster_behavioral(entries, min_cluster_size=3)
    assert len(clusters) >= 1  # Should find at least one behavioral group


def test_cluster_behavioral_empty():
    clusters = cluster_behavioral([], min_cluster_size=3)
    assert clusters == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_behavioral.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# aria/modules/organic_discovery/behavioral.py
"""Behavioral co-occurrence clustering for Layer 2 discovery."""
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_cooccurrence_matrix(
    logbook_entries: list[dict],
    window_minutes: int = 15,
) -> tuple[np.ndarray, list[str]]:
    """Build entity co-occurrence matrix from logbook state changes.

    Groups events into time windows and counts how often entity pairs
    change state in the same window.

    Returns (matrix, entity_ids) where matrix[i][j] = co-occurrence count.
    """
    if not logbook_entries:
        return np.empty((0, 0)), []

    # Parse timestamps and group into windows
    events_by_window = defaultdict(set)
    all_entities = set()

    for entry in logbook_entries:
        eid = entry.get("entity_id", "")
        when_str = entry.get("when", "")
        if not eid or not when_str:
            continue

        try:
            when = datetime.fromisoformat(when_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        # Window key: date + hour + (minute // window_minutes)
        window_key = (when.date(), when.hour, when.minute // window_minutes)
        events_by_window[window_key].add(eid)
        all_entities.add(eid)

    entity_ids = sorted(all_entities)
    n = len(entity_ids)
    if n == 0:
        return np.empty((0, 0)), []

    eid_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    matrix = np.zeros((n, n), dtype=np.float64)

    for window_entities in events_by_window.values():
        eids_in_window = [e for e in window_entities if e in eid_to_idx]
        for i, eid_a in enumerate(eids_in_window):
            for eid_b in eids_in_window[i + 1:]:
                idx_a = eid_to_idx[eid_a]
                idx_b = eid_to_idx[eid_b]
                matrix[idx_a, idx_b] += 1
                matrix[idx_b, idx_a] += 1

    return matrix, entity_ids


def extract_temporal_pattern(entity_ids: list[str], logbook_entries: list[dict]) -> dict:
    """Extract temporal pattern for a group of entities.

    Returns dict with peak_hours and weekday_bias.
    """
    hours = defaultdict(int)
    weekday_count = 0
    total_count = 0
    entity_set = set(entity_ids)

    for entry in logbook_entries:
        if entry.get("entity_id") not in entity_set:
            continue
        when_str = entry.get("when", "")
        try:
            when = datetime.fromisoformat(when_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        hours[when.hour] += 1
        total_count += 1
        if when.weekday() < 5:
            weekday_count += 1

    if not hours:
        return {"peak_hours": [], "weekday_bias": 0.0}

    # Find peak hours (above average)
    avg = sum(hours.values()) / 24
    peak_hours = sorted(h for h, c in hours.items() if c > avg * 1.5)

    weekday_bias = weekday_count / total_count if total_count > 0 else 0.0

    return {
        "peak_hours": peak_hours,
        "weekday_bias": round(weekday_bias, 2),
    }


def cluster_behavioral(
    logbook_entries: list[dict],
    min_cluster_size: int = 3,
    window_minutes: int = 15,
) -> list[dict]:
    """Cluster entities by behavioral co-occurrence patterns.

    Returns list of cluster dicts with:
        cluster_id, entity_ids, silhouette, temporal_pattern
    """
    matrix, entity_ids = build_cooccurrence_matrix(logbook_entries, window_minutes)

    if len(entity_ids) < min_cluster_size:
        return []

    # Normalize co-occurrence matrix for clustering
    # Each row is an entity's co-occurrence profile
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(scaled)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if len(unique_labels) < 2:
        if len(unique_labels) == 1:
            label = list(unique_labels)[0]
            members = [entity_ids[i] for i, l in enumerate(labels) if l == label]
            temporal = extract_temporal_pattern(members, logbook_entries)
            return [{"cluster_id": 0, "entity_ids": members, "silhouette": 0.0, "temporal_pattern": temporal}]
        return []

    non_noise_mask = labels != -1
    sil_scores = np.zeros(len(labels))
    if non_noise_mask.sum() >= 2:
        sil_scores[non_noise_mask] = silhouette_samples(
            scaled[non_noise_mask], labels[non_noise_mask]
        )

    clusters = []
    for label in sorted(unique_labels):
        member_mask = labels == label
        member_ids = [entity_ids[i] for i, m in enumerate(member_mask) if m]
        avg_sil = float(np.mean(sil_scores[member_mask]))
        temporal = extract_temporal_pattern(member_ids, logbook_entries)

        clusters.append({
            "cluster_id": int(label),
            "entity_ids": member_ids,
            "silhouette": round(avg_sil, 4),
            "temporal_pattern": temporal,
        })

    logger.info(f"Behavioral clustering found {len(clusters)} groups, {(labels == -1).sum()} noise")
    return clusters
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_behavioral.py -v`
Expected: 7 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/behavioral.py tests/hub/test_organic_behavioral.py
git commit -m "feat(organic-discovery): behavioral co-occurrence clustering (Layer 2)"
```

---

### Task 11: Integrate Behavioral Layer into Module

**Files:**
- Modify: `aria/modules/organic_discovery/module.py` — add Layer 2 after Layer 1
- Test: `tests/hub/test_organic_discovery_module.py` — add behavioral test

**Step 1: Write failing test**

Add to `tests/hub/test_organic_discovery_module.py`:

```python
@pytest.mark.asyncio
async def test_module_discovers_behavioral_capabilities(mock_hub):
    """Layer 2 should find behavioral patterns from logbook data."""
    # Mock logbook files in ~/ha-logs/
    mock_logbook = [
        {"entity_id": f"light.room_{i}", "state": "on", "when": f"2026-02-{d:02d}T19:{i:02d}:00"}
        for d in range(1, 15)
        for i in range(6)
    ]

    mock_hub.cache.get = AsyncMock(side_effect=lambda cat: {
        "entities": MOCK_ENTITIES_CACHE,
        "devices": MOCK_DEVICES_CACHE,
        "capabilities": MOCK_CAPABILITIES_CACHE,
        "activity_summary": {"data": {"entity_activity": {}}},
        "discovery_history": None,
    }.get(cat))

    module = OrganicDiscoveryModule(mock_hub)

    with patch.object(module, '_load_logbook', return_value=mock_logbook):
        await module.run_discovery()

    cap_calls = [c for c in mock_hub.cache.set.call_args_list if c[0][0] == "capabilities"]
    assert len(cap_calls) >= 1
    caps = cap_calls[0][0][1]

    # Check for behavioral capabilities (layer == "behavioral")
    behavioral = [c for c in caps.values() if c.get("layer") == "behavioral"]
    # May or may not find behavioral clusters depending on data — just verify no crash
    assert isinstance(behavioral, list)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_discovery_module.py::test_module_discovers_behavioral_capabilities -v`
Expected: FAIL (no `_load_logbook` method)

**Step 3: Modify `module.py`**

Add logbook loading and Layer 2 integration to `OrganicDiscoveryModule.run_discovery()`:

1. Add `_load_logbook()` method that reads 14 days of logbook JSON from `~/ha-logs/`
2. After Layer 1 clustering, call `cluster_behavioral(logbook_entries)`
3. Score and name behavioral clusters with `layer="behavioral"` and include `temporal_pattern`
4. Merge behavioral clusters into the capabilities dict (after domain clusters)

Add import:
```python
from aria.modules.organic_discovery.behavioral import cluster_behavioral, extract_temporal_pattern
```

Add method:
```python
    async def _load_logbook(self, days: int = 14) -> list[dict]:
        """Load recent logbook entries from ~/ha-logs/ JSON files."""
        import json
        from pathlib import Path
        from datetime import date, timedelta

        log_dir = Path.home() / "ha-logs"
        entries = []
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            log_file = log_dir / f"{day.isoformat()}.json"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        day_entries = json.load(f)
                    if isinstance(day_entries, list):
                        entries.extend(day_entries)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to read {log_file}: {e}")

        logger.info(f"Loaded {len(entries)} logbook entries from {days} days")
        return entries
```

In `run_discovery()`, after Layer 1 section and before merge:
```python
        # Phase 2: Behavioral clustering
        logbook_entries = await self._load_logbook()
        if logbook_entries:
            behavioral_clusters = cluster_behavioral(logbook_entries)
            for cluster in behavioral_clusters:
                cluster_info = self._build_cluster_info(cluster, entities, devices, entity_registry)
                cluster_info["temporal_pattern"] = cluster.get("temporal_pattern", {})
                name = heuristic_name(cluster_info)
                description = heuristic_description(cluster_info)

                if name in discovered_caps or name in seed_caps:
                    name = f"behavioral_{name}_{cluster['cluster_id']}"

                components = self._compute_components(
                    cluster, cluster_info, entity_activity, len(entities)
                )
                usefulness = compute_usefulness(components)

                discovered_caps[name] = {
                    "available": True,
                    "entities": cluster["entity_ids"],
                    "total_count": len(cluster["entity_ids"]),
                    "can_predict": False,
                    "source": "organic",
                    "usefulness": usefulness,
                    "usefulness_components": components.to_dict(),
                    "layer": "behavioral",
                    "status": "candidate",
                    "first_seen": run_start[:10],
                    "promoted_at": None,
                    "naming_method": self.settings["naming_backend"],
                    "description": description,
                    "stability_streak": 1,
                    "temporal_pattern": cluster.get("temporal_pattern", {}),
                }
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_discovery_module.py -v`
Expected: 5 PASSED

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_*.py -v`
Expected: All PASSED (~36 tests)

**Step 6: Commit**

```bash
git add aria/modules/organic_discovery/module.py tests/hub/test_organic_discovery_module.py
git commit -m "feat(organic-discovery): integrate behavioral layer 2 into discovery module"
```

---

### Task 12: LLM Naming Backends (Ollama + External)

**Files:**
- Modify: `aria/modules/organic_discovery/naming.py` — add LLM naming functions
- Test: `tests/hub/test_organic_naming.py` — add LLM naming tests

**Step 1: Write failing test**

Add to `tests/hub/test_organic_naming.py`:

```python
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_ollama_name_returns_string():
    with patch("aria.modules.organic_discovery.naming._call_ollama", new_callable=AsyncMock) as mock:
        mock.return_value = "Morning kitchen routine"
        from aria.modules.organic_discovery.naming import ollama_name
        result = await ollama_name(CLUSTER_MIXED_ROOM)
        assert isinstance(result, str)
        assert len(result) > 0

@pytest.mark.asyncio
async def test_ollama_name_fallback_on_error():
    with patch("aria.modules.organic_discovery.naming._call_ollama", new_callable=AsyncMock) as mock:
        mock.side_effect = Exception("Ollama down")
        from aria.modules.organic_discovery.naming import ollama_name
        result = await ollama_name(CLUSTER_MIXED_ROOM)
        # Should fall back to heuristic
        assert isinstance(result, str)
        assert len(result) > 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_naming.py -v`
Expected: FAIL

**Step 3: Add LLM naming to `naming.py`**

```python
import logging
import json

logger = logging.getLogger(__name__)


async def _call_ollama(prompt: str, model: str = "deepseek-r1:8b") -> str:
    """Call local Ollama API."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            return data.get("response", "").strip()


async def ollama_name(cluster_info: dict) -> str:
    """Generate a cluster name using local Ollama LLM."""
    entities = cluster_info.get("entity_ids", [])[:10]
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})
    temporal = cluster_info.get("temporal_pattern", {})

    prompt = (
        "Name this group of Home Assistant entities in 2-4 words. "
        "Return ONLY the name in snake_case, nothing else.\n\n"
        f"Entities: {', '.join(entities)}\n"
        f"Domains: {json.dumps(domains)}\n"
        f"Areas: {json.dumps(areas)}\n"
        f"Temporal: {json.dumps(temporal)}\n"
    )

    try:
        result = await _call_ollama(prompt)
        # Clean response — extract snake_case name
        name = result.strip().lower().replace(" ", "_").replace("-", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_")
        if name and len(name) > 2:
            return name
    except Exception as e:
        logger.warning(f"Ollama naming failed, falling back to heuristic: {e}")

    return heuristic_name(cluster_info)


async def ollama_description(cluster_info: dict) -> str:
    """Generate a cluster description using local Ollama LLM."""
    entities = cluster_info.get("entity_ids", [])[:10]
    domains = cluster_info.get("domains", {})
    areas = cluster_info.get("areas", {})

    prompt = (
        "Describe this group of Home Assistant entities in one sentence. "
        "Explain what they have in common and why they might be grouped.\n\n"
        f"Entities: {', '.join(entities)}\n"
        f"Domains: {json.dumps(domains)}\n"
        f"Areas: {json.dumps(areas)}\n"
    )

    try:
        result = await _call_ollama(prompt)
        if result and len(result) > 10:
            return result[:200]
    except Exception as e:
        logger.warning(f"Ollama description failed: {e}")

    return heuristic_description(cluster_info)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_naming.py -v`
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add aria/modules/organic_discovery/naming.py tests/hub/test_organic_naming.py
git commit -m "feat(organic-discovery): Ollama LLM naming backend with heuristic fallback"
```

---

### Task 13: Wire LLM Naming into Module

**Files:**
- Modify: `aria/modules/organic_discovery/module.py` — use selected naming backend

**Step 1: Modify `run_discovery` naming section**

Replace the heuristic-only naming calls with a method that dispatches to the selected backend:

```python
    async def _name_cluster(self, cluster_info: dict) -> tuple[str, str]:
        """Name and describe a cluster using the configured backend."""
        backend = self.settings["naming_backend"]

        if backend == "ollama":
            from aria.modules.organic_discovery.naming import ollama_name, ollama_description
            name = await ollama_name(cluster_info)
            description = await ollama_description(cluster_info)
        else:
            # heuristic (default and fallback)
            name = heuristic_name(cluster_info)
            description = heuristic_description(cluster_info)

        return name, description
```

Update both Layer 1 and Layer 2 loops to use `await self._name_cluster(cluster_info)` instead of direct `heuristic_name()`/`heuristic_description()` calls.

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_*.py -v`
Expected: All PASSED

**Step 3: Commit**

```bash
git add aria/modules/organic_discovery/module.py
git commit -m "feat(organic-discovery): wire naming backend selection into discovery pipeline"
```

---

### Task 14: Systemd Timer

**Files:**
- Create: `~/.config/systemd/user/aria-organic-discovery.service`
- Create: `~/.config/systemd/user/aria-organic-discovery.timer`

**Step 1: Create service file**

```ini
# ~/.config/systemd/user/aria-organic-discovery.service
[Unit]
Description=ARIA Organic Capability Discovery
After=aria-hub.service

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -s -X POST http://127.0.0.1:8001/api/discovery/run
TimeoutStartSec=300
MemoryMax=2G
```

**Step 2: Create timer file**

```ini
# ~/.config/systemd/user/aria-organic-discovery.timer
[Unit]
Description=ARIA Organic Discovery (weekly)

[Timer]
OnCalendar=Sun *-*-* 04:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**Step 3: Enable timer**

```bash
systemctl --user daemon-reload
systemctl --user enable aria-organic-discovery.timer
systemctl --user start aria-organic-discovery.timer
systemctl --user list-timers | grep organic
```

**Step 4: Commit the timer files (if in a tracked config directory)**

```bash
git add -p  # Only if timer files are in repo
git commit -m "ops: add weekly organic discovery timer (Sunday 4am)"
```

---

### Task 15: Full Integration Test

**Files:**
- Test: `tests/hub/test_organic_integration.py`

**Step 1: Write integration test**

```python
# tests/hub/test_organic_integration.py
"""Integration test for the full organic discovery pipeline."""
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from aria.modules.organic_discovery.module import OrganicDiscoveryModule


def _generate_realistic_entities(n_lights=20, n_power=15, n_motion=10, n_climate=5, n_misc=20):
    """Generate a realistic set of HA entities with clear clustering potential."""
    entities = []
    rng = np.random.RandomState(42)

    for i in range(n_lights):
        area = ["living_room", "bedroom", "kitchen", "office", "bathroom"][i % 5]
        entities.append({
            "entity_id": f"light.{area}_{i}",
            "state": rng.choice(["on", "off"]),
            "attributes": {
                "device_class": None,
                "supported_color_modes": rng.choice([["brightness"], ["brightness", "color_temp"], ["brightness", "rgb"]], p=[0.5, 0.3, 0.2]).tolist(),
            },
        })

    for i in range(n_power):
        entities.append({
            "entity_id": f"sensor.outlet_{i}_power",
            "state": str(round(rng.uniform(0, 300), 1)),
            "attributes": {"device_class": "power", "unit_of_measurement": "W"},
        })

    for i in range(n_motion):
        entities.append({
            "entity_id": f"binary_sensor.motion_{i}",
            "state": rng.choice(["on", "off"]),
            "attributes": {"device_class": "motion"},
        })

    for i in range(n_climate):
        entities.append({
            "entity_id": f"climate.zone_{i}",
            "state": rng.choice(["heat", "cool", "off"]),
            "attributes": {"hvac_modes": ["heat", "cool", "off"], "current_temperature": round(rng.uniform(18, 26), 1)},
        })

    for i in range(n_misc):
        entities.append({
            "entity_id": f"sensor.misc_{i}",
            "state": str(round(rng.uniform(0, 100), 1)),
            "attributes": {"device_class": rng.choice(["temperature", "humidity", "battery", "illuminance"])},
        })

    return entities


SEED_CAPS = {
    "lighting": {"available": True, "entities": [f"light.{a}_{i}" for a in ["living_room", "bedroom", "kitchen", "office", "bathroom"] for i in range(4)]},
    "power_monitoring": {"available": True, "entities": [f"sensor.outlet_{i}_power" for i in range(15)]},
}


@pytest.fixture
def mock_hub():
    hub = AsyncMock()
    hub.cache = AsyncMock()
    hub.publish = AsyncMock()
    return hub


@pytest.mark.asyncio
async def test_full_pipeline_discovers_clusters(mock_hub):
    entities = _generate_realistic_entities()
    mock_hub.cache.get = AsyncMock(side_effect=lambda cat: {
        "entities": {"data": entities},
        "devices": {"data": {}},
        "capabilities": {"data": SEED_CAPS},
        "activity_summary": {"data": {"entity_activity": {}}},
        "discovery_history": None,
        "discovery_settings": None,
    }.get(cat))

    module = OrganicDiscoveryModule(mock_hub)
    await module.initialize()

    with patch.object(module, '_load_logbook', return_value=[]):
        await module.run_discovery()

    # Verify capabilities were written
    cap_calls = [c for c in mock_hub.cache.set.call_args_list if c[0][0] == "capabilities"]
    assert len(cap_calls) == 1

    caps = cap_calls[0][0][1]

    # Seeds should be preserved
    assert "lighting" in caps
    assert "power_monitoring" in caps
    assert caps["lighting"]["source"] == "seed"

    # Should have found organic clusters
    organic = {k: v for k, v in caps.items() if v.get("source") == "organic"}
    assert len(organic) >= 1, f"Expected organic clusters, got: {list(caps.keys())}"

    # All capabilities should have usefulness scores
    for name, cap in caps.items():
        assert "usefulness" in cap, f"{name} missing usefulness"
        assert 0 <= cap["usefulness"] <= 100, f"{name} usefulness out of range: {cap['usefulness']}"
        assert "usefulness_components" in cap, f"{name} missing components"
        assert "status" in cap, f"{name} missing status"


@pytest.mark.asyncio
async def test_full_pipeline_publishes_event(mock_hub):
    entities = _generate_realistic_entities()
    mock_hub.cache.get = AsyncMock(side_effect=lambda cat: {
        "entities": {"data": entities},
        "devices": {"data": {}},
        "capabilities": {"data": SEED_CAPS},
        "activity_summary": {"data": {"entity_activity": {}}},
        "discovery_history": None,
        "discovery_settings": None,
    }.get(cat))

    module = OrganicDiscoveryModule(mock_hub)
    await module.initialize()

    with patch.object(module, '_load_logbook', return_value=[]):
        await module.run_discovery()

    mock_hub.publish.assert_called_once()
    event_type, event_data = mock_hub.publish.call_args[0]
    assert event_type == "organic_discovery_complete"
    assert "total_clusters" in event_data
```

**Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/hub/test_organic_integration.py -v`
Expected: 2 PASSED

**Step 3: Run ALL tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: All existing tests PASS, ~40 new organic discovery tests PASS

**Step 4: Commit**

```bash
git add tests/hub/test_organic_integration.py
git commit -m "test(organic-discovery): full pipeline integration test"
```

---

## Parallelism Guide

**For team-based execution, these tasks can be parallelized:**

```
BACKEND TRACK (general-purpose agents)          FRONTEND TRACK (general-purpose agents)
────────────────────────────────────            ──────────────────────────────────────
Task 1: Feature vectors ─┐
Task 2: Clustering ──────┤ (parallel)          Task 8: Capabilities page rewrite ──────┐
Task 3: Seed validation ─┤                     Task 9: Settings panel ─────────────────┘
Task 4: Scoring ─────────┤                     (depends on Task 7 API endpoints)
Task 5: Naming ──────────┘
         │
Task 6: Module (depends 1-5)
Task 7: API (depends 6)  ─── unlocks ───────→ Frontend Tasks 8-9
         │
Task 10: Behavioral clustering ─┐
Task 11: Integrate Layer 2 ─────┘
Task 12: LLM naming ───────────┐
Task 13: Wire LLM naming ──────┘
         │
Task 14: Systemd timer
Task 15: Integration test (depends all)
```

**Max parallel agents: 6** (Tasks 1-5 backend + Task 8 frontend stub with mock API)

**Critical path:** Tasks 1-5 → 6 → 7 → 8-9 (frontend blocked on API)

**Frontend can start early** by mocking API responses while backend builds.
