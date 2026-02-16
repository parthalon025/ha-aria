"""Tests for seed capability validation against discovered clusters."""

import pytest

from aria.modules.organic_discovery.seed_validation import jaccard_similarity, validate_seeds

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
