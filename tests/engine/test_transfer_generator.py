"""Tests for transfer hypothesis generation from organic capabilities."""

from aria.engine.transfer import TransferType
from aria.engine.transfer_generator import generate_transfer_candidates


class TestRoomToRoomGeneration:
    """Test room-to-room transfer hypothesis generation."""

    def test_generates_candidate_from_similar_rooms(self):
        """Two capabilities in different areas with overlapping domains."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1", "light.kitchen_2", "binary_sensor.kitchen_motion"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1", "binary_sensor.bedroom_motion"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.kitchen_2": {"entity_id": "light.kitchen_2", "domain": "light", "area_id": "kitchen"},
            "binary_sensor.kitchen_motion": {
                "entity_id": "binary_sensor.kitchen_motion",
                "domain": "binary_sensor",
                "area_id": "kitchen",
                "device_class": "motion",
            },
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "binary_sensor.bedroom_motion": {
                "entity_id": "binary_sensor.bedroom_motion",
                "domain": "binary_sensor",
                "area_id": "bedroom",
                "device_class": "motion",
            },
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)

        assert len(candidates) >= 1
        # At least one room-to-room candidate
        r2r = [c for c in candidates if c.transfer_type == TransferType.ROOM_TO_ROOM]
        assert len(r2r) >= 1

    def test_no_candidate_for_dissimilar_rooms(self):
        """Totally different domain compositions don't generate candidates."""
        capabilities = {
            "power_monitoring": {
                "entities": ["sensor.power_1", "sensor.power_2"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1", "light.bedroom_2"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "sensor.power_1": {
                "entity_id": "sensor.power_1",
                "domain": "sensor",
                "area_id": "office",
                "device_class": "power",
            },
            "sensor.power_2": {
                "entity_id": "sensor.power_2",
                "domain": "sensor",
                "area_id": "office",
                "device_class": "power",
            },
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
            "light.bedroom_2": {"entity_id": "light.bedroom_2", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        assert len(candidates) == 0

    def test_skips_seed_capabilities(self):
        """Seed capabilities are not source candidates for transfer."""
        capabilities = {
            "power_monitoring": {
                "entities": ["sensor.power_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "seed",
            },
            "bedroom_power": {
                "entities": ["sensor.power_2"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "sensor.power_1": {"entity_id": "sensor.power_1", "domain": "sensor", "area_id": "kitchen"},
            "sensor.power_2": {"entity_id": "sensor.power_2", "domain": "sensor", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        # Should not use seed as source
        for c in candidates:
            assert c.source_capability != "power_monitoring"

    def test_only_promoted_as_source(self):
        """Only promoted capabilities can be source of transfer."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1"],
                "layer": "domain",
                "status": "candidate",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        assert len(candidates) == 0


class TestRoutineToRoutineGeneration:
    """Test routine-to-routine transfer hypothesis generation."""

    def test_generates_from_behavioral_with_different_timing(self):
        """Behavioral capabilities with overlapping entities but different peak hours."""
        capabilities = {
            "weekday_morning": {
                "entities": ["light.kitchen_1", "sensor.motion_kitchen"],
                "layer": "behavioral",
                "status": "promoted",
                "source": "organic",
                "temporal_pattern": {"peak_hours": [7, 8], "weekday_bias": 0.9},
            },
            "weekend_morning": {
                "entities": ["light.kitchen_1", "sensor.motion_kitchen"],
                "layer": "behavioral",
                "status": "candidate",
                "source": "organic",
                "temporal_pattern": {"peak_hours": [9, 10], "weekday_bias": 0.2},
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "sensor.motion_kitchen": {"entity_id": "sensor.motion_kitchen", "domain": "sensor", "area_id": "kitchen"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        r2r = [c for c in candidates if c.transfer_type == TransferType.ROUTINE_TO_ROUTINE]
        assert len(r2r) >= 1
        # Should have a timing offset
        assert r2r[0].timing_offset_minutes != 0

    def test_no_duplicate_candidates(self):
        """Same pair should not produce duplicate candidates."""
        capabilities = {
            "kitchen_lighting": {
                "entities": ["light.kitchen_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
            "bedroom_lighting": {
                "entities": ["light.bedroom_1"],
                "layer": "domain",
                "status": "promoted",
                "source": "organic",
            },
        }
        entities_cache = {
            "light.kitchen_1": {"entity_id": "light.kitchen_1", "domain": "light", "area_id": "kitchen"},
            "light.bedroom_1": {"entity_id": "light.bedroom_1", "domain": "light", "area_id": "bedroom"},
        }

        candidates = generate_transfer_candidates(capabilities, entities_cache, min_similarity=0.5)
        # For each ordered pair (A->B), there should be at most one candidate
        pairs = [(c.source_capability, c.target_context) for c in candidates]
        assert len(pairs) == len(set(pairs))
