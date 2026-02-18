"""Tests for cross-domain transfer candidate data model."""

from datetime import datetime

import pytest

from aria.engine.transfer import (
    TransferCandidate,
    TransferType,
    compute_jaccard_similarity,
)


class TestTransferType:
    """Test transfer type enum."""

    def test_enum_values(self):
        assert TransferType.ROOM_TO_ROOM.value == "room_to_room"
        assert TransferType.ROUTINE_TO_ROUTINE.value == "routine_to_routine"


class TestJaccardSimilarity:
    """Test Jaccard similarity computation."""

    def test_identical_sets(self):
        assert compute_jaccard_similarity({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint_sets(self):
        assert compute_jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        result = compute_jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert abs(result - 0.5) < 0.01  # 2 shared / 4 total

    def test_empty_sets(self):
        assert compute_jaccard_similarity(set(), set()) == 0.0

    def test_one_empty_set(self):
        assert compute_jaccard_similarity({"a"}, set()) == 0.0


class TestTransferCandidate:
    """Test TransferCandidate lifecycle."""

    def test_create_room_to_room(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1", "light.kitchen_2"],
            target_entities=["light.bedroom_1", "light.bedroom_2"],
        )
        assert tc.state == "hypothesis"
        assert tc.shadow_tests == 0
        assert tc.shadow_hits == 0
        assert tc.hit_rate == 0.0

    def test_create_routine_to_routine(self):
        tc = TransferCandidate(
            source_capability="weekday_morning",
            target_context="weekend_morning",
            transfer_type=TransferType.ROUTINE_TO_ROUTINE,
            similarity_score=0.65,
            source_entities=["light.kitchen_1", "sensor.motion_kitchen"],
            target_entities=["light.kitchen_1", "sensor.motion_kitchen"],
            timing_offset_minutes=120,
        )
        assert tc.transfer_type == TransferType.ROUTINE_TO_ROUTINE
        assert tc.timing_offset_minutes == 120

    def test_record_shadow_result_hit(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        tc.record_shadow_result(hit=True)
        assert tc.shadow_tests == 1
        assert tc.shadow_hits == 1
        assert tc.hit_rate == 1.0
        assert tc.state == "testing"

    def test_record_shadow_result_miss(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        tc.record_shadow_result(hit=False)
        assert tc.shadow_tests == 1
        assert tc.shadow_hits == 0
        assert tc.hit_rate == 0.0

    def test_promotion_threshold(self):
        """After 7+ days with >=60% hit rate, state becomes promoted."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        # Simulate 20 tests, 15 hits (75% hit rate)
        for _ in range(15):
            tc.record_shadow_result(hit=True)
        for _ in range(5):
            tc.record_shadow_result(hit=False)
        # Force testing_since to 8 days ago
        tc.testing_since = datetime(2026, 2, 9)
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "promoted"

    def test_rejection_threshold(self):
        """After 7+ days with <30% hit rate, state becomes rejected."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        # Simulate 20 tests, 4 hits (20% hit rate)
        for _ in range(4):
            tc.record_shadow_result(hit=True)
        for _ in range(16):
            tc.record_shadow_result(hit=False)
        tc.testing_since = datetime(2026, 2, 9)
        tc.check_promotion(min_days=7, min_hit_rate=0.6, reject_below=0.3)
        assert tc.state == "rejected"

    def test_not_promoted_before_min_days(self):
        """Promotion requires minimum testing period."""
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        for _ in range(20):
            tc.record_shadow_result(hit=True)
        # testing_since is now (very recent), so min_days=7 not met
        tc.check_promotion(min_days=7, min_hit_rate=0.6)
        assert tc.state == "testing"

    def test_to_dict(self):
        tc = TransferCandidate(
            source_capability="kitchen_lighting",
            target_context="bedroom",
            transfer_type=TransferType.ROOM_TO_ROOM,
            similarity_score=0.72,
            source_entities=["light.kitchen_1"],
            target_entities=["light.bedroom_1"],
        )
        d = tc.to_dict()
        assert d["source_capability"] == "kitchen_lighting"
        assert d["target_context"] == "bedroom"
        assert d["transfer_type"] == "room_to_room"
        assert d["similarity_score"] == 0.72
        assert d["state"] == "hypothesis"

    def test_similarity_below_threshold_raises(self):
        """Similarity below 0.4 should not create a candidate."""
        with pytest.raises(ValueError, match="similarity"):
            TransferCandidate(
                source_capability="kitchen_lighting",
                target_context="bedroom",
                transfer_type=TransferType.ROOM_TO_ROOM,
                similarity_score=0.3,
                source_entities=["light.kitchen_1"],
                target_entities=["light.bedroom_1"],
            )
