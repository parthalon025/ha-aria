"""Tests for co-occurrence detection and adaptive time windows."""

from aria.shared.co_occurrence import (
    compute_adaptive_window,
    find_co_occurring_sets,
)


class TestFindCoOccurringSets:
    def test_basic_co_occurrence(self):
        """Two entities that always change together should be detected."""
        events = []
        for i in range(10):
            base_ts = f"2026-02-{i + 1:02d}T07:00:00"
            events.append({"timestamp": base_ts, "entity_id": "light.bedroom", "new_state": "on", "area_id": "bedroom"})
            events.append(
                {
                    "timestamp": f"2026-02-{i + 1:02d}T07:01:00",
                    "entity_id": "binary_sensor.bedroom_motion",
                    "new_state": "on",
                    "area_id": "bedroom",
                }
            )
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) >= 1
        entity_sets = [c.entities for c in clusters]
        assert any({"light.bedroom", "binary_sensor.bedroom_motion"} <= s for s in entity_sets)

    def test_different_windows_no_overlap(self):
        """Events far apart in time should NOT co-occur."""
        events = []
        for i in range(10):
            events.append(
                {
                    "timestamp": f"2026-02-{i + 1:02d}T07:00:00",
                    "entity_id": "light.bedroom",
                    "new_state": "on",
                    "area_id": "bedroom",
                }
            )
            events.append(
                {
                    "timestamp": f"2026-02-{i + 1:02d}T20:00:00",
                    "entity_id": "light.kitchen",
                    "new_state": "on",
                    "area_id": "kitchen",
                }
            )
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        entity_sets = [c.entities for c in clusters]
        assert not any({"light.bedroom", "light.kitchen"} <= s for s in entity_sets)

    def test_three_entity_cluster(self):
        """Three entities changing together should form a cluster."""
        events = []
        for i in range(8):
            base = f"2026-02-{i + 1:02d}T07"
            events.append(
                {"timestamp": f"{base}:00:00", "entity_id": "light.bedroom", "new_state": "on", "area_id": "bedroom"}
            )
            events.append(
                {"timestamp": f"{base}:01:00", "entity_id": "cover.bedroom", "new_state": "open", "area_id": "bedroom"}
            )
            events.append(
                {
                    "timestamp": f"{base}:02:00",
                    "entity_id": "binary_sensor.bedroom_motion",
                    "new_state": "on",
                    "area_id": "bedroom",
                }
            )
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) >= 1
        # Should find the 3-entity cluster
        found_triple = any(len(c.entities) >= 3 for c in clusters)
        assert found_triple

    def test_below_min_count(self):
        """Clusters below min_count should not be returned."""
        events = []
        for i in range(3):  # Only 3 occurrences
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:00:00", "entity_id": "light.a", "new_state": "on"})
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:01:00", "entity_id": "light.b", "new_state": "on"})
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) == 0

    def test_empty_events(self):
        """Empty event list should return empty clusters."""
        clusters = find_co_occurring_sets([], window_minutes=20, min_count=5)
        assert clusters == []

    def test_cluster_has_count(self):
        """Clusters should report observation count."""
        events = []
        for i in range(7):
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:00:00", "entity_id": "light.a", "new_state": "on"})
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:01:00", "entity_id": "light.b", "new_state": "on"})
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) >= 1
        assert clusters[0].count >= 7

    def test_cluster_has_typical_ordering(self):
        """Clusters should track the most common entity ordering."""
        events = []
        for i in range(10):
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:00:00", "entity_id": "light.a", "new_state": "on"})
            events.append({"timestamp": f"2026-02-{i + 1:02d}T07:02:00", "entity_id": "light.b", "new_state": "on"})
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) >= 1
        # light.a always comes first
        assert clusters[0].typical_ordering[0] == "light.a"

    def test_single_entity_not_cluster(self):
        """A single entity appearing repeatedly is not a cluster."""
        events = [
            {"timestamp": f"2026-02-{i + 1:02d}T07:00:00", "entity_id": "light.a", "new_state": "on"} for i in range(20)
        ]
        clusters = find_co_occurring_sets(events, window_minutes=20, min_count=5)
        assert len(clusters) == 0


class TestComputeAdaptiveWindow:
    def test_consistent_times(self):
        """Timestamps with low variance should produce narrow window."""
        # All at ~07:00 with ±5 min variation
        timestamps = [
            "2026-02-01T06:55:00",
            "2026-02-02T07:00:00",
            "2026-02-03T07:05:00",
            "2026-02-04T06:58:00",
            "2026-02-05T07:02:00",
            "2026-02-06T07:01:00",
            "2026-02-07T06:57:00",
        ]
        median_time, sigma_minutes, skip = compute_adaptive_window(timestamps)
        assert not skip
        assert sigma_minutes < 10  # Very consistent

    def test_high_variance_skips(self):
        """Timestamps spread across the day should skip time condition."""
        timestamps = [
            "2026-02-01T06:00:00",
            "2026-02-02T12:00:00",
            "2026-02-03T18:00:00",
            "2026-02-04T03:00:00",
            "2026-02-05T22:00:00",
            "2026-02-06T15:00:00",
        ]
        _, sigma_minutes, skip = compute_adaptive_window(timestamps)
        assert skip  # σ > 90 minutes

    def test_custom_max_sigma(self):
        """Custom max_sigma threshold should be respected."""
        # Moderate spread ~2 hours
        timestamps = [
            "2026-02-01T06:00:00",
            "2026-02-02T07:00:00",
            "2026-02-03T08:00:00",
            "2026-02-04T06:30:00",
            "2026-02-05T07:30:00",
        ]
        _, sigma_minutes, skip = compute_adaptive_window(timestamps, max_sigma_minutes=30)
        assert skip  # With strict threshold, this should skip

    def test_single_timestamp(self):
        """Single timestamp should have zero sigma, no skip."""
        median_time, sigma, skip = compute_adaptive_window(["2026-02-01T07:00:00"])
        assert sigma == 0.0
        assert not skip

    def test_empty_timestamps(self):
        """Empty list should return safe defaults."""
        median_time, sigma, skip = compute_adaptive_window([])
        assert skip  # Can't compute window from nothing

    def test_median_time_reasonable(self):
        """Median time should be near the center of the timestamps."""
        timestamps = [
            "2026-02-01T07:00:00",
            "2026-02-02T07:10:00",
            "2026-02-03T07:20:00",
        ]
        median_time, _, _ = compute_adaptive_window(timestamps)
        # Median should be around 07:10
        assert "07:" in median_time
