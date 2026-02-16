"""Tests for organic discovery behavioral clustering (Layer 2 — co-occurrence)."""

from __future__ import annotations

import datetime as dt

import numpy as np

from aria.modules.organic_discovery.behavioral import (
    build_cooccurrence_matrix,
    cluster_behavioral,
    extract_temporal_pattern,
)


def _make_logbook(patterns: list[tuple], days: int = 14) -> list[dict]:
    """Generate synthetic logbook entries from co-occurrence patterns.

    Args:
        patterns: list of (entity_ids, hour, minute_offset) tuples.
            Each tuple produces one state change per entity per day at the
            given hour:minute_offset.
        days: number of days to generate.

    Returns:
        Sorted list of logbook entry dicts.
    """
    base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    entries: list[dict] = []
    for day in range(days):
        for entity_ids, hour, minute_offset in patterns:
            for entity_id in entity_ids:
                ts = base + dt.timedelta(days=day, hours=hour, minutes=minute_offset)
                entries.append(
                    {
                        "entity_id": entity_id,
                        "state": "on",
                        "when": ts.isoformat(),
                    }
                )
    entries.sort(key=lambda e: e["when"])
    return entries


class TestBuildCooccurrenceMatrix:
    """Tests for the co-occurrence matrix builder."""

    def test_correct_shape(self):
        """Matrix should be n_entities x n_entities."""
        entries = _make_logbook(
            [
                (["light.a", "light.b", "switch.c"], 8, 0),
            ]
        )
        matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
        n = len(entity_ids)
        assert n == 3
        assert matrix.shape == (n, n)

    def test_symmetric(self):
        """Co-occurrence matrix must be symmetric."""
        entries = _make_logbook(
            [
                (["light.a", "light.b", "switch.c"], 8, 0),
            ]
        )
        matrix, _ = build_cooccurrence_matrix(entries, window_minutes=15)
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_same_window_positive_count(self):
        """Entities firing in the same window should have positive co-occurrence."""
        entries = _make_logbook(
            [
                (["light.a", "light.b"], 8, 0),
            ],
            days=5,
        )
        matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
        i = entity_ids.index("light.a")
        j = entity_ids.index("light.b")
        assert matrix[i][j] > 0
        assert matrix[i][j] == 5  # one co-occurrence per day

    def test_different_windows_zero_count(self):
        """Entities that never share a window should have zero co-occurrence."""
        # light.a fires at 08:00, light.b fires at 20:00 — never in same 15-min window
        entries = _make_logbook(
            [
                (["light.a"], 8, 0),
                (["light.b"], 20, 0),
            ],
            days=5,
        )
        matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
        i = entity_ids.index("light.a")
        j = entity_ids.index("light.b")
        assert matrix[i][j] == 0

    def test_diagonal_is_self_count(self):
        """Diagonal entry should be the number of windows the entity appeared in."""
        entries = _make_logbook(
            [
                (["light.a", "light.b"], 8, 0),
            ],
            days=7,
        )
        matrix, entity_ids = build_cooccurrence_matrix(entries, window_minutes=15)
        i = entity_ids.index("light.a")
        assert matrix[i][i] == 7

    def test_empty_input(self):
        """Empty logbook should return empty matrix and empty entity list."""
        matrix, entity_ids = build_cooccurrence_matrix([])
        assert len(entity_ids) == 0
        assert matrix.shape == (0, 0)

    def test_custom_window_minutes(self):
        """Larger window should capture entities that would be in different small windows."""
        # light.a at :00, light.b at :20 — different in 15-min windows, same in 30-min windows
        entries = _make_logbook(
            [
                (["light.a"], 8, 0),
                (["light.b"], 8, 20),
            ],
            days=5,
        )
        matrix_15, ids_15 = build_cooccurrence_matrix(entries, window_minutes=15)
        matrix_30, ids_30 = build_cooccurrence_matrix(entries, window_minutes=30)

        i15 = ids_15.index("light.a")
        j15 = ids_15.index("light.b")
        assert matrix_15[i15][j15] == 0  # Different 15-min windows

        i30 = ids_30.index("light.a")
        j30 = ids_30.index("light.b")
        assert matrix_30[i30][j30] == 5  # Same 30-min windows


class TestExtractTemporalPattern:
    """Tests for temporal pattern extraction."""

    def test_finds_correct_peak_hours(self):
        """Peak hours should include the hour where all activity happens."""
        entries = _make_logbook(
            [
                (["light.a", "light.b"], 19, 0),
            ],
            days=14,
        )
        pattern = extract_temporal_pattern(["light.a", "light.b"], entries)
        assert 19 in pattern["peak_hours"]

    def test_weekday_bias_all_weekdays(self):
        """Events only on weekdays should give weekday_bias close to 1.0."""
        base = dt.datetime(2026, 1, 5, 10, 0, tzinfo=dt.UTC)  # Monday
        entries = []
        for week in range(4):
            for weekday in range(5):  # Mon-Fri only
                ts = base + dt.timedelta(weeks=week, days=weekday)
                entries.append(
                    {
                        "entity_id": "light.a",
                        "state": "on",
                        "when": ts.isoformat(),
                    }
                )
        pattern = extract_temporal_pattern(["light.a"], entries)
        assert pattern["weekday_bias"] == 1.0

    def test_weekday_bias_all_weekends(self):
        """Events only on weekends should give weekday_bias of 0.0."""
        base = dt.datetime(2026, 1, 3, 10, 0, tzinfo=dt.UTC)  # Saturday
        entries = []
        for week in range(4):
            for day in (0, 1):  # Sat, Sun
                ts = base + dt.timedelta(weeks=week, days=day)
                entries.append(
                    {
                        "entity_id": "light.a",
                        "state": "on",
                        "when": ts.isoformat(),
                    }
                )
        pattern = extract_temporal_pattern(["light.a"], entries)
        assert pattern["weekday_bias"] == 0.0

    def test_peak_hours_threshold(self):
        """Hours with activity <= 1.5x average should NOT be peak hours."""
        # Spread events evenly across hours 0-23 — no peaks
        base = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
        entries = []
        for day in range(14):
            for hour in range(24):
                ts = base + dt.timedelta(days=day, hours=hour)
                entries.append(
                    {
                        "entity_id": "light.a",
                        "state": "on",
                        "when": ts.isoformat(),
                    }
                )
        pattern = extract_temporal_pattern(["light.a"], entries)
        # Uniform distribution — no hour should be a peak
        assert pattern["peak_hours"] == []

    def test_empty_entries_for_entities(self):
        """If no entries match the given entities, return sensible defaults."""
        entries = _make_logbook(
            [
                (["light.other"], 8, 0),
            ]
        )
        pattern = extract_temporal_pattern(["light.nonexistent"], entries)
        assert pattern["peak_hours"] == []
        assert pattern["weekday_bias"] == 0.0


class TestClusterBehavioral:
    """Tests for the full behavioral clustering pipeline."""

    def test_finds_groups_in_structured_data(self):
        """Two clear behavioral groups should produce clusters."""
        # Group 1: morning routine — 5 entities at 07:00
        # Group 2: evening routine — 5 entities at 20:00
        morning = [f"light.morning_{i}" for i in range(6)]
        evening = [f"light.evening_{i}" for i in range(6)]
        entries = _make_logbook(
            [
                (morning, 7, 0),
                (evening, 20, 0),
            ],
            days=30,
        )
        clusters = cluster_behavioral(entries, min_cluster_size=3, window_minutes=15)
        assert len(clusters) >= 1
        # Check structure
        for c in clusters:
            assert "cluster_id" in c
            assert "entity_ids" in c
            assert "silhouette" in c
            assert "temporal_pattern" in c

    def test_empty_input_returns_empty(self):
        """No logbook entries should return empty list."""
        assert cluster_behavioral([]) == []

    def test_single_entity_returns_empty(self):
        """One entity cannot form a cluster."""
        entries = _make_logbook(
            [
                (["light.solo"], 8, 0),
            ],
            days=14,
        )
        clusters = cluster_behavioral(entries, min_cluster_size=3)
        assert clusters == []

    def test_temporal_pattern_present_in_clusters(self):
        """Each cluster should have a temporal_pattern with expected keys."""
        entities = [f"switch.group_{i}" for i in range(8)]
        entries = _make_logbook(
            [
                (entities, 19, 0),
            ],
            days=30,
        )
        clusters = cluster_behavioral(entries, min_cluster_size=3)
        for c in clusters:
            tp = c["temporal_pattern"]
            assert "peak_hours" in tp
            assert "weekday_bias" in tp
