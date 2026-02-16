"""Tests for snapshot validation before model training."""

import pytest
from aria.engine.validation import validate_snapshot, validate_snapshot_batch


class TestSnapshotValidation:
    """Ensure corrupt/incomplete snapshots are caught before training."""

    def test_valid_snapshot_passes(self):
        snap = {
            "date": "2026-02-15",
            "entities": {"total": 3050, "unavailable": 50},
            "power": {"total_watts": 200.0},
            "occupancy": {"people_home": ["Justin"], "device_count_home": 25},
            "motion": {"sensors": {"Closet motion": "on"}, "active_count": 1},
            "lights": {"on": 5, "off": 60},
        }
        errors = validate_snapshot(snap)
        assert errors == []

    def test_missing_date_rejected(self):
        snap = {"entities": {"total": 100}}
        errors = validate_snapshot(snap)
        assert any("date" in e for e in errors)

    def test_zero_entities_rejected(self):
        snap = {"date": "2026-02-15", "entities": {"total": 0, "unavailable": 0}}
        errors = validate_snapshot(snap)
        assert any("entities" in e.lower() for e in errors)

    def test_high_unavailable_ratio_flagged(self):
        """If >50% entities unavailable, HA was likely restarting."""
        snap = {
            "date": "2026-02-15",
            "entities": {"total": 3050, "unavailable": 2000},
            "power": {"total_watts": 0},
            "occupancy": {},
            "motion": {},
            "lights": {},
        }
        errors = validate_snapshot(snap)
        assert any("unavailable" in e.lower() for e in errors)

    def test_batch_filters_bad_snapshots(self):
        good = {"date": "2026-02-15", "entities": {"total": 3050, "unavailable": 50},
                "power": {"total_watts": 200}, "occupancy": {}, "motion": {}, "lights": {}}
        bad = {"date": "2026-02-14", "entities": {"total": 0, "unavailable": 0}}
        valid, rejected = validate_snapshot_batch([good, bad])
        assert len(valid) == 1
        assert len(rejected) == 1
        assert valid[0]["date"] == "2026-02-15"

    def test_missing_entities_section(self):
        snap = {"date": "2026-02-15"}
        errors = validate_snapshot(snap)
        assert any("entities" in e.lower() for e in errors)

    def test_empty_batch(self):
        valid, rejected = validate_snapshot_batch([])
        assert valid == []
        assert rejected == []

    def test_all_bad_batch(self):
        bad1 = {"entities": {"total": 0}}
        bad2 = {"date": "2026-02-15", "entities": {"total": 5, "unavailable": 0}}
        valid, rejected = validate_snapshot_batch([bad1, bad2])
        assert len(valid) == 0
        assert len(rejected) == 2
