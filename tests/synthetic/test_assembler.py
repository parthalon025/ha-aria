"""Tests for snapshot assembler using real ARIA collectors."""

import pytest

from tests.synthetic.assembler import SnapshotAssembler
from tests.synthetic.entities import DeviceRoster
from tests.synthetic.people import Person, Schedule
from tests.synthetic.weather import WeatherProfile


@pytest.fixture
def assembler():
    roster = DeviceRoster.typical_home()
    people = [
        Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
        Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
    ]
    weather = WeatherProfile("southeast_us", month=2)
    return SnapshotAssembler(roster, people, weather, seed=42)


class TestSnapshotAssembler:
    def test_build_snapshot_for_day(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert snapshot["date"] == "2026-02-14"
        assert snapshot["day_of_week"] == "Saturday"
        assert "power" in snapshot
        assert "lights" in snapshot
        assert "occupancy" in snapshot
        assert "climate" in snapshot
        assert "locks" in snapshot
        assert "motion" in snapshot
        assert "entities" in snapshot

    def test_snapshot_has_nonzero_power(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert snapshot["power"]["total_watts"] > 0

    def test_snapshot_has_occupancy(self, assembler):
        snapshot = assembler.build_snapshot(day=5, date_str="2026-02-19")
        occ = snapshot["occupancy"]
        assert "people_home" in occ or "device_count_home" in occ

    def test_snapshot_has_weather(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert "temp_f" in snapshot["weather"]
        assert snapshot["weather"]["temp_f"] > 0

    def test_snapshot_has_time_features(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert "time_features" in snapshot
        assert "hour_sin" in snapshot["time_features"]

    def test_build_daily_series(self, assembler):
        snapshots = assembler.build_daily_series(days=7, start_date="2026-02-14")
        assert len(snapshots) == 7
        dates = [s["date"] for s in snapshots]
        assert dates[0] == "2026-02-14"
        assert dates[6] == "2026-02-20"

    def test_snapshots_are_deterministic(self, assembler):
        a = assembler.build_daily_series(days=3, start_date="2026-02-14")
        b = assembler.build_daily_series(days=3, start_date="2026-02-14")
        for sa, sb in zip(a, b, strict=False):
            assert sa["power"]["total_watts"] == sb["power"]["total_watts"]
            assert sa["lights"]["on"] == sb["lights"]["on"]

    def test_snapshots_compatible_with_training(self, assembler):
        """Snapshots should work with build_training_data()."""
        from aria.engine.features.vector_builder import build_training_data

        snapshots = assembler.build_daily_series(days=14, start_date="2026-02-01")
        names, X, targets = build_training_data(snapshots)
        assert len(names) > 0
        assert len(X) > 0
        assert "power_watts" in targets
        assert len(targets["power_watts"]) == len(X)
