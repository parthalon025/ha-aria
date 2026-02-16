"""Tests for HouseholdSimulator and scenarios."""

import pytest

from tests.synthetic.simulator import INTRADAY_HOURS, HouseholdSimulator

HOURS_PER_DAY = len(INTRADAY_HOURS)


class TestHouseholdSimulator:
    def test_stable_couple_scenario(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 7 * HOURS_PER_DAY

    def test_new_roommate_scenario(self):
        sim = HouseholdSimulator(scenario="new_roommate", days=21, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 21 * HOURS_PER_DAY

    def test_vacation_scenario(self):
        sim = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14 * HOURS_PER_DAY

    def test_work_from_home_scenario(self):
        sim = HouseholdSimulator(scenario="work_from_home", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14 * HOURS_PER_DAY

    def test_sensor_degradation_scenario(self):
        sim = HouseholdSimulator(scenario="sensor_degradation", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14 * HOURS_PER_DAY

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            HouseholdSimulator(scenario="nonexistent", days=7, seed=42)

    def test_deterministic(self):
        a = HouseholdSimulator(scenario="stable_couple", days=7, seed=42).generate()
        b = HouseholdSimulator(scenario="stable_couple", days=7, seed=42).generate()
        for sa, sb in zip(a, b, strict=False):
            assert sa["power"]["total_watts"] == sb["power"]["total_watts"]

    def test_snapshots_have_variation(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        powers = [s["power"]["total_watts"] for s in snapshots]
        assert len(set(powers)) > 1

    def test_custom_hours_per_day(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate(hours_per_day=[8.0, 20.0])
        assert len(snapshots) == 7 * 2

    def test_intraday_snapshots_have_varied_hours(self):
        """Snapshots for a single day should have different time features."""
        sim = HouseholdSimulator(scenario="stable_couple", days=1, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == HOURS_PER_DAY

        # Use (sin, cos) pairs since sin alone is not unique across all hours
        hour_encodings = [
            (round(s["time_features"]["hour_sin"], 4), round(s["time_features"]["hour_cos"], 4)) for s in snapshots
        ]
        assert len(set(hour_encodings)) == HOURS_PER_DAY

    def test_lights_vary_between_day_and_night(self):
        """Night snapshots should have more lights on than midday."""
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()

        # Collect lights_on for hour=21 (night) vs hour=12 (midday)
        night_lights = [s["lights"]["on"] for s in snapshots if s["time_features"]["is_night"]]
        day_lights = [
            s["lights"]["on"]
            for s in snapshots
            if not s["time_features"]["is_night"] and not s["time_features"]["is_work_hours"]
        ]

        if night_lights and day_lights:
            avg_night = sum(night_lights) / len(night_lights)
            avg_day = sum(day_lights) / len(day_lights)
            assert avg_night >= avg_day, f"Night lights ({avg_night:.1f}) should >= day lights ({avg_day:.1f})"

    def test_useful_events_correlates_with_occupancy(self):
        """Vacation days should have fewer useful_events than occupied days."""
        sim = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        snapshots = sim.generate()

        # Vacation days are 10-17 (0-indexed in the scenario config)
        occupied_events = []
        vacation_events = []
        for i, s in enumerate(snapshots):
            day_idx = i // HOURS_PER_DAY
            events = s["logbook_summary"]["useful_events"]
            if 10 <= day_idx <= 17:
                vacation_events.append(events)
            elif day_idx < 10:
                occupied_events.append(events)

        assert len(occupied_events) > 0 and len(vacation_events) > 0
        avg_occupied = sum(occupied_events) / len(occupied_events)
        avg_vacation = sum(vacation_events) / len(vacation_events)
        assert avg_occupied > avg_vacation, (
            f"Occupied events ({avg_occupied:.0f}) should > vacation ({avg_vacation:.0f})"
        )

    def test_vacation_has_low_occupancy_midweek(self):
        sim = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        snapshots = sim.generate()
        # Vacation days 10-13 (0-indexed). With 6 snapshots/day, check all
        # snapshots for those days.
        for day_idx in range(10, min(14, 14)):
            for hour_offset in range(HOURS_PER_DAY):
                snap_idx = day_idx * HOURS_PER_DAY + hour_offset
                if snap_idx < len(snapshots):
                    s = snapshots[snap_idx]
                    people_home = s["occupancy"].get("people_home", [])
                    assert len(people_home) == 0, (
                        f"Day {day_idx} hour {INTRADAY_HOURS[hour_offset]}: expected empty house, got {people_home}"
                    )
