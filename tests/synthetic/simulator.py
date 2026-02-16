"""HouseholdSimulator — top-level entry point for synthetic data generation."""

from __future__ import annotations

from datetime import datetime, timedelta

from tests.synthetic.assembler import SnapshotAssembler
from tests.synthetic.people import Person, Schedule
from tests.synthetic.scenarios.household import SCENARIOS

# Default hours that capture key diurnal patterns:
# early morning, morning, midday, afternoon, evening, night
INTRADAY_HOURS = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0]


class HouseholdSimulator:
    """Generate realistic household data for ARIA pipeline testing."""

    def __init__(self, scenario: str, days: int, seed: int = 42, start_date: str = "2026-02-01"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
        self.scenario_name = scenario
        self.scenario_config = SCENARIOS[scenario](seed)
        self.days = days
        self.seed = seed
        self.start_date = start_date

    def generate(self, hours_per_day: list[float] | None = None) -> list[dict]:
        """Generate intraday snapshots for the scenario.

        Produces one snapshot per hour per day, giving the ML models varied
        time features and more training samples.
        """
        if hours_per_day is None:
            hours_per_day = INTRADAY_HOURS

        config = self.scenario_config
        people = list(config["people"])
        roster = config["roster"]
        weather = config["weather"]

        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        snapshots = []

        for day in range(self.days):
            dt = start + timedelta(days=day)
            date_str = dt.strftime("%Y-%m-%d")
            is_weekend = dt.weekday() >= 5

            day_people = self._get_people_for_day(people, config, day, is_weekend)

            for hour in hours_per_day:
                assembler = SnapshotAssembler(roster, day_people, weather, self.seed)
                snapshot = assembler.build_snapshot(day=day, date_str=date_str, hour=hour)

                self._apply_scenario_mods(snapshot, config, day)
                snapshots.append(snapshot)

        return snapshots

    def _get_people_for_day(self, base_people: list[Person], config: dict, day: int, is_weekend: bool) -> list[Person]:
        people = list(base_people)

        vacation_days = config.get("vacation_days")
        if vacation_days and day in vacation_days:
            # Everyone is away all day. Use depart=-1 so they leave before any
            # other transition, wake=25/sleep=26/arrive=27 so all home transitions
            # are pushed past hour 24 and never affect the snapshot at hour 18.
            away_sched = Schedule(wake=25, sleep=26, depart=-1, arrive=27)
            return [Person(p.name, away_sched, away_sched) for p in people]

        add_day = config.get("add_person_at_day")
        if add_day and day >= add_day:
            new_person = config["new_person"]
            if new_person.name not in [p.name for p in people]:
                people.append(new_person)

        wfh_person = config.get("wfh_person")
        wfh_start = config.get("wfh_start_day")
        if wfh_person and wfh_start and day >= wfh_start and not is_weekend:
            wfh_sched = Schedule(wake=7, sleep=23, depart=None, arrive=None)
            people = [Person(p.name, wfh_sched, p.schedule_weekend) if p.name == wfh_person else p for p in people]

        return people

    def _apply_scenario_mods(self, snapshot: dict, config: dict, day: int):
        """Apply post-snapshot modifications for scenario-specific effects."""
        # Holiday flags
        holiday_days = config.get("holiday_days")
        if holiday_days and day in holiday_days:
            snapshot["is_holiday"] = True
            snapshot["holiday_name"] = "Holiday"

        # Sensor degradation: mark battery sensors as unavailable after threshold day
        degrade_day = config.get("degrade_start_day")
        if degrade_day and day >= degrade_day:
            # Degrade entities_summary by marking battery sensors unavailable
            entities = snapshot.get("entities_summary", {})
            unavailable = entities.get("unavailable_entities", [])
            days_degraded = day - degrade_day
            # Progressive: more sensors go unavailable over time
            battery_sensors = [
                "sensor.front_door_lock_battery",
                "sensor.back_door_lock_battery",
                "sensor.luda_battery",
            ]
            for i, sensor_id in enumerate(battery_sensors):
                if days_degraded >= i * 3 and sensor_id not in unavailable:
                    unavailable.append(sensor_id)
            entities["unavailable_entities"] = unavailable
            snapshot["entities_summary"] = entities
