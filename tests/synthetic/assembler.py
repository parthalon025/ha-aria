"""Snapshot assembler — feeds synthetic entity states through real ARIA collectors."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta

import aria.engine.collectors.extractors  # noqa: F401 — triggers registration
from aria.engine.collectors.registry import CollectorRegistry
from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.config import HolidayConfig, SafetyConfig
from aria.engine.features.time_features import build_time_features
from tests.synthetic.entities import DeviceRoster, EntityStateGenerator
from tests.synthetic.people import Person
from tests.synthetic.weather import WeatherProfile


class SnapshotAssembler:
    """Builds ARIA snapshots from synthetic entity states using real collectors."""

    def __init__(
        self,
        roster: DeviceRoster,
        people: list[Person],
        weather: WeatherProfile,
        seed: int = 42,
    ):
        self.roster = roster
        self.people = people
        self.weather = weather
        self.seed = seed
        self.entity_gen = EntityStateGenerator(roster, people, seed)

    def build_snapshot(self, day: int, date_str: str, hour: float = 18.0) -> dict:
        """Build a single snapshot for a given day using real collectors."""
        import random

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
        holidays_config = HolidayConfig()
        safety_config = SafetyConfig()

        states = self.entity_gen.generate_states(
            day=day,
            hour=hour,
            is_weekend=is_weekend,
            sunrise=self.weather.sunrise,
            sunset=self.weather.sunset,
        )

        # Patch synthetic states to match what real HA provides and collectors expect
        self._patch_states_for_collectors(states, date_str, hour)

        snapshot = build_empty_snapshot(date_str, holidays_config)

        for name, collector_cls in CollectorRegistry.all().items():
            collector = collector_cls(safety_config=safety_config) if name == "entities_summary" else collector_cls()
            collector.extract(snapshot, states)

        # PowerCollector looks for usp_pdu_pro entities which don't exist in
        # synthetic data. Inject power from the synthetic total_power sensor.
        if snapshot["power"]["total_watts"] == 0:
            for s in states:
                if s["entity_id"] == "sensor.total_power":
                    with contextlib.suppress(ValueError, TypeError):
                        snapshot["power"]["total_watts"] = float(s["state"])
                    break

        # Enrich motion with active_count (done in intraday snapshot code, not by collector)
        snapshot["motion"]["active_count"] = sum(1 for v in snapshot["motion"]["sensors"].values() if v == "on")

        # Weather from synthetic profile
        weather_cond = self.weather.get_conditions(day, hour, self.seed)
        snapshot["weather"] = weather_cond

        # Time features
        timestamp_str = f"{date_str}T{int(hour):02d}:{int((hour % 1) * 60):02d}:00"
        sun_data = None
        for s in states:
            if s["entity_id"] == "sun.sun":
                sun_data = s["attributes"]
                break
        snapshot["time_features"] = build_time_features(timestamp_str, sun_data, date_str)

        # Logbook summary — varies with occupancy, time, and active devices
        rng = random.Random(self.seed + day * 100 + int(hour))
        snapshot["logbook_summary"] = self._build_logbook_summary(snapshot, day, hour, rng)

        return snapshot

    def _build_logbook_summary(self, snapshot: dict, day: int, hour: float, rng) -> dict:
        """Build realistic logbook event counts correlated with activity."""
        people_home = len(snapshot.get("occupancy", {}).get("people_home", []))
        lights_on = snapshot.get("lights", {}).get("on", 0)
        media_active = snapshot.get("media", {}).get("total_active", 0)

        # Peak during waking hours (7am-11pm), low at night
        if 7 <= hour <= 23:
            hour_factor = 1.0
        elif 5 <= hour < 7 or hour > 23:
            hour_factor = 0.3
        else:
            hour_factor = 0.1

        base_events = int((80 + people_home * 40 + lights_on * 10 + media_active * 15) * hour_factor)
        useful = int(base_events * 0.8)
        noise = rng.randint(-10, 10)

        return {
            "total_events": max(0, base_events + noise),
            "useful_events": max(0, useful + noise),
            "by_domain": {
                "light": max(0, lights_on * 5 + rng.randint(0, 10)),
                "switch": rng.randint(5, 20),
                "binary_sensor": max(0, int(30 * hour_factor) + rng.randint(0, 15)),
            },
            "hourly": {},
        }

    def _patch_states_for_collectors(self, states: list[dict], date_str: str, hour: float) -> None:
        """Mutate synthetic states so real collectors can extract data correctly.

        Real HA entities have attributes that the synthetic Device.to_ha_state()
        doesn't produce. This method adds them.
        """
        # Build sunrise/sunset ISO strings for SunCollector (expects next_rising/next_setting)
        sunrise_hour = int(self.weather.sunrise)
        sunrise_min = int((self.weather.sunrise % 1) * 60)
        sunset_hour = int(self.weather.sunset)
        sunset_min = int((self.weather.sunset % 1) * 60)
        next_rising = f"{date_str}T{sunrise_hour:02d}:{sunrise_min:02d}:00+00:00"
        next_setting = f"{date_str}T{sunset_hour:02d}:{sunset_min:02d}:00+00:00"

        for s in states:
            eid = s["entity_id"]

            # SunCollector expects next_rising / next_setting attributes
            if eid == "sun.sun":
                s["attributes"]["next_rising"] = next_rising
                s["attributes"]["next_setting"] = next_setting
                # Also provide sunrise/sunset in HH:MM for time_features sun_data
                s["attributes"]["sunrise"] = f"{sunrise_hour:02d}:{sunrise_min:02d}"
                s["attributes"]["sunset"] = f"{sunset_hour:02d}:{sunset_min:02d}"

            # EVCollector checks for "mi" in unit_of_measurement for range
            if eid == "sensor.luda_range":
                s["attributes"]["unit_of_measurement"] = "mi"

    def build_daily_series(self, days: int, start_date: str = "2026-02-01") -> list[dict]:
        """Build a series of daily snapshots."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        snapshots = []
        for day in range(days):
            dt = start + timedelta(days=day)
            date_str = dt.strftime("%Y-%m-%d")
            snapshot = self.build_snapshot(day=day, date_str=date_str)
            snapshots.append(snapshot)
        return snapshots
