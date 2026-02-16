"""Household scenario definitions."""

from __future__ import annotations

from tests.synthetic.entities import DeviceRoster
from tests.synthetic.people import Person, Schedule
from tests.synthetic.weather import WeatherProfile


def stable_couple(seed: int = 42) -> dict:
    """Two residents with consistent schedules."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
    }


def new_roommate(seed: int = 42) -> dict:
    """Two residents for 14 days, third joins at day 15."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "add_person_at_day": 15,
        "new_person": Person("alex", Schedule.weekday_office(7.5, 23.5), Schedule.weekend(9, 0)),
    }


def vacation(seed: int = 42) -> dict:
    """Both residents away days 10-17."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "vacation_days": range(10, 18),
    }


def work_from_home(seed: int = 42) -> dict:
    """One resident switches to WFH at day 8."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "wfh_person": "justin",
        "wfh_start_day": 8,
    }


def sensor_degradation(seed: int = 42) -> dict:
    """Battery sensors start reporting unavailable after day 20."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "degrade_start_day": 20,
    }


def holiday_week(seed: int = 42) -> dict:
    """Normal schedule with holiday flags."""
    return {
        "people": [
            Person("justin", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("lisa", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=12),
        "holiday_days": range(24, 27),
    }


SCENARIOS = {
    "stable_couple": stable_couple,
    "new_roommate": new_roommate,
    "vacation": vacation,
    "work_from_home": work_from_home,
    "sensor_degradation": sensor_degradation,
    "holiday_week": holiday_week,
}
