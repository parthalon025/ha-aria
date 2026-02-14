"""Tests for household person simulation."""
import pytest
from tests.synthetic.people import Person, Schedule


class TestSchedule:
    def test_weekday_office_has_departure_and_arrival(self):
        sched = Schedule.weekday_office(wake=6.5, sleep=23.0)
        assert sched.wake == 6.5
        assert sched.sleep == 23.0
        assert sched.depart is not None
        assert sched.arrive is not None
        assert sched.depart < sched.arrive

    def test_weekend_schedule_has_no_work(self):
        sched = Schedule.weekend(wake=8.0, sleep=23.5)
        assert sched.depart is None
        assert sched.arrive is None

    def test_schedule_jitter_is_deterministic_with_seed(self):
        sched = Schedule.weekday_office(wake=6.5, sleep=23.0)
        times_a = sched.resolve(day=5, seed=42)
        times_b = sched.resolve(day=5, seed=42)
        assert times_a == times_b

    def test_schedule_jitter_varies_by_day(self):
        sched = Schedule.weekday_office(wake=6.5, sleep=23.0)
        times_a = sched.resolve(day=1, seed=42)
        times_b = sched.resolve(day=2, seed=42)
        assert times_a != times_b

    def test_resolve_returns_hour_floats(self):
        sched = Schedule.weekday_office(wake=6.5, sleep=23.0)
        times = sched.resolve(day=1, seed=42)
        assert "wake" in times
        assert "sleep" in times
        assert isinstance(times["wake"], float)
        assert 5.5 <= times["wake"] <= 7.5


class TestPerson:
    def test_person_has_name_and_schedule(self):
        p = Person("justin", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        assert p.name == "justin"

    def test_get_schedule_for_weekday(self):
        p = Person("justin", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        sched = p.get_schedule(day=0, is_weekend=False)
        assert sched.depart is not None

    def test_get_schedule_for_weekend(self):
        p = Person("justin", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        sched = p.get_schedule(day=5, is_weekend=True)
        assert sched.depart is None

    def test_room_transitions_for_day(self):
        p = Person("justin", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        transitions = p.get_room_transitions(day=0, is_weekend=False, seed=42)
        assert len(transitions) >= 4
        for hour, room in transitions:
            assert isinstance(hour, float)
            assert isinstance(room, str)
        assert transitions[0][1] in ("bedroom", "bathroom")
        hours = [h for h, _ in transitions]
        assert hours == sorted(hours)
