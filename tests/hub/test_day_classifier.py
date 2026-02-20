"""Tests for day classifier — workday/weekend/holiday/vacation/wfh classification."""

import pytest

from aria.shared.day_classifier import classify_days


@pytest.fixture
def default_config():
    return {
        "calendar.holiday_keywords": ["holiday", "vacation", "PTO", "trip", "out of office", "off"],
        "calendar.wfh_keywords": ["WFH", "remote", "work from home"],
    }


class TestBasicClassification:
    def test_weekday_is_workday(self, default_config):
        """Monday-Friday without calendar events are workdays."""
        results = classify_days(
            start_date="2026-02-16",  # Monday
            end_date="2026-02-17",  # just Monday
            calendar_events=[],
            person_away_days=set(),
            config=default_config,
        )
        assert len(results) == 1
        assert results[0].day_type == "workday"
        assert results[0].date == "2026-02-16"

    def test_saturday_is_weekend(self, default_config):
        results = classify_days(
            start_date="2026-02-21",  # Saturday
            end_date="2026-02-22",
            calendar_events=[],
            person_away_days=set(),
            config=default_config,
        )
        assert len(results) == 1
        assert results[0].day_type == "weekend"

    def test_sunday_is_weekend(self, default_config):
        results = classify_days(
            start_date="2026-02-22",  # Sunday
            end_date="2026-02-23",
            calendar_events=[],
            person_away_days=set(),
            config=default_config,
        )
        assert len(results) == 1
        assert results[0].day_type == "weekend"

    def test_full_week_classification(self, default_config):
        """Mon-Sun classifies correctly."""
        results = classify_days(
            start_date="2026-02-16",  # Monday
            end_date="2026-02-23",  # to Sunday (exclusive)
            calendar_events=[],
            person_away_days=set(),
            config=default_config,
        )
        assert len(results) == 7
        day_types = [r.day_type for r in results]
        assert day_types == ["workday", "workday", "workday", "workday", "workday", "weekend", "weekend"]


class TestHolidayClassification:
    def test_holiday_keyword_match(self, default_config):
        """Calendar event with holiday keyword overrides workday."""
        results = classify_days(
            start_date="2026-12-25",  # Thursday
            end_date="2026-12-26",
            calendar_events=[{"summary": "Christmas Holiday", "start": "2026-12-25", "end": "2026-12-26"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "holiday"
        assert "Christmas Holiday" in results[0].calendar_events

    def test_pto_keyword_match(self, default_config):
        """PTO keyword triggers holiday classification."""
        results = classify_days(
            start_date="2026-03-10",  # Tuesday
            end_date="2026-03-11",
            calendar_events=[{"summary": "PTO - dentist", "start": "2026-03-10", "end": "2026-03-11"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "holiday"

    def test_case_insensitive_keyword(self, default_config):
        """Keyword matching is case-insensitive."""
        results = classify_days(
            start_date="2026-03-10",
            end_date="2026-03-11",
            calendar_events=[{"summary": "Day Off", "start": "2026-03-10", "end": "2026-03-11"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "holiday"

    def test_holiday_on_weekend_is_holiday(self, default_config):
        """Holiday on weekend is classified as holiday (pooling happens in analysis)."""
        results = classify_days(
            start_date="2026-02-21",  # Saturday
            end_date="2026-02-22",
            calendar_events=[{"summary": "Holiday party", "start": "2026-02-21", "end": "2026-02-22"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "holiday"


class TestVacationClassification:
    def test_multi_day_vacation(self, default_config):
        """Multi-day event with vacation keyword marks all covered days."""
        results = classify_days(
            start_date="2026-03-02",  # Monday
            end_date="2026-03-07",  # to Saturday (exclusive), Mon-Fri
            calendar_events=[{"summary": "Spring vacation trip", "start": "2026-03-02", "end": "2026-03-07"}],
            person_away_days=set(),
            config=default_config,
        )
        vacation_dates = {"2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06"}
        weekday_results = [r for r in results if r.date in vacation_dates]
        for r in weekday_results:
            assert r.day_type == "vacation", f"{r.date} should be vacation, got {r.day_type}"

    def test_person_away_marks_vacation(self, default_config):
        """Days in person_away_days set become vacation."""
        results = classify_days(
            start_date="2026-02-16",  # Monday
            end_date="2026-02-17",
            calendar_events=[],
            person_away_days={"2026-02-16"},
            config=default_config,
        )
        assert results[0].day_type == "vacation"


class TestWFHClassification:
    def test_wfh_keyword(self, default_config):
        """WFH keyword on weekday classifies as wfh."""
        results = classify_days(
            start_date="2026-02-16",  # Monday
            end_date="2026-02-17",
            calendar_events=[{"summary": "WFH day", "start": "2026-02-16", "end": "2026-02-17"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "wfh"

    def test_work_from_home_keyword(self, default_config):
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-17",
            calendar_events=[{"summary": "Work from home", "start": "2026-02-16", "end": "2026-02-17"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "wfh"

    def test_wfh_on_weekend_stays_weekend(self, default_config):
        """WFH keyword on weekend stays weekend."""
        results = classify_days(
            start_date="2026-02-21",  # Saturday
            end_date="2026-02-22",
            calendar_events=[{"summary": "WFH", "start": "2026-02-21", "end": "2026-02-22"}],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "weekend"


class TestNoCalendar:
    def test_no_calendar_defaults(self, default_config):
        """Without calendar events, only weekday/weekend classification."""
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-23",
            calendar_events=[],
            person_away_days=set(),
            config=default_config,
        )
        for r in results:
            assert r.day_type in ("workday", "weekend")

    def test_empty_config_uses_defaults(self):
        """Missing config keys use sensible defaults."""
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-17",
            calendar_events=[],
            person_away_days=set(),
            config={},
        )
        assert len(results) == 1
        assert results[0].day_type == "workday"


class TestDayContextOutput:
    def test_away_all_day_flag(self, default_config):
        """Person away day sets away_all_day flag."""
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-17",
            calendar_events=[],
            person_away_days={"2026-02-16"},
            config=default_config,
        )
        assert results[0].away_all_day is True

    def test_calendar_events_stored(self, default_config):
        """Calendar event summaries stored in DayContext."""
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-17",
            calendar_events=[{"summary": "Team Meeting", "start": "2026-02-16", "end": "2026-02-16T17:00:00"}],
            person_away_days=set(),
            config=default_config,
        )
        assert "Team Meeting" in results[0].calendar_events

    def test_priority_vacation_over_wfh(self, default_config):
        """If both vacation and WFH keywords match, vacation wins."""
        results = classify_days(
            start_date="2026-02-16",
            end_date="2026-02-17",
            calendar_events=[
                {"summary": "vacation trip", "start": "2026-02-16", "end": "2026-02-17"},
                {"summary": "WFH", "start": "2026-02-16", "end": "2026-02-17"},
            ],
            person_away_days=set(),
            config=default_config,
        )
        assert results[0].day_type == "vacation"
