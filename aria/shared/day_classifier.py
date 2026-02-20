"""Day classifier — classifies each day in analysis window by type.

Uses calendar events and person-away data to classify days as
workday/weekend/holiday/vacation/wfh. This segmentation drives
per-day-type pattern detection in the normalizer pipeline.

Priority order: vacation > holiday > wfh > workday (weekends stay weekend).
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

from aria.automation.models import DayContext

logger = logging.getLogger(__name__)

DEFAULT_HOLIDAY_KEYWORDS = ["holiday", "vacation", "PTO", "trip", "out of office", "off"]
DEFAULT_WFH_KEYWORDS = ["WFH", "remote", "work from home"]


def classify_days(
    start_date: str,
    end_date: str,
    calendar_events: list[dict],
    person_away_days: set[str],
    config: dict[str, Any],
) -> list[DayContext]:
    """Classify each day in the range [start_date, end_date).

    Args:
        start_date: Start date (YYYY-MM-DD), inclusive.
        end_date: End date (YYYY-MM-DD), exclusive.
        calendar_events: List of dicts with summary, start, end.
        person_away_days: Set of YYYY-MM-DD strings where person was away all day.
        config: Config dict with calendar.holiday_keywords, calendar.wfh_keywords.

    Returns:
        List of DayContext, one per day in the range.
    """
    holiday_keywords = [k.lower() for k in config.get("calendar.holiday_keywords", DEFAULT_HOLIDAY_KEYWORDS)]
    wfh_keywords = [k.lower() for k in config.get("calendar.wfh_keywords", DEFAULT_WFH_KEYWORDS)]

    start = _parse_date(start_date)
    end = _parse_date(end_date)

    # Build per-day event index
    day_events: dict[str, list[str]] = {}
    for event in calendar_events:
        summaries = _expand_event_to_days(event, start, end)
        for day_str, summary in summaries:
            day_events.setdefault(day_str, []).append(summary)

    results = []
    current = start
    while current < end:
        day_str = current.isoformat()
        is_weekend = current.weekday() >= 5
        summaries = day_events.get(day_str, [])
        away = day_str in person_away_days

        keywords = {"holiday": holiday_keywords, "wfh": wfh_keywords}
        day_type = _classify_single_day(is_weekend, summaries, away, keywords)
        results.append(
            DayContext(
                date=day_str,
                day_type=day_type,
                calendar_events=summaries,
                away_all_day=away,
            )
        )
        current += timedelta(days=1)

    return results


def _classify_single_day(
    is_weekend: bool,
    summaries: list[str],
    away: bool,
    keywords: dict[str, list[str]],
) -> str:
    """Classify a single day. Priority: vacation > holiday > weekend > wfh > workday.

    Holidays are classified independently even on weekends — the analysis
    engine handles pooling (merging holidays into weekend pool if <10).

    Args:
        is_weekend: True for Saturday/Sunday.
        summaries: Calendar event summaries for this day.
        away: True if person was away all day.
        keywords: Dict with "holiday" and "wfh" keyword lists.
    """
    if away:
        return "vacation"

    holiday_keywords = keywords.get("holiday", [])
    wfh_keywords = keywords.get("wfh", [])

    has_vacation = False
    has_holiday = False
    has_wfh = False

    for summary in summaries:
        lower = summary.lower()
        if any(kw in lower for kw in ["vacation", "trip"]):
            has_vacation = True
        if any(kw in lower for kw in holiday_keywords):
            has_holiday = True
        if any(kw in lower for kw in wfh_keywords):
            has_wfh = True

    # Priority: vacation > holiday > weekend > wfh > workday
    if has_vacation:
        return "vacation"
    if has_holiday:
        return "holiday"
    if is_weekend:
        return "weekend"
    if has_wfh:
        return "wfh"

    return "workday"


def _expand_event_to_days(event: dict, range_start: date, range_end: date) -> list[tuple[str, str]]:
    """Expand a calendar event to per-day (day_str, summary) tuples."""
    summary = event.get("summary", "")
    event_start = _parse_date(event.get("start", ""))
    event_end_raw = event.get("end", "")
    event_end = _parse_date(event_end_raw) if event_end_raw else event_start + timedelta(days=1)

    # For intra-day events (end same day as start), ensure at least the start day is included
    if event_end <= event_start:
        event_end = event_start + timedelta(days=1)

    # Clamp to analysis range
    effective_start = max(event_start, range_start)
    effective_end = min(event_end, range_end)

    result = []
    current = effective_start
    while current < effective_end:
        result.append((current.isoformat(), summary))
        current += timedelta(days=1)

    # Handle edge case: event starts and ends within the same day
    if not result and effective_start == effective_end and range_start <= effective_start < range_end:
        result.append((effective_start.isoformat(), summary))

    return result


def _parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD or ISO datetime to date."""
    if not date_str:
        return date.today()
    # Handle full ISO datetime
    if "T" in date_str:
        return datetime.fromisoformat(date_str).date()
    return date.fromisoformat(date_str)
