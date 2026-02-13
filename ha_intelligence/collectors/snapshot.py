"""Snapshot construction — empty, intraday, daily, and aggregation."""

import statistics
from datetime import datetime

from ha_intelligence.config import AppConfig, HolidayConfig
from ha_intelligence.storage.data_store import DataStore
from ha_intelligence.collectors.registry import CollectorRegistry
from ha_intelligence.collectors.ha_api import (
    fetch_ha_states,
    fetch_weather,
    parse_weather,
    fetch_calendar_events,
)
from ha_intelligence.collectors.logbook import summarize_logbook
from ha_intelligence.collectors.extractors import EntitiesSummaryCollector


def build_empty_snapshot(date_str: str, holidays_config: HolidayConfig) -> dict:
    """Build an empty snapshot with metadata filled in."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    us_holidays = holidays_config.get_holidays()
    return {
        "date": date_str,
        "day_of_week": dt.strftime("%A"),
        "is_weekend": dt.weekday() >= 5,
        "is_holiday": date_str in us_holidays if us_holidays else False,
        "holiday_name": us_holidays.get(date_str, None) if us_holidays else None,
        "weather": {},
        "calendar_events": [],
        "entities": {"total": 0, "unavailable": 0, "by_domain": {}},
        "power": {"total_watts": 0.0, "outlets": {}},
        "occupancy": {"people_home": [], "people_away": [], "device_count_home": 0},
        "climate": [],
        "locks": [],
        "lights": {"on": 0, "off": 0, "unavailable": 0, "total_brightness": 0},
        "motion": {"events_24h": 0, "sensors": {}},
        "automations": {"on": 0, "off": 0, "unavailable": 0, "fired_24h": 0},
        "ev": {},
        "logbook_summary": {"total_events": 0, "useful_events": 0, "by_domain": {}, "hourly": {}},
    }


def build_intraday_snapshot(hour: int | None, date_str: str | None,
                            config: AppConfig, store: DataStore) -> dict:
    """Build an intra-day snapshot capturing current state with time features."""
    now = datetime.now()
    if date_str is None:
        date_str = now.strftime("%Y-%m-%d")
    if hour is None:
        hour = now.hour
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Start with the standard snapshot structure
    snapshot = build_empty_snapshot(date_str, config.holidays)
    snapshot["hour"] = hour
    snapshot["timestamp"] = timestamp

    # HA entities
    states = fetch_ha_states(config.ha)
    if states:
        # Run all registered collectors
        for name, collector_cls in CollectorRegistry.all().items():
            if name == "entities_summary":
                collector = collector_cls(safety_config=config.safety)
            else:
                collector = collector_cls()
            collector.extract(snapshot, states)

        # Intraday-specific enrichments
        snapshot["occupancy"]["people_home_count"] = len(snapshot["occupancy"]["people_home"])

        # Enhanced lights: avg brightness, rooms lit
        if snapshot["lights"]["on"] > 0:
            snapshot["lights"]["avg_brightness"] = round(
                snapshot["lights"]["total_brightness"] / snapshot["lights"]["on"], 1)
        else:
            snapshot["lights"]["avg_brightness"] = 0

        # Rooms lit from individual lights
        rooms_lit = []
        for s in states:
            if s["entity_id"].startswith("light.") and s.get("state") == "on":
                name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
                rooms_lit.append(name)
        snapshot["lights"]["rooms_lit"] = rooms_lit

        # Motion active count
        snapshot["motion"]["active_count"] = sum(
            1 for v in snapshot["motion"]["sensors"].values() if v == "on")

        # Add is_charging for EV
        for ev_name, ev_data in snapshot["ev"].items():
            ev_data["is_charging"] = ev_data.get("charger_power_kw", 0) > 0

    # Note: time_features will be added by the features module when it's migrated.
    # For now, snapshot["time_features"] is not set here.

    # Weather
    weather_raw = fetch_weather(config.weather)
    snapshot["weather"] = parse_weather(weather_raw)

    # Logbook summary
    entries = store.load_logbook()
    snapshot["logbook_summary"] = summarize_logbook(entries)

    return snapshot


def aggregate_intraday_to_daily(date_str: str, store: DataStore) -> dict | None:
    """Aggregate intra-day snapshots into daily summary curves and stats.

    Returns a dict with intraday_curves, daily_aggregates, derived_features,
    and batteries_snapshot. Returns None if no intra-day data exists.
    """
    intraday = store.load_intraday_snapshots(date_str)
    if not intraday:
        return None

    # Build curves from intra-day snapshots
    power_curve = [s.get("power", {}).get("total_watts", 0) for s in intraday]
    occ_curve = [s.get("occupancy", {}).get("people_home_count",
        len(s.get("occupancy", {}).get("people_home", []))) for s in intraday]
    lights_curve = [s.get("lights", {}).get("on", 0) for s in intraday]
    motion_curve = [s.get("motion", {}).get("active_count", 0) for s in intraday]

    # Daily aggregates
    power_vals = [v for v in power_curve if v > 0] or [0]
    lights_vals = lights_curve or [0]

    daily_agg = {
        "power_mean": round(statistics.mean(power_vals), 1),
        "power_max": round(max(power_vals), 1),
        "power_min": round(min(power_vals), 1),
        "power_std": round(statistics.stdev(power_vals), 1) if len(power_vals) >= 2 else 0,
        "lights_mean": round(statistics.mean(lights_vals), 1),
        "lights_max": max(lights_vals),
        "occupancy_mean_people": round(statistics.mean(occ_curve), 1) if occ_curve else 0,
        "total_snapshots": len(intraday),
    }

    # Find peak power hour
    if power_curve:
        peak_idx = power_curve.index(max(power_curve))
        daily_agg["peak_power_hour"] = intraday[peak_idx].get("hour", peak_idx * 4)

    # Battery drain rates from first and last snapshot
    batteries_snapshot = {}
    if len(intraday) >= 2:
        first_batt = intraday[0].get("batteries", {})
        last_batt = intraday[-1].get("batteries", {})
        for eid, data in last_batt.items():
            level = data.get("level")
            if level is None:
                continue
            first_level = first_batt.get(eid, {}).get("level")
            if first_level is not None and first_level > 0:
                drain = first_level - level
                drain_rate = drain  # per day (these are same-day measurements)
                days_to_empty = round(level / drain_rate, 1) if drain_rate > 0 else 999
                batteries_snapshot[eid] = {
                    "level": level,
                    "drain_rate_per_day": round(drain_rate, 2),
                    "days_to_empty": days_to_empty,
                }

    # Derived features
    total_people = sum(occ_curve) if occ_curve else 1
    avg_people = total_people / len(occ_curve) if occ_curve else 1
    avg_power = daily_agg["power_mean"]
    derived = {
        "watts_per_person_home": round(avg_power / max(avg_people, 0.5), 1),
    }

    # EV miles driven (range delta)
    if len(intraday) >= 2:
        first_ev = intraday[0].get("ev", {}).get("TARS", {})
        last_ev = intraday[-1].get("ev", {}).get("TARS", {})
        first_range = first_ev.get("range_miles", 0) or 0
        last_range = last_ev.get("range_miles", 0) or 0
        if first_range > 0 and last_range > 0:
            derived["ev_miles_driven"] = round(abs(first_range - last_range), 1)

    return {
        "intraday_curves": {
            "power_curve": power_curve,
            "occupancy_curve": occ_curve,
            "lights_curve": lights_curve,
            "motion_events_curve": motion_curve,
        },
        "daily_aggregates": daily_agg,
        "derived_features": derived,
        "batteries_snapshot": batteries_snapshot,
    }


def build_snapshot(date_str: str | None, config: AppConfig, store: DataStore) -> dict:
    """Build a complete daily snapshot from all sources."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    snapshot = build_empty_snapshot(date_str, config.holidays)

    # HA entities
    states = fetch_ha_states(config.ha)
    if states:
        # Run all registered collectors
        for name, collector_cls in CollectorRegistry.all().items():
            if name == "entities_summary":
                collector = collector_cls(safety_config=config.safety)
            else:
                collector = collector_cls()
            collector.extract(snapshot, states)

    # Weather
    weather_raw = fetch_weather(config.weather)
    snapshot["weather"] = parse_weather(weather_raw)

    # Calendar
    snapshot["calendar_events"] = fetch_calendar_events()

    # Logbook
    entries = store.load_logbook()
    snapshot["logbook_summary"] = summarize_logbook(entries)

    # Intra-day aggregation (if intra-day snapshots exist for this date)
    intraday_agg = aggregate_intraday_to_daily(date_str, store)
    if intraday_agg:
        snapshot["intraday_curves"] = intraday_agg["intraday_curves"]
        snapshot["daily_aggregates"] = intraday_agg["daily_aggregates"]
        snapshot["derived_features"] = intraday_agg["derived_features"]
        snapshot["batteries_snapshot"] = intraday_agg["batteries_snapshot"]

    return snapshot
