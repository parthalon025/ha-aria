"""Snapshot construction — empty, intraday, daily, and aggregation."""

import fcntl
import json
import logging
import sqlite3
import statistics
import urllib.request
from datetime import datetime
from pathlib import Path

import aria.engine.collectors.extractors  # noqa: F401 — trigger decorator registration
from aria.engine.collectors.ha_api import (
    fetch_calendar_events,
    fetch_ha_states,
    fetch_weather,
    parse_weather,
)
from aria.engine.collectors.logbook import summarize_logbook
from aria.engine.collectors.registry import CollectorRegistry
from aria.engine.config import AppConfig, HolidayConfig
from aria.engine.storage.data_store import DataStore

logger = logging.getLogger(__name__)


def _validate_presence(snapshot: dict) -> None:
    """Detect all-zero presence features and set presence_valid flag.

    If all numeric presence fields are zero, marks the snapshot with
    presence_valid=False so downstream consumers can treat presence
    data as unreliable (cold-start or hub unavailable).
    """
    presence = snapshot.get("presence", {})
    if not presence:
        snapshot["presence_valid"] = False
        return

    numeric_fields = [
        presence.get("overall_probability", 0),
        presence.get("occupied_room_count", 0),
        presence.get("identified_person_count", 0),
        presence.get("camera_signal_count", 0),
    ]
    all_zero = all(v == 0 for v in numeric_fields)
    snapshot["presence_valid"] = not all_zero


def _fetch_presence_cache():
    """Fetch presence cache from hub API or SQLite fallback.

    Returns the presence data dict, or None if unavailable.
    """
    # Try hub API first (fast, already parsed)
    try:
        req = urllib.request.Request("http://127.0.0.1:8001/api/cache/presence")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            # API returns {"category": "presence", "data": {...}, ...}
            return data.get("data", data) if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("Presence hub API request failed: %s", e)

    # Fallback: direct SQLite read
    db_path = str(Path.home() / "ha-logs" / "intelligence" / "cache" / "hub.db")
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.execute("SELECT data FROM cache WHERE category = ?", ("presence",))
        row = cursor.fetchone()
        conn.close()
        if row:
            data = json.loads(row[0])
            return data.get("data", data) if isinstance(data, dict) else None
    except Exception as e:
        logger.warning("Presence SQLite fallback failed: %s", e)

    logger.debug("Presence cache unavailable — snapshot will have zero presence data")
    return None


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


def _enrich_intraday(snapshot, states):
    """Add intraday-specific enrichments after running collectors."""
    snapshot["occupancy"]["people_home_count"] = len(snapshot["occupancy"]["people_home"])

    # Enhanced lights: avg brightness, rooms lit
    snapshot["lights"]["avg_brightness"] = (
        round(snapshot["lights"]["total_brightness"] / snapshot["lights"]["on"], 1)
        if snapshot["lights"]["on"] > 0
        else 0
    )

    # Rooms lit from individual lights
    rooms_lit = []
    for s in states:
        if s["entity_id"].startswith("light.") and s.get("state") == "on":
            name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
            rooms_lit.append(name)
    snapshot["lights"]["rooms_lit"] = rooms_lit

    # Motion active count
    snapshot["motion"]["active_count"] = sum(1 for v in snapshot["motion"]["sensors"].values() if v == "on")

    # Add is_charging for EV
    for _ev_name, ev_data in snapshot["ev"].items():
        ev_data["is_charging"] = ev_data.get("charger_power_kw", 0) > 0


def build_intraday_snapshot(hour: int | None, date_str: str | None, config: AppConfig, store: DataStore) -> dict:
    """Build an intra-day snapshot capturing current state with time features.

    Uses fcntl file locking to prevent concurrent builds for the same
    hour-window (race condition fix — #22).  If another process already
    holds the lock the caller blocks until it completes.
    """
    now = datetime.now()
    if date_str is None:
        date_str = now.strftime("%Y-%m-%d")
    if hour is None:
        hour = now.hour
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

    # --- #22: File lock to prevent concurrent builds for the same hour ---
    lock_dir = store.paths.intraday_dir / date_str
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{hour:02d}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Check if snapshot for this hour already exists (deduplication)
        existing_path = lock_dir / f"{hour:02d}.json"
        if existing_path.exists():
            logger.info("Intraday snapshot already exists for %s hour %02d — skipping build", date_str, hour)
            with open(existing_path) as f:
                return json.load(f)

        # Start with the standard snapshot structure
        snapshot = build_empty_snapshot(date_str, config.holidays)
        snapshot["hour"] = hour
        snapshot["timestamp"] = timestamp

        # HA entities
        states = fetch_ha_states(config.ha)
        if states:
            # Run all registered collectors
            for name, collector_cls in CollectorRegistry.all().items():
                if name == "presence":
                    continue  # Presence uses hub cache, not HA states — handled separately below
                collector = (
                    collector_cls(safety_config=config.safety) if name == "entities_summary" else collector_cls()
                )
                collector.extract(snapshot, states)

            # Intraday-specific enrichments
            _enrich_intraday(snapshot, states)

        # Presence data (from hub cache, not HA states)
        presence_cache = _fetch_presence_cache()
        presence_collector = CollectorRegistry.get("presence")()
        presence_collector.inject_presence(snapshot, presence_cache)

        # #23: Validate presence data (cold-start detection)
        _validate_presence(snapshot)

        # Data quality flags
        snapshot["data_quality"] = {
            "ha_reachable": states is not None and len(states) > 0,
            "entity_count": len(states) if states else 0,
        }

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
    occ_curve = [
        s.get("occupancy", {}).get("people_home_count", len(s.get("occupancy", {}).get("people_home", [])))
        for s in intraday
    ]
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
            if name == "presence":
                continue  # Presence uses hub cache, not HA states — handled separately below
            collector = collector_cls(safety_config=config.safety) if name == "entities_summary" else collector_cls()
            collector.extract(snapshot, states)

    # Presence data (from hub cache, not HA states)
    presence_cache = _fetch_presence_cache()
    presence_collector = CollectorRegistry.get("presence")()
    presence_collector.inject_presence(snapshot, presence_cache)

    # #23: Validate presence data (cold-start detection)
    _validate_presence(snapshot)

    # Data quality flags
    snapshot["data_quality"] = {
        "ha_reachable": states is not None and len(states) > 0,
        "entity_count": len(states) if states else 0,
    }

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
