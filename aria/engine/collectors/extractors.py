"""Domain data extractors — registered collectors for each HA device domain.

Each collector class extracts data from raw HA entity states into the snapshot dict.
Registration happens via @CollectorRegistry.register() decorator at import time.
"""

import contextlib
import logging

from aria.engine.collectors.registry import BaseCollector, CollectorRegistry
from aria.engine.config import SafetyConfig

logger = logging.getLogger(__name__)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _time_to_minutes(time_str):
    """Convert HH:MM string to minutes since midnight."""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


@CollectorRegistry.register("power")
class PowerCollector(BaseCollector):
    """Extract USP PDU Pro power data."""

    def extract(self, snapshot, states):
        total = 0.0
        outlets = {}
        for s in states:
            eid = s["entity_id"]
            if "usp_pdu_pro" in eid and "outlet" in eid and "power" in eid:
                name = s.get("attributes", {}).get("friendly_name", eid)
                try:
                    watts = float(s["state"])
                    outlets[name] = watts
                    total += watts
                except (ValueError, TypeError):
                    logger.warning("Failed to parse power value for %s: %r", eid, s.get("state"))
            elif eid == "sensor.usp_pdu_pro_ac_power_consumption":
                with contextlib.suppress(ValueError, TypeError):
                    total = float(s["state"])
        snapshot["power"]["total_watts"] = total
        snapshot["power"]["outlets"] = outlets


@CollectorRegistry.register("occupancy")
class OccupancyCollector(BaseCollector):
    """Extract person and device tracker occupancy."""

    def extract(self, snapshot, states):
        home = []
        away = []
        device_home = 0
        for s in states:
            eid = s["entity_id"]
            state = s.get("state", "")
            name = s.get("attributes", {}).get("friendly_name", eid)
            if eid.startswith("person."):
                if state == "home":
                    home.append(name)
                elif state == "not_home":
                    away.append(name)
            elif eid.startswith("device_tracker.") and state == "home":
                device_home += 1
        snapshot["occupancy"]["people_home"] = home
        snapshot["occupancy"]["people_away"] = away
        snapshot["occupancy"]["device_count_home"] = device_home


@CollectorRegistry.register("climate")
class ClimateCollector(BaseCollector):
    """Extract climate/thermostat data."""

    def extract(self, snapshot, states):
        zones = []
        for s in states:
            if s["entity_id"].startswith("climate."):
                attrs = s.get("attributes", {})
                name = attrs.get("friendly_name", s["entity_id"])
                if "tars" in name.lower() or "tessy" in name.lower():
                    continue
                zones.append(
                    {
                        "name": name,
                        "state": s.get("state", "unknown"),
                        "current_temp": attrs.get("current_temperature"),
                        "target_temp": attrs.get("temperature"),
                        "hvac_action": attrs.get("hvac_action", ""),
                    }
                )
        snapshot["climate"] = zones


@CollectorRegistry.register("lights")
class LightsCollector(BaseCollector):
    """Extract light state summary."""

    def extract(self, snapshot, states):
        on = off = unavail = 0
        total_brightness = 0
        for s in states:
            if s["entity_id"].startswith("light."):
                state = s.get("state", "")
                if state == "on":
                    on += 1
                    total_brightness += s.get("attributes", {}).get("brightness", 0) or 0
                elif state == "off":
                    off += 1
                elif state == "unavailable":
                    unavail += 1
        snapshot["lights"]["on"] = on
        snapshot["lights"]["off"] = off
        snapshot["lights"]["unavailable"] = unavail
        snapshot["lights"]["total_brightness"] = total_brightness


@CollectorRegistry.register("locks")
class LocksCollector(BaseCollector):
    """Extract lock states and battery levels."""

    def extract(self, snapshot, states):
        locks = []
        for s in states:
            if s["entity_id"].startswith("lock."):
                attrs = s.get("attributes", {})
                name = attrs.get("friendly_name", s["entity_id"])
                locks.append(
                    {
                        "name": name,
                        "state": s.get("state", "unknown"),
                        "battery": attrs.get("battery_level"),
                    }
                )
        snapshot["locks"] = locks


@CollectorRegistry.register("automations")
class AutomationsCollector(BaseCollector):
    """Extract automation summary."""

    def extract(self, snapshot, states):
        on = off = unavail = 0
        for s in states:
            if s["entity_id"].startswith("automation."):
                state = s.get("state", "")
                if state == "on":
                    on += 1
                elif state == "off":
                    off += 1
                elif state == "unavailable":
                    unavail += 1
        snapshot["automations"]["on"] = on
        snapshot["automations"]["off"] = off
        snapshot["automations"]["unavailable"] = unavail


@CollectorRegistry.register("motion")
class MotionCollector(BaseCollector):
    """Extract motion sensor data."""

    def extract(self, snapshot, states):
        sensors = {}
        for s in states:
            if s["entity_id"].startswith("binary_sensor."):
                dc = s.get("attributes", {}).get("device_class", "")
                if dc == "motion":
                    name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
                    sensors[name] = s.get("state", "off")
        snapshot["motion"]["sensors"] = sensors


@CollectorRegistry.register("ev")
class EVCollector(BaseCollector):
    """Extract Tesla/EV data. Maps luda_* entities to TARS."""

    def extract(self, snapshot, states):
        ev_data = {}
        for s in states:
            eid = s["entity_id"]
            state_val = s.get("state", "")
            attrs = s.get("attributes", {})
            if "luda_battery" in eid and attrs.get("unit_of_measurement") == "%":
                ev_data.setdefault("TARS", {})["battery_pct"] = _safe_float(state_val)
            elif "luda_charger_power" in eid:
                ev_data.setdefault("TARS", {})["charger_power_kw"] = _safe_float(state_val)
            elif "luda_range" in eid and "mi" in str(attrs.get("unit_of_measurement", "")):
                ev_data.setdefault("TARS", {})["range_miles"] = _safe_float(state_val)
            elif "luda_charging_rate" in eid:
                ev_data.setdefault("TARS", {})["charging_rate_mph"] = _safe_float(state_val)
            elif "luda_energy_added" in eid:
                ev_data.setdefault("TARS", {})["energy_added_kwh"] = _safe_float(state_val)
        snapshot["ev"] = ev_data


@CollectorRegistry.register("entities_summary")
class EntitiesSummaryCollector(BaseCollector):
    """Extract high-level entity counts."""

    def __init__(self, safety_config: SafetyConfig | None = None):
        self._unavailable_exclude_domains = (
            safety_config.unavailable_exclude_domains if safety_config else {"update", "tts", "stt"}
        )

    def extract(self, snapshot, states):
        total = len(states)
        unavail = 0
        by_domain = {}
        for s in states:
            eid = s["entity_id"]
            domain = eid.split(".")[0] if "." in eid else "unknown"
            by_domain[domain] = by_domain.get(domain, 0) + 1
            if s.get("state") == "unavailable" and domain not in self._unavailable_exclude_domains:
                unavail += 1
        snapshot["entities"]["total"] = total
        snapshot["entities"]["unavailable"] = unavail
        snapshot["entities"]["by_domain"] = by_domain
        snapshot["entities"]["unavailable_list"] = [
            s["entity_id"]
            for s in states
            if s.get("state") == "unavailable" and s["entity_id"].split(".")[0] not in self._unavailable_exclude_domains
        ]


@CollectorRegistry.register("doors_windows")
class DoorsWindowsCollector(BaseCollector):
    """Extract door/window binary sensor data."""

    def extract(self, snapshot, states):
        dw = {}
        for s in states:
            if s["entity_id"].startswith("binary_sensor."):
                dc = s.get("attributes", {}).get("device_class", "")
                if dc in ("door", "window", "garage_door"):
                    name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
                    dw[name] = {"state": s.get("state", "unknown"), "open_count_today": 0}
        snapshot["doors_windows"] = dw


@CollectorRegistry.register("batteries")
class BatteriesCollector(BaseCollector):
    """Extract battery levels from all entities with battery_level attribute."""

    def extract(self, snapshot, states):
        batteries = {}
        for s in states:
            attrs = s.get("attributes", {})
            battery = attrs.get("battery_level")
            if battery is None:
                # Also check device_class battery sensors
                if s["entity_id"].startswith("sensor.") and attrs.get("device_class") == "battery":
                    try:
                        battery = float(s["state"])
                    except (ValueError, TypeError):
                        continue
                else:
                    continue
            batteries[s["entity_id"]] = {
                "level": battery,
                "entity_type": s["entity_id"].split(".")[0],
            }
        snapshot["batteries"] = batteries


@CollectorRegistry.register("network")
class NetworkCollector(BaseCollector):
    """Extract device_tracker domain summary."""

    def extract(self, snapshot, states):
        home = away = unavail = 0
        for s in states:
            if s["entity_id"].startswith("device_tracker."):
                st = s.get("state", "")
                if st == "home":
                    home += 1
                elif st == "not_home":
                    away += 1
                elif st == "unavailable":
                    unavail += 1
        snapshot["network"] = {
            "devices_home": home,
            "devices_away": away,
            "devices_unavailable": unavail,
        }


@CollectorRegistry.register("media")
class MediaCollector(BaseCollector):
    """Extract media_player state summary."""

    def extract(self, snapshot, states):
        active = []
        for s in states:
            if s["entity_id"].startswith("media_player.") and s.get("state") == "playing":
                name = s.get("attributes", {}).get("friendly_name", s["entity_id"])
                active.append(name)
        snapshot["media"] = {"active_players": active, "total_active": len(active)}


@CollectorRegistry.register("sun")
class SunCollector(BaseCollector):
    """Extract sun.sun entity for sunrise/sunset data."""

    def extract(self, snapshot, states):
        sun_data = {"sunrise": "06:00", "sunset": "18:00", "daylight_hours": 12.0, "solar_elevation": 0}
        for s in states:
            if s["entity_id"] == "sun.sun":
                attrs = s.get("attributes", {})
                # HA sun entity has next_rising, next_setting, elevation
                rising = attrs.get("next_rising", "")
                setting = attrs.get("next_setting", "")
                if rising and len(rising) >= 16:
                    sun_data["sunrise"] = rising[11:16]
                if setting and len(setting) >= 16:
                    sun_data["sunset"] = setting[11:16]
                sun_data["solar_elevation"] = attrs.get("elevation", 0) or 0
                # Compute daylight hours
                with contextlib.suppress(Exception):
                    sr = _time_to_minutes(sun_data["sunrise"])
                    ss = _time_to_minutes(sun_data["sunset"])
                    sun_data["daylight_hours"] = round(max(0, ss - sr) / 60.0, 2)
                break
        snapshot["sun"] = sun_data


@CollectorRegistry.register("vacuum")
class VacuumCollector(BaseCollector):
    """Extract vacuum domain data."""

    def extract(self, snapshot, states):
        vacuums = {}
        for s in states:
            if s["entity_id"].startswith("vacuum."):
                attrs = s.get("attributes", {})
                name = attrs.get("friendly_name", s["entity_id"])
                vacuums[name] = {
                    "status": s.get("state", "unknown"),
                    "battery": attrs.get("battery_level"),
                }
        snapshot["vacuum"] = vacuums


@CollectorRegistry.register("presence")
class PresenceCollector(BaseCollector):
    """Collects presence summary from hub cache for engine snapshots.

    Unlike other collectors, this does NOT extract from HA entity states.
    Instead, it receives presence cache data separately via the presence_cache parameter.
    Call extract() with states=[] and pass presence_cache as a keyword argument,
    or call inject_presence() directly.
    """

    def extract(self, snapshot, states, **kwargs):
        """Standard collector interface — delegates to inject_presence."""
        presence_cache = kwargs.get("presence_cache")
        self.inject_presence(snapshot, presence_cache)

    def inject_presence(self, snapshot, presence_cache=None):
        """Inject presence summary into snapshot from hub cache data."""
        if not presence_cache:
            snapshot["presence"] = {
                "overall_probability": 0,
                "occupied_room_count": 0,
                "identified_person_count": 0,
                "camera_signal_count": 0,
                "rooms": {},
            }
            return

        rooms = presence_cache.get("rooms", {})
        occupied = [r for r, d in rooms.items() if d.get("probability", 0) > 0.5]
        persons = presence_cache.get("identified_persons", {})

        # Count camera signals across all rooms
        camera_signals = 0
        for _r, d in rooms.items():
            for s in d.get("signals", []):
                if isinstance(s, dict) and s.get("type", "").startswith("camera_"):
                    camera_signals += 1

        # Overall probability: max of all room probabilities
        probs = [d.get("probability", 0) for d in rooms.values()]
        overall = max(probs) if probs else 0

        snapshot["presence"] = {
            "overall_probability": round(overall, 3),
            "occupied_room_count": len(occupied),
            "identified_person_count": len(persons),
            "camera_signal_count": camera_signals,
            "rooms": {
                room: {
                    "probability": round(d.get("probability", 0), 3),
                    "person_count": len(d.get("persons", [])),
                }
                for room, d in rooms.items()
            },
        }
