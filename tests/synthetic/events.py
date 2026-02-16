"""EventStreamGenerator -- converts scenario snapshots into state_changed events."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

EVENT_DOMAINS = {
    "light",
    "switch",
    "binary_sensor",
    "lock",
    "media_player",
    "cover",
    "climate",
    "vacuum",
    "person",
    "device_tracker",
    "fan",
    "sensor",
}


class EventStreamGenerator:
    """Convert scenario snapshots into a stream of state_changed events.

    ARIA snapshots contain structured collector output (lights, motion, locks,
    occupancy, climate, etc.) rather than raw HA entity states. This generator
    reconstructs entity-level states from the snapshot structure, diffs
    consecutive snapshots, and produces HA-style state_changed events with
    interpolated timestamps.
    """

    def __init__(self, snapshots: list[dict], seed: int = 42):
        self.snapshots = snapshots
        self.rng = random.Random(seed)

    def generate(self) -> list[dict]:
        """Generate all events from the snapshot sequence."""
        if len(self.snapshots) < 2:
            return []
        events: list[dict] = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]
            batch = self._diff_snapshots(prev, curr)
            events.extend(batch)
        events.sort(key=lambda e: e["timestamp"])
        return events

    def _diff_snapshots(self, prev: dict, curr: dict) -> list[dict]:
        """Find state changes between two consecutive snapshots."""
        prev_states = self._extract_entity_states(prev)
        curr_states = self._extract_entity_states(curr)
        prev_ts = self._parse_timestamp(prev)
        curr_ts = self._parse_timestamp(curr)
        span = (curr_ts - prev_ts).total_seconds()
        if span <= 0:
            span = 3600  # fallback 1 hour

        events: list[dict] = []
        for entity_id, new_state in curr_states.items():
            old_state = prev_states.get(entity_id)
            if old_state is None or old_state != new_state:
                offset = self.rng.uniform(0.1, 0.9) * span
                ts = prev_ts + timedelta(seconds=offset)
                domain = entity_id.split(".")[0]
                events.append(
                    {
                        "entity_id": entity_id,
                        "old_state": old_state or "unknown",
                        "new_state": new_state,
                        "timestamp": ts.isoformat(),
                        "domain": domain,
                        "attributes": self._make_attributes(entity_id, new_state),
                    }
                )
        return events

    def _extract_entity_states(self, snapshot: dict) -> dict[str, str]:
        """Reconstruct entity states from ARIA snapshot structure.

        ARIA snapshots don't store raw entity states -- they store structured
        collector output. We reverse-engineer entity states from the snapshot
        sections that correspond to tracked domains.
        """
        states: dict[str, str] = {}
        self._extract_lights(snapshot, states)
        self._extract_motion(snapshot, states)
        self._extract_locks(snapshot, states)
        self._extract_climate(snapshot, states)
        self._extract_occupancy(snapshot, states)
        self._extract_media(snapshot, states)
        self._extract_misc(snapshot, states)
        return states

    def _extract_lights(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract light entity states from snapshot."""
        lights = snapshot.get("lights", {})
        rooms_lit = lights.get("rooms_lit", [])
        lit_names = {name.lower().replace(" light", "").replace(" ", "_") for name in rooms_lit}
        for room in [
            "kitchen",
            "living_room",
            "bedroom",
            "office",
            "bathroom",
            "atrium",
            "garage",
            "porch",
        ]:
            states[f"light.{room}"] = "on" if room in lit_names else "off"

    def _extract_motion(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract binary_sensor motion states from snapshot."""
        motion = snapshot.get("motion", {})
        sensors = motion.get("sensors", {})
        for friendly_name, state in sensors.items():
            # "Kitchen Motion" -> "binary_sensor.kitchen_motion"
            entity_name = friendly_name.lower().replace(" ", "_")
            states[f"binary_sensor.{entity_name}"] = state

    def _extract_locks(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract lock entity states from snapshot."""
        locks = snapshot.get("locks", [])
        for lock_data in locks:
            name = lock_data.get("name", "")
            state = lock_data.get("state", "locked")
            # "Front Door Lock" -> "lock.front_door"
            entity_name = name.lower().replace(" lock", "").replace(" ", "_")
            states[f"lock.{entity_name}"] = state

    def _extract_climate(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract climate entity states from snapshot."""
        climate_list = snapshot.get("climate", [])
        for climate_data in climate_list:
            name = climate_data.get("name", "")
            state = climate_data.get("state", "off")
            # "Bedroom Climate" -> "climate.bedroom"
            entity_name = name.lower().replace(" climate", "").replace(" ", "_")
            states[f"climate.{entity_name}"] = state

    def _extract_occupancy(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract person and device_tracker states from snapshot."""
        occupancy = snapshot.get("occupancy", {})
        people_home = occupancy.get("people_home", [])
        people_away = occupancy.get("people_away", [])
        for name in people_home:
            states[f"person.{name.lower()}"] = "home"
            states[f"device_tracker.{name.lower()}_iphone"] = "home"
        for name in people_away:
            states[f"person.{name.lower()}"] = "not_home"
            states[f"device_tracker.{name.lower()}_iphone"] = "not_home"

    def _extract_media(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract media_player states from snapshot."""
        media = snapshot.get("media", {})
        media_active = media.get("total_active", 0)
        for room in ["living_room", "bedroom"]:
            if media_active > 0:
                states[f"media_player.{room}"] = "playing"
                media_active -= 1
            else:
                states[f"media_player.{room}"] = "idle"

    def _extract_misc(self, snapshot: dict, states: dict[str, str]) -> None:
        """Extract vacuum, cover, and sensor states from snapshot."""
        vacuum = snapshot.get("vacuum", {})
        states["vacuum.roborock"] = vacuum.get("state", "docked")

        doors = snapshot.get("doors_windows", {})
        states["cover.garage_door"] = doors.get("garage_door", "closed") if isinstance(doors, dict) else "closed"

        power = snapshot.get("power", {})
        states["sensor.total_power"] = str(round(power.get("total_watts", 0), 1))

    def _parse_timestamp(self, snapshot: dict) -> datetime:
        """Extract timestamp from snapshot using date + time_features.hour."""
        date_str = snapshot.get("date", "2026-02-01")
        # time_features.hour is the reliable source
        tf = snapshot.get("time_features", {})
        hour = tf.get("hour", 12.0)
        # Fallback to top-level hour if time_features missing
        if hour is None:
            hour = snapshot.get("hour", 12.0) or 12.0
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(hour=int(hour), minute=int((hour % 1) * 60))

    def _make_attributes(self, entity_id: str, state: str) -> dict:
        """Build minimal attributes for an event."""
        domain = entity_id.split(".")[0]
        attrs: dict = {
            "friendly_name": entity_id.replace("_", " ").replace(".", " ").title(),
        }
        if domain == "light" and state == "on":
            attrs["brightness"] = self.rng.randint(50, 255)
        elif domain == "sensor":
            attrs["device_class"] = "power"
            attrs["unit_of_measurement"] = "W"
        elif domain == "binary_sensor":
            attrs["device_class"] = "motion"
        return attrs
