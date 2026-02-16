"""Device roster and entity state generation for synthetic HA data."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from tests.synthetic.people import Person

# Unit of measurement lookup by device_class
SENSOR_UNITS: dict[str, str] = {
    "power": "W",
    "energy": "kWh",
    "temperature": "\u00b0C",
    "humidity": "%",
    "battery": "%",
    "illuminance": "lx",
    "distance": "km",
}


def _friendly_name(entity_id: str) -> str:
    """Derive a friendly name from entity_id: 'light.kitchen' -> 'Kitchen Light'."""
    _, name = entity_id.split(".", 1)
    parts = name.split("_")
    # Put domain-style suffix at end for readability
    domain = entity_id.split(".")[0]
    domain_label = {
        "light": "Light",
        "binary_sensor": "",
        "sensor": "",
        "climate": "Climate",
        "lock": "Lock",
        "media_player": "Media Player",
        "switch": "",
        "device_tracker": "",
        "automation": "",
        "vacuum": "",
        "cover": "",
        "person": "",
        "sun": "",
    }.get(domain, "")
    name_part = " ".join(p.capitalize() for p in parts)
    if domain_label and domain_label.lower() not in name_part.lower():
        return f"{name_part} {domain_label}".strip()
    return name_part


@dataclass
class Device:
    """A single HA entity with its physical characteristics."""

    entity_id: str
    domain: str
    device_class: str | None
    watts: float
    rooms: list[str]
    battery: bool = False

    def to_ha_state(self, state: str, **attrs: Any) -> dict[str, Any]:
        """Return an HA-format state dict."""
        attributes: dict[str, Any] = {"friendly_name": _friendly_name(self.entity_id)}
        if self.device_class:
            attributes["device_class"] = self.device_class
        if self.device_class in SENSOR_UNITS:
            attributes["unit_of_measurement"] = SENSOR_UNITS[self.device_class]
        attributes.update(attrs)
        return {
            "entity_id": self.entity_id,
            "state": state,
            "attributes": attributes,
        }


class DeviceRoster:
    """Collection of devices representing a home."""

    def __init__(self, devices: list[Device]) -> None:
        self.devices = devices
        self._by_id: dict[str, Device] = {d.entity_id: d for d in devices}

    @classmethod
    def typical_home(cls) -> DeviceRoster:  # noqa: C901
        """Create a roster of ~50+ entities for a typical smart home."""
        devices: list[Device] = []

        # 8 lights
        for room in ["kitchen", "living_room", "bedroom", "office", "bathroom", "atrium", "garage", "porch"]:
            devices.append(
                Device(
                    entity_id=f"light.{room}",
                    domain="light",
                    device_class=None,
                    watts=60 if room != "porch" else 25,
                    rooms=[room],
                )
            )

        # 2 persons
        for name in ["justin", "lisa"]:
            devices.append(
                Device(
                    entity_id=f"person.{name}",
                    domain="person",
                    device_class=None,
                    watts=0,
                    rooms=[],
                )
            )

        # 2 climate
        for room in ["bedroom", "living_room"]:
            devices.append(
                Device(
                    entity_id=f"climate.{room}",
                    domain="climate",
                    device_class=None,
                    watts=1500,
                    rooms=[room],
                )
            )

        # 2 locks with batteries
        for loc in ["front_door", "back_door"]:
            devices.append(
                Device(
                    entity_id=f"lock.{loc}",
                    domain="lock",
                    device_class=None,
                    watts=0,
                    rooms=["hallway"],
                    battery=True,
                )
            )

        # 4 motion sensors
        for room in ["kitchen", "living_room", "bedroom", "hallway"]:
            devices.append(
                Device(
                    entity_id=f"binary_sensor.{room}_motion",
                    domain="binary_sensor",
                    device_class="motion",
                    watts=0,
                    rooms=[room],
                )
            )

        # 4 door/window sensors
        for name in ["front_door", "back_door", "garage_door", "kitchen_window"]:
            room = "kitchen" if "kitchen" in name else ("garage" if "garage" in name else "hallway")
            devices.append(
                Device(
                    entity_id=f"binary_sensor.{name}",
                    domain="binary_sensor",
                    device_class="door" if "door" in name else "window",
                    watts=0,
                    rooms=[room],
                )
            )

        # 1 power sensor
        devices.append(
            Device(
                entity_id="sensor.total_power",
                domain="sensor",
                device_class="power",
                watts=0,
                rooms=[],
            )
        )

        # 3 temperature sensors
        for loc in ["bedroom", "living_room", "outside"]:
            devices.append(
                Device(
                    entity_id=f"sensor.{loc}_temperature",
                    domain="sensor",
                    device_class="temperature",
                    watts=0,
                    rooms=[loc] if loc != "outside" else [],
                )
            )

        # 2 battery sensors (for locks)
        for loc in ["front_door", "back_door"]:
            devices.append(
                Device(
                    entity_id=f"sensor.{loc}_lock_battery",
                    domain="sensor",
                    device_class="battery",
                    watts=0,
                    rooms=["hallway"],
                )
            )

        # 2 media players
        for room in ["living_room", "bedroom"]:
            devices.append(
                Device(
                    entity_id=f"media_player.{room}",
                    domain="media_player",
                    device_class=None,
                    watts=80,
                    rooms=[room],
                )
            )

        # 3 device trackers
        for name in ["justin_iphone", "lisa_iphone", "justin_macbook"]:
            devices.append(
                Device(
                    entity_id=f"device_tracker.{name}",
                    domain="device_tracker",
                    device_class=None,
                    watts=0,
                    rooms=[],
                )
            )

        # 3 switches
        devices.append(
            Device(
                entity_id="switch.coffee_maker",
                domain="switch",
                device_class=None,
                watts=900,
                rooms=["kitchen"],
            )
        )
        devices.append(
            Device(
                entity_id="switch.office_fan",
                domain="switch",
                device_class=None,
                watts=45,
                rooms=["office"],
            )
        )
        devices.append(
            Device(
                entity_id="switch.garage_opener",
                domain="switch",
                device_class=None,
                watts=200,
                rooms=["garage"],
            )
        )

        # 4 automations
        for name in ["arrive_justin", "arrive_lisa", "bedtime", "morning_lights"]:
            devices.append(
                Device(
                    entity_id=f"automation.{name}",
                    domain="automation",
                    device_class=None,
                    watts=0,
                    rooms=[],
                )
            )

        # 3 EV sensors
        devices.append(
            Device(
                entity_id="sensor.luda_battery",
                domain="sensor",
                device_class="battery",
                watts=0,
                rooms=["garage"],
            )
        )
        devices.append(
            Device(
                entity_id="sensor.luda_charger_power",
                domain="sensor",
                device_class="power",
                watts=0,
                rooms=["garage"],
            )
        )
        devices.append(
            Device(
                entity_id="sensor.luda_range",
                domain="sensor",
                device_class="distance",
                watts=0,
                rooms=["garage"],
            )
        )

        # 1 sun
        devices.append(
            Device(
                entity_id="sun.sun",
                domain="sun",
                device_class=None,
                watts=0,
                rooms=[],
            )
        )

        # 1 vacuum
        devices.append(
            Device(
                entity_id="vacuum.roborock",
                domain="vacuum",
                device_class=None,
                watts=65,
                rooms=["living_room"],
            )
        )

        # 1 cover
        devices.append(
            Device(
                entity_id="cover.garage_door",
                domain="cover",
                device_class="garage",
                watts=150,
                rooms=["garage"],
            )
        )

        return cls(devices)

    def get_devices_in_room(self, room: str) -> list[Device]:
        return [d for d in self.devices if room in d.rooms]

    def get_devices_by_domain(self, domain: str) -> list[Device]:
        return [d for d in self.devices if d.domain == domain]

    def get_device(self, entity_id: str) -> Device | None:
        return self._by_id.get(entity_id)


class EntityStateGenerator:
    """Generate HA-format state dicts for all entities at a point in time."""

    def __init__(
        self,
        roster: DeviceRoster,
        people: list[Person],
        seed: int = 0,
    ) -> None:
        self.roster = roster
        self.people = people
        self.seed = seed

    def generate_states(
        self,
        day: int,
        hour: float,
        is_weekend: bool,
        sunrise: float = 6.5,
        sunset: float = 18.5,
    ) -> list[dict[str, Any]]:
        """Generate all entity states for a given moment."""
        rng = random.Random(self.seed * 10000 + day * 100 + int(hour * 10))
        is_dark = hour < sunrise or hour > sunset

        # Resolve person locations
        person_locations: dict[str, str] = {}
        for person in self.people:
            transitions = person.get_room_transitions(day, is_weekend, self.seed)
            # Find current room: last transition at or before current hour
            current_room = "bedroom"  # default (sleeping)
            for t_hour, room in transitions:
                if t_hour <= hour:
                    current_room = room
                else:
                    break
            person_locations[person.name] = current_room

        occupied_rooms = {room for room in person_locations.values() if room != "away"}
        anyone_home = len(occupied_rooms) > 0
        late_night = hour >= 23 or hour < 5

        states: list[dict[str, Any]] = []
        active_watts = 150.0  # base load (fridge, router, standby)

        for device in self.roster.devices:
            state = self._generate_device_state(
                device,
                rng,
                hour,
                is_dark,
                is_weekend,
                person_locations,
                occupied_rooms,
                anyone_home,
                late_night,
            )
            states.append(state)
            # Track power for active devices
            if device.watts > 0 and state["state"] in ("on", "playing", "heat", "cool", "cleaning"):
                active_watts += device.watts

        # Update total power sensor
        for s in states:
            if s["entity_id"] == "sensor.total_power":
                s["state"] = str(round(active_watts, 1))
                break

        return states

    def _generate_device_state(  # noqa: C901, PLR0911, PLR0912, PLR0913, PLR0915
        self,
        device: Device,
        rng: random.Random,
        hour: float,
        is_dark: bool,
        is_weekend: bool,
        person_locations: dict[str, str],
        occupied_rooms: set[str],
        anyone_home: bool,
        late_night: bool,
    ) -> dict[str, Any]:
        domain = device.domain
        eid = device.entity_id

        # --- Person ---
        if domain == "person":
            name = eid.split(".")[1]
            loc = person_locations.get(name, "home")
            ha_state = "not_home" if loc == "away" else "home"
            return device.to_ha_state(ha_state)

        # --- Device tracker ---
        if domain == "device_tracker":
            # Match owner to person location
            for pname, loc in person_locations.items():
                if pname in eid:
                    ha_state = "not_home" if loc == "away" else "home"
                    return device.to_ha_state(ha_state)
            return device.to_ha_state("home")

        # --- Light ---
        if domain == "light":
            room_occupied = any(r in occupied_rooms for r in device.rooms)
            if room_occupied and is_dark and rng.random() > 0.3:
                brightness = rng.randint(120, 255)
                return device.to_ha_state("on", brightness=brightness)
            # Small chance lights on in occupied room even when not dark
            if room_occupied and not is_dark and rng.random() < 0.1:
                return device.to_ha_state("on", brightness=rng.randint(80, 180))
            return device.to_ha_state("off")

        # --- Binary sensor (motion) ---
        if domain == "binary_sensor" and device.device_class == "motion":
            room_occupied = any(r in occupied_rooms for r in device.rooms)
            if room_occupied and rng.random() < 0.4:
                return device.to_ha_state("on")
            return device.to_ha_state("off")

        # --- Binary sensor (door/window) ---
        if domain == "binary_sensor" and device.device_class in ("door", "window"):
            # Doors/windows mostly closed; small chance open during day if home
            if anyone_home and not late_night and rng.random() < 0.1:
                return device.to_ha_state("on")
            return device.to_ha_state("off")

        # --- Lock ---
        if domain == "lock":
            if not anyone_home or late_night:
                return device.to_ha_state("locked")
            # Small chance unlocked during day
            if rng.random() < 0.15:
                return device.to_ha_state("unlocked")
            return device.to_ha_state("locked")

        # --- Climate ---
        if domain == "climate":
            if not anyone_home:
                return device.to_ha_state("off", temperature=20, current_temperature=20.0)
            if hour >= 22 or hour < 6:
                return device.to_ha_state("heat", temperature=18, current_temperature=18.5)
            return device.to_ha_state("heat", temperature=21, current_temperature=21.0 + rng.random())

        # --- Media player ---
        if domain == "media_player":
            room_occupied = any(r in occupied_rooms for r in device.rooms)
            evening = 18 <= hour <= 23
            if room_occupied and evening and rng.random() < 0.5:
                return device.to_ha_state(
                    "playing", media_content_type="music", volume_level=round(rng.uniform(0.2, 0.6), 2)
                )
            return device.to_ha_state("idle")

        # --- Switch ---
        if domain == "switch":
            if "coffee" in eid:
                morning = 6 <= hour <= 9
                if anyone_home and morning and rng.random() < 0.6:
                    return device.to_ha_state("on")
                return device.to_ha_state("off")
            if "fan" in eid:
                room_occupied = any(r in occupied_rooms for r in device.rooms)
                if room_occupied and rng.random() < 0.4:
                    return device.to_ha_state("on")
                return device.to_ha_state("off")
            if "garage" in eid:
                return device.to_ha_state("off")
            return device.to_ha_state("off")

        # --- Automation ---
        if domain == "automation":
            return device.to_ha_state("on")

        # --- Sensor (temperature) ---
        if device.device_class == "temperature":
            if "outside" in eid:
                # Simple sinusoidal outdoor temp
                base = 10 + 8 * math.sin((hour - 6) * math.pi / 12)
                temp = round(base + rng.uniform(-1, 1), 1)
            else:
                temp = round(20.5 + rng.uniform(-1.5, 1.5), 1)
            return device.to_ha_state(str(temp))

        # --- Sensor (battery) ---
        if device.device_class == "battery":
            level = rng.randint(40, 95) if "luda" in eid else rng.randint(70, 100)
            return device.to_ha_state(str(level))

        # --- Sensor (power) ---
        if device.device_class == "power":
            if "luda_charger" in eid:
                # Charger: on overnight, off during day
                if hour >= 22 or hour < 6:
                    return device.to_ha_state(str(round(rng.uniform(6000, 7500), 1)))
                return device.to_ha_state("0.0")
            # total_power handled after loop
            return device.to_ha_state("0.0")

        # --- Sensor (distance / EV range) ---
        if device.device_class == "distance":
            return device.to_ha_state(str(rng.randint(150, 380)))

        # --- Sun ---
        if domain == "sun":
            if is_dark:
                return device.to_ha_state("below_horizon", elevation=round(-5 - rng.random() * 10, 1))
            return device.to_ha_state(
                "above_horizon", elevation=round(10 + 30 * math.sin((hour - 6) * math.pi / 12), 1)
            )

        # --- Vacuum ---
        if domain == "vacuum":
            # Runs mid-day when nobody home on weekdays
            if not anyone_home and 10 <= hour <= 14 and not is_weekend and rng.random() < 0.3:
                return device.to_ha_state("cleaning")
            return device.to_ha_state("docked")

        # --- Cover ---
        if domain == "cover":
            return device.to_ha_state("closed")

        # Fallback
        return device.to_ha_state("unknown")
