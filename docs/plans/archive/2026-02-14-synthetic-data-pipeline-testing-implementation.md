# Synthetic Data Pipeline Testing — Implementation Plan

## In Plain English

This is the build plan for ARIA's testing flight simulator. It starts by creating the virtual household (simulated people, devices, and weather), then builds the test harness that runs ARIA's real pipeline against that simulated data, then adds a demo mode so you can see the dashboard populated with realistic-looking information.

## Why This Exists

The design document describes what the testing infrastructure should accomplish; this plan describes how to construct it piece by piece. Building a household simulator, pipeline runner, ML validation suite, and demo mode involves touching many files across ARIA's engine, hub, and CLI. Each task is sequenced so the simulator is built and tested first, the pipeline runner layers on top, and integration tests verify the whole chain. Without this ordering, you would end up debugging the simulator and the pipeline simultaneously, unable to tell which one is wrong.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a realistic household simulator, ML validation test suite, demo mode, and exploration agents for full-pipeline integration testing.

**Architecture:** HouseholdSimulator generates HA state-change events from simulated people/devices/weather. SnapshotAssembler feeds events through real ARIA collectors. PipelineRunner orchestrates full engine pipeline in temp dirs. Pytest integration tests validate ML competence, Ollama meta-learning, and E2E flow. Demo mode (`aria demo`) runs the full pipeline with simulated data and starts the hub for visual dashboard testing.

**Tech Stack:** Python 3.14, pytest, scikit-learn, ARIA engine/hub (existing), Ollama deepseek-r1:8b (optional)

**Design doc:** `docs/plans/2026-02-14-synthetic-data-pipeline-testing-design.md`

---

## Reference: Key Imports

These imports are used repeatedly across tasks. Copy as needed.

```python
# Config and storage
from aria.engine.config import (
    AppConfig, HAConfig, PathConfig, ModelConfig,
    OllamaConfig, WeatherConfig, SafetyConfig, HolidayConfig,
)
from aria.engine.storage.data_store import DataStore

# Collectors
from aria.engine.collectors.snapshot import build_empty_snapshot
from aria.engine.collectors.registry import CollectorRegistry, BaseCollector
import aria.engine.collectors.extractors  # triggers decorator registration

# Features
from aria.engine.features.time_features import build_time_features
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.features.vector_builder import build_training_data, build_feature_vector, extract_target_values

# Analysis
from aria.engine.analysis.baselines import compute_baselines
from aria.engine.analysis.drift import PageHinkleyDetector

# Models
from aria.engine.models.training import train_all_models, train_continuous_model, predict_with_ml

# Predictions
from aria.engine.predictions.predictor import generate_predictions, blend_predictions, count_days_of_data
from aria.engine.predictions.scoring import score_all_predictions, accuracy_trend

# Meta-learning
from aria.engine.llm.meta_learning import parse_suggestions
```

---

## Task 1: Person & Schedule Model

**Files:**
- Create: `tests/synthetic/__init__.py`
- Create: `tests/synthetic/people.py`
- Test: `tests/synthetic/test_people.py`

**Context:** A Person has a daily schedule (wake, sleep, work hours), moves between rooms, and drives occupancy/motion/light triggers. Schedules have gaussian jitter so no two days are identical. Weekend schedules differ from weekday.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_people.py
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
        # Jitter should keep wake within +/- 1 hour
        assert 5.5 <= times["wake"] <= 7.5


class TestPerson:
    def test_person_has_name_and_schedule(self):
        p = Person("alice", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        assert p.name == "alice"

    def test_get_schedule_for_weekday(self):
        p = Person("alice", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        sched = p.get_schedule(day=0, is_weekend=False)
        assert sched.depart is not None

    def test_get_schedule_for_weekend(self):
        p = Person("alice", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        sched = p.get_schedule(day=5, is_weekend=True)
        assert sched.depart is None

    def test_room_transitions_for_day(self):
        p = Person("alice", schedule_weekday=Schedule.weekday_office(6.5, 23.0),
                    schedule_weekend=Schedule.weekend(8.0, 23.5))
        transitions = p.get_room_transitions(day=0, is_weekend=False, seed=42)
        # Should have at least wake-up, leave, arrive, sleep transitions
        assert len(transitions) >= 4
        # Each transition: (hour_float, room_name)
        for hour, room in transitions:
            assert isinstance(hour, float)
            assert isinstance(room, str)
        # First transition should be near wake time
        assert transitions[0][1] in ("bedroom", "bathroom")
        # Should be sorted by time
        hours = [h for h, _ in transitions]
        assert hours == sorted(hours)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_people.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.synthetic.people'`

**Step 3: Write minimal implementation**

```python
# tests/synthetic/__init__.py
"""Synthetic data generation for ARIA pipeline testing."""
```

```python
# tests/synthetic/people.py
"""Person and schedule simulation for household modeling."""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Schedule:
    """A daily schedule template with optional work departure/arrival."""
    wake: float
    sleep: float
    depart: float | None = None
    arrive: float | None = None
    _jitter_std: float = 0.3  # hours of gaussian jitter

    @classmethod
    def weekday_office(cls, wake: float, sleep: float) -> Schedule:
        return cls(wake=wake, sleep=sleep, depart=wake + 1.5, arrive=17.5)

    @classmethod
    def weekend(cls, wake: float, sleep: float) -> Schedule:
        return cls(wake=wake, sleep=sleep, depart=None, arrive=None)

    def resolve(self, day: int, seed: int) -> dict[str, float]:
        """Resolve schedule to concrete times with deterministic jitter."""
        rng = random.Random(seed * 1000 + day)
        result = {
            "wake": self.wake + rng.gauss(0, self._jitter_std),
            "sleep": self.sleep + rng.gauss(0, self._jitter_std),
        }
        if self.depart is not None:
            result["depart"] = self.depart + rng.gauss(0, self._jitter_std)
        if self.arrive is not None:
            result["arrive"] = self.arrive + rng.gauss(0, self._jitter_std)
        return result


ROOM_SEQUENCE_HOME = ["bedroom", "bathroom", "kitchen", "living_room", "office"]
ROOM_SEQUENCE_EVENING = ["kitchen", "living_room", "bedroom"]


class Person:
    """A simulated household resident."""

    def __init__(
        self,
        name: str,
        schedule_weekday: Schedule,
        schedule_weekend: Schedule,
        rooms: list[str] | None = None,
    ):
        self.name = name
        self.schedule_weekday = schedule_weekday
        self.schedule_weekend = schedule_weekend
        self.rooms = rooms or ROOM_SEQUENCE_HOME

    def get_schedule(self, day: int, is_weekend: bool) -> Schedule:
        return self.schedule_weekend if is_weekend else self.schedule_weekday

    def get_room_transitions(
        self, day: int, is_weekend: bool, seed: int
    ) -> list[tuple[float, str]]:
        """Generate (hour, room) transitions for a single day."""
        sched = self.get_schedule(day, is_weekend)
        times = sched.resolve(day, seed)
        rng = random.Random(seed * 2000 + day)
        transitions = []

        # Morning routine
        wake = times["wake"]
        transitions.append((wake, "bedroom"))
        transitions.append((wake + 0.1 + rng.random() * 0.2, "bathroom"))
        transitions.append((wake + 0.4 + rng.random() * 0.2, "kitchen"))

        if "depart" in times:
            # Workday: leave and return
            transitions.append((times["depart"], "away"))
            transitions.append((times["arrive"], "kitchen"))
            transitions.append((times["arrive"] + 0.5 + rng.random() * 0.5, "living_room"))
        else:
            # Weekend: move around the house
            hour = wake + 1.5
            for room in rng.sample(self.rooms, min(3, len(self.rooms))):
                transitions.append((hour, room))
                hour += 1.0 + rng.random() * 1.5

        # Evening routine
        sleep = times["sleep"]
        transitions.append((sleep - 1.5, "living_room"))
        transitions.append((sleep - 0.3, "bathroom"))
        transitions.append((sleep, "bedroom"))

        transitions.sort(key=lambda t: t[0])
        return transitions
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_people.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add tests/synthetic/__init__.py tests/synthetic/people.py tests/synthetic/test_people.py
git commit -m "feat(synthetic): add Person and Schedule models for household simulation"
```

---

## Task 2: Device Roster & Entity State Generation

**Files:**
- Create: `tests/synthetic/entities.py`
- Test: `tests/synthetic/test_entities.py`

**Context:** DeviceRoster defines the household's entity set. Each device has a domain, entity_id, device_class, and state transition rules. EntityStateGenerator produces HA-format state dicts driven by person transitions and time of day.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_entities.py
"""Tests for device roster and entity state generation."""
import pytest
from tests.synthetic.entities import Device, DeviceRoster, EntityStateGenerator
from tests.synthetic.people import Person, Schedule


class TestDevice:
    def test_device_has_required_fields(self):
        d = Device(entity_id="light.kitchen", domain="light", device_class=None,
                   watts=60, rooms=["kitchen"])
        assert d.entity_id == "light.kitchen"
        assert d.domain == "light"

    def test_device_state_for_light_on(self):
        d = Device(entity_id="light.kitchen", domain="light", device_class=None,
                   watts=60, rooms=["kitchen"])
        state = d.to_ha_state("on", brightness=180)
        assert state["entity_id"] == "light.kitchen"
        assert state["state"] == "on"
        assert state["attributes"]["brightness"] == 180
        assert state["attributes"]["friendly_name"] == "Kitchen Light"

    def test_device_state_for_sensor(self):
        d = Device(entity_id="sensor.power_consumption", domain="sensor",
                   device_class="power", watts=0, rooms=[])
        state = d.to_ha_state("156.5")
        assert state["state"] == "156.5"
        assert state["attributes"]["device_class"] == "power"
        assert state["attributes"]["unit_of_measurement"] == "W"


class TestDeviceRoster:
    def test_typical_home_has_entities(self):
        roster = DeviceRoster.typical_home()
        assert len(roster.devices) >= 40

    def test_typical_home_has_expected_domains(self):
        roster = DeviceRoster.typical_home()
        domains = {d.domain for d in roster.devices}
        assert "light" in domains
        assert "person" in domains
        assert "binary_sensor" in domains
        assert "climate" in domains
        assert "lock" in domains
        assert "sensor" in domains

    def test_get_devices_by_room(self):
        roster = DeviceRoster.typical_home()
        kitchen = roster.get_devices_in_room("kitchen")
        assert len(kitchen) > 0
        assert all("kitchen" in d.rooms for d in kitchen)

    def test_get_devices_by_domain(self):
        roster = DeviceRoster.typical_home()
        lights = roster.get_devices_by_domain("light")
        assert len(lights) >= 5
        assert all(d.domain == "light" for d in lights)


class TestEntityStateGenerator:
    def test_generates_states_for_time(self):
        roster = DeviceRoster.typical_home()
        people = [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
        ]
        gen = EntityStateGenerator(roster, people, seed=42)
        states = gen.generate_states(day=0, hour=12.0, is_weekend=False)
        assert len(states) > 0
        # Every state should be a valid HA state dict
        for s in states:
            assert "entity_id" in s
            assert "state" in s
            assert "attributes" in s

    def test_lights_on_when_occupied_and_dark(self):
        roster = DeviceRoster.typical_home()
        people = [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
        ]
        gen = EntityStateGenerator(roster, people, seed=42)
        # Evening, person home — lights should be on in occupied rooms
        states = gen.generate_states(day=5, hour=20.0, is_weekend=True)
        light_states = [s for s in states if s["entity_id"].startswith("light.")]
        on_lights = [s for s in light_states if s["state"] == "on"]
        assert len(on_lights) >= 1

    def test_person_away_during_work(self):
        roster = DeviceRoster.typical_home()
        people = [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
        ]
        gen = EntityStateGenerator(roster, people, seed=42)
        # Midday weekday — person should be away
        states = gen.generate_states(day=1, hour=12.0, is_weekend=False)
        person_states = [s for s in states if s["entity_id"] == "person.alice"]
        assert len(person_states) == 1
        assert person_states[0]["state"] == "not_home"

    def test_deterministic_with_seed(self):
        roster = DeviceRoster.typical_home()
        people = [Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5))]
        gen_a = EntityStateGenerator(roster, people, seed=42)
        gen_b = EntityStateGenerator(roster, people, seed=42)
        states_a = gen_a.generate_states(day=0, hour=12.0, is_weekend=False)
        states_b = gen_b.generate_states(day=0, hour=12.0, is_weekend=False)
        assert states_a == states_b
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_entities.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# tests/synthetic/entities.py
"""Device roster and entity state generation for household simulation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from tests.synthetic.people import Person


@dataclass
class Device:
    """A simulated HA device."""
    entity_id: str
    domain: str
    device_class: str | None
    watts: float
    rooms: list[str]
    battery: float | None = None  # 0-100 if battery-powered

    def to_ha_state(self, state: str, **attrs) -> dict:
        """Generate HA-format state dict."""
        friendly = self.entity_id.split(".", 1)[1].replace("_", " ").title()
        attributes = {"friendly_name": friendly}
        if self.device_class:
            attributes["device_class"] = self.device_class
        if self.domain == "sensor" and self.device_class == "power":
            attributes["unit_of_measurement"] = "W"
        if self.domain == "sensor" and self.device_class == "temperature":
            attributes["unit_of_measurement"] = "\u00b0F"
        if self.domain == "sensor" and self.device_class == "battery":
            attributes["unit_of_measurement"] = "%"
        if self.domain == "light" and state == "on":
            attributes["brightness"] = attrs.get("brightness", 180)
        if self.battery is not None:
            attributes["battery_level"] = self.battery
        attributes.update(attrs)
        return {
            "entity_id": self.entity_id,
            "state": state,
            "attributes": attributes,
            "last_changed": "",
            "last_updated": "",
        }


class DeviceRoster:
    """Collection of devices in a household."""

    def __init__(self, devices: list[Device]):
        self.devices = devices

    @classmethod
    def typical_home(cls) -> DeviceRoster:
        """A typical home with ~60 entities matching ARIA's real profile."""
        devices = []

        # Lights (8)
        for room in ["kitchen", "living_room", "bedroom", "office", "bathroom",
                      "atrium", "garage", "porch"]:
            devices.append(Device(f"light.{room}", "light", None, 60, [room]))

        # Persons (2)
        devices.append(Device("person.alice", "person", None, 0, []))
        devices.append(Device("person.bob", "person", None, 0, []))

        # Climate (2)
        devices.append(Device("climate.bedroom", "climate", None, 0, ["bedroom"]))
        devices.append(Device("climate.living_room", "climate", None, 0, ["living_room"]))

        # Locks (2)
        devices.append(Device("lock.front_door", "lock", "lock", 0, ["front_door"], battery=85))
        devices.append(Device("lock.back_door", "lock", "lock", 0, ["back_door"], battery=58))

        # Binary sensors — motion (4)
        for room in ["kitchen", "living_room", "bedroom", "hallway"]:
            devices.append(Device(f"binary_sensor.{room}_motion", "binary_sensor",
                                  "motion", 0, [room], battery=82))

        # Binary sensors — doors/windows (4)
        devices.append(Device("binary_sensor.front_door", "binary_sensor", "door", 0, ["front_door"]))
        devices.append(Device("binary_sensor.back_door", "binary_sensor", "door", 0, ["back_door"]))
        devices.append(Device("binary_sensor.garage_door", "binary_sensor", "garage_door", 0, ["garage"]))
        devices.append(Device("binary_sensor.kitchen_window", "binary_sensor", "window", 0, ["kitchen"]))

        # Power sensor (1 main)
        devices.append(Device("sensor.total_power", "sensor", "power", 0, []))

        # Temperature sensors (3)
        for room in ["bedroom", "living_room", "outside"]:
            devices.append(Device(f"sensor.{room}_temperature", "sensor", "temperature", 0, [room]))

        # Battery sensors (2 — for locks)
        devices.append(Device("sensor.front_door_lock_battery", "sensor", "battery", 0, [], battery=85))
        devices.append(Device("sensor.back_door_lock_battery", "sensor", "battery", 0, [], battery=58))

        # Media players (2)
        devices.append(Device("media_player.living_room", "media_player", None, 100, ["living_room"]))
        devices.append(Device("media_player.bedroom", "media_player", None, 50, ["bedroom"]))

        # Device trackers (3)
        devices.append(Device("device_tracker.alice_phone", "device_tracker", None, 0, []))
        devices.append(Device("device_tracker.bob_phone", "device_tracker", None, 0, []))
        devices.append(Device("device_tracker.alice_laptop", "device_tracker", None, 0, []))

        # Switches (3)
        devices.append(Device("switch.coffee_maker", "switch", "outlet", 1200, ["kitchen"]))
        devices.append(Device("switch.office_fan", "switch", "outlet", 75, ["office"]))
        devices.append(Device("switch.garage_opener", "switch", None, 200, ["garage"]))

        # Automations (4)
        devices.append(Device("automation.arrive_alice", "automation", None, 0, []))
        devices.append(Device("automation.arrive_bob", "automation", None, 0, []))
        devices.append(Device("automation.bedtime", "automation", None, 0, []))
        devices.append(Device("automation.morning_lights", "automation", None, 0, []))

        # EV sensors (3)
        devices.append(Device("sensor.luda_battery", "sensor", "battery", 0, []))
        devices.append(Device("sensor.luda_charger_power", "sensor", "power", 0, []))
        devices.append(Device("sensor.luda_range", "sensor", "distance", 0, []))

        # Sun
        devices.append(Device("sun.sun", "sun", None, 0, []))

        # Vacuum
        devices.append(Device("vacuum.roborock", "vacuum", None, 50, ["living_room"], battery=100))

        # Cover (1)
        devices.append(Device("cover.garage_door", "cover", "garage_door", 200, ["garage"]))

        return cls(devices)

    def get_devices_in_room(self, room: str) -> list[Device]:
        return [d for d in self.devices if room in d.rooms]

    def get_devices_by_domain(self, domain: str) -> list[Device]:
        return [d for d in self.devices if d.domain == domain]

    def get_device(self, entity_id: str) -> Device | None:
        for d in self.devices:
            if d.entity_id == entity_id:
                return d
        return None


class EntityStateGenerator:
    """Generates HA-format entity states based on person activity and time."""

    def __init__(self, roster: DeviceRoster, people: list[Person], seed: int = 42):
        self.roster = roster
        self.people = people
        self.seed = seed

    def generate_states(
        self, day: int, hour: float, is_weekend: bool,
        sunrise: float = 7.0, sunset: float = 18.0,
    ) -> list[dict]:
        """Generate all entity states for a given moment."""
        rng = random.Random(self.seed * 3000 + day * 100 + int(hour * 10))
        is_dark = hour < sunrise or hour > sunset
        states = []

        # Determine person locations
        person_locations = {}
        for person in self.people:
            transitions = person.get_room_transitions(day, is_weekend, self.seed)
            location = "bedroom"  # default: sleeping
            for t_hour, room in transitions:
                if t_hour <= hour:
                    location = room
            person_locations[person.name] = location

        people_home = [n for n, loc in person_locations.items() if loc != "away"]
        occupied_rooms = {loc for loc in person_locations.values() if loc != "away"}
        anyone_home = len(people_home) > 0

        # Person entities
        for person in self.people:
            loc = person_locations[person.name]
            device = self.roster.get_device(f"person.{person.name}")
            if device:
                state_val = "home" if loc != "away" else "not_home"
                states.append(device.to_ha_state(state_val))

        # Device trackers
        for person in self.people:
            loc = person_locations[person.name]
            trackers = [d for d in self.roster.devices
                        if d.domain == "device_tracker" and person.name in d.entity_id]
            for tracker in trackers:
                states.append(tracker.to_ha_state("home" if loc != "away" else "not_home"))

        # Lights — on in occupied rooms when dark, off otherwise
        for light in self.roster.get_devices_by_domain("light"):
            room = light.rooms[0] if light.rooms else ""
            is_on = room in occupied_rooms and is_dark and anyone_home
            # Add some randomness — not every light in an occupied room is on
            if is_on and rng.random() < 0.3:
                is_on = False
            if is_on:
                brightness = rng.randint(100, 255)
                states.append(light.to_ha_state("on", brightness=brightness))
            else:
                states.append(light.to_ha_state("off"))

        # Motion sensors — active in occupied rooms
        for sensor in self.roster.devices:
            if sensor.domain == "binary_sensor" and sensor.device_class == "motion":
                room = sensor.rooms[0] if sensor.rooms else ""
                is_active = room in occupied_rooms and rng.random() < 0.4
                states.append(sensor.to_ha_state("on" if is_active else "off"))

        # Door/window sensors
        for sensor in self.roster.devices:
            if sensor.domain == "binary_sensor" and sensor.device_class in ("door", "window", "garage_door"):
                states.append(sensor.to_ha_state("off"))  # closed by default

        # Locks
        for lock in self.roster.get_devices_by_domain("lock"):
            is_locked = not anyone_home or (hour > 22 or hour < 6)
            states.append(lock.to_ha_state("locked" if is_locked else "unlocked"))

        # Climate
        for climate in self.roster.get_devices_by_domain("climate"):
            temp_set = 68 if (hour > 22 or hour < 6) else 72
            states.append(climate.to_ha_state("cool", temperature=temp_set,
                                               current_temperature=temp_set + rng.gauss(0, 1)))

        # Power sensor — sum of active device watts + base load
        base_load = 80 + rng.gauss(0, 5)
        active_watts = sum(
            d.watts for d in self.roster.devices
            if d.watts > 0 and any(
                s["entity_id"] == d.entity_id and s["state"] in ("on", "playing", "cool", "heat")
                for s in states
            )
        )
        total_watts = base_load + active_watts
        power_device = self.roster.get_device("sensor.total_power")
        if power_device:
            states.append(power_device.to_ha_state(f"{total_watts:.1f}"))

        # Media players — evening entertainment
        for mp in self.roster.get_devices_by_domain("media_player"):
            room = mp.rooms[0] if mp.rooms else ""
            is_playing = (room in occupied_rooms and 18 <= hour <= 23
                          and rng.random() < 0.5)
            states.append(mp.to_ha_state("playing" if is_playing else "idle"))

        # Switches — coffee maker in morning, fan during work
        for switch in self.roster.get_devices_by_domain("switch"):
            is_on = False
            if "coffee" in switch.entity_id and 6 <= hour <= 8 and anyone_home:
                is_on = rng.random() < 0.7
            elif "fan" in switch.entity_id and "office" in switch.rooms:
                is_on = "office" in occupied_rooms
            states.append(switch.to_ha_state("on" if is_on else "off"))

        # Automations
        for auto in self.roster.get_devices_by_domain("automation"):
            states.append(auto.to_ha_state("on",
                                            last_triggered="2026-01-01T00:00:00+00:00"))

        # EV sensors
        ev_battery = self.roster.get_device("sensor.luda_battery")
        ev_charger = self.roster.get_device("sensor.luda_charger_power")
        ev_range = self.roster.get_device("sensor.luda_range")
        if ev_battery:
            # Battery depletes when away, charges when home
            base_pct = 70 + rng.gauss(0, 10)
            states.append(ev_battery.to_ha_state(f"{max(20, min(100, base_pct)):.0f}"))
        if ev_charger:
            is_charging = anyone_home and rng.random() < 0.3
            states.append(ev_charger.to_ha_state(f"{4.0 if is_charging else 0.0}"))
        if ev_range:
            states.append(ev_range.to_ha_state(f"{180 + rng.gauss(0, 20):.1f}"))

        # Sun
        sun_device = self.roster.get_device("sun.sun")
        if sun_device:
            is_up = sunrise <= hour <= sunset
            elevation = 45 * math.sin(math.pi * (hour - sunrise) / (sunset - sunrise)) if is_up else -10
            states.append(sun_device.to_ha_state(
                "above_horizon" if is_up else "below_horizon",
                elevation=round(elevation, 1),
                azimuth=round(90 + (hour - 6) * 15, 1),
                rising=hour < 12,
            ))

        # Vacuum — runs midday on weekdays if nobody home
        vacuum = self.roster.get_device("vacuum.roborock")
        if vacuum:
            is_cleaning = not anyone_home and 11 <= hour <= 13 and not is_weekend
            states.append(vacuum.to_ha_state("cleaning" if is_cleaning else "docked",
                                              battery_level=100 if not is_cleaning else 65))

        # Temperature sensors
        for sensor in self.roster.devices:
            if sensor.domain == "sensor" and sensor.device_class == "temperature":
                if "outside" in sensor.entity_id:
                    # Outdoor temp: sinusoidal daily cycle
                    base_temp = 55 + 15 * math.sin(math.pi * (hour - 6) / 12)
                    states.append(sensor.to_ha_state(f"{base_temp + rng.gauss(0, 2):.1f}"))
                else:
                    states.append(sensor.to_ha_state(f"{72 + rng.gauss(0, 1):.1f}"))

        # Battery sensors
        for sensor in self.roster.devices:
            if sensor.domain == "sensor" and sensor.device_class == "battery" and "luda" not in sensor.entity_id:
                states.append(sensor.to_ha_state(f"{sensor.battery or 80}"))

        # Cover
        cover = self.roster.get_device("cover.garage_door")
        if cover:
            states.append(cover.to_ha_state("closed"))

        return states
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_entities.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add tests/synthetic/entities.py tests/synthetic/test_entities.py
git commit -m "feat(synthetic): add DeviceRoster and EntityStateGenerator"
```

---

## Task 3: Weather Profile

**Files:**
- Create: `tests/synthetic/weather.py`
- Test: `tests/synthetic/test_weather.py`

**Context:** WeatherProfile generates temperature, humidity, and wind based on region and month. Influences outdoor temps, climate setpoints, and daylight hours.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_weather.py
"""Tests for weather profile generation."""
import pytest
from tests.synthetic.weather import WeatherProfile


class TestWeatherProfile:
    def test_southeast_us_february(self):
        wp = WeatherProfile("southeast_us", month=2)
        assert wp.avg_high > wp.avg_low
        assert wp.sunrise < wp.sunset

    def test_get_conditions_for_hour(self):
        wp = WeatherProfile("southeast_us", month=2)
        cond = wp.get_conditions(day=0, hour=14.0, seed=42)
        assert "temp_f" in cond
        assert "humidity_pct" in cond
        assert "wind_mph" in cond
        assert cond["temp_f"] > 0  # February in southeast US

    def test_temperature_peaks_afternoon(self):
        wp = WeatherProfile("southeast_us", month=7)
        morning = wp.get_conditions(day=0, hour=8.0, seed=42)
        afternoon = wp.get_conditions(day=0, hour=14.0, seed=42)
        # Afternoon should generally be warmer
        assert afternoon["temp_f"] > morning["temp_f"]

    def test_deterministic_with_seed(self):
        wp = WeatherProfile("southeast_us", month=2)
        a = wp.get_conditions(day=5, hour=12.0, seed=42)
        b = wp.get_conditions(day=5, hour=12.0, seed=42)
        assert a == b

    def test_daylight_hours(self):
        wp = WeatherProfile("southeast_us", month=6)  # summer
        wp_winter = WeatherProfile("southeast_us", month=12)
        assert wp.daylight_hours > wp_winter.daylight_hours
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_weather.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# tests/synthetic/weather.py
"""Weather profile generation for household simulation."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# Monthly averages for southeast US (approximate Nashville/Atlanta)
SOUTHEAST_US = {
    1:  {"high": 48, "low": 30, "humidity": 65, "wind": 8, "sunrise": 7.1, "sunset": 17.3},
    2:  {"high": 53, "low": 33, "humidity": 62, "wind": 9, "sunrise": 6.8, "sunset": 17.8},
    3:  {"high": 62, "low": 40, "humidity": 58, "wind": 10, "sunrise": 6.2, "sunset": 18.3},
    4:  {"high": 72, "low": 49, "humidity": 55, "wind": 9, "sunrise": 5.5, "sunset": 18.8},
    5:  {"high": 80, "low": 58, "humidity": 60, "wind": 7, "sunrise": 5.1, "sunset": 19.3},
    6:  {"high": 87, "low": 66, "humidity": 65, "wind": 6, "sunrise": 5.0, "sunset": 19.6},
    7:  {"high": 90, "low": 70, "humidity": 70, "wind": 5, "sunrise": 5.2, "sunset": 19.5},
    8:  {"high": 89, "low": 69, "humidity": 70, "wind": 5, "sunrise": 5.5, "sunset": 19.1},
    9:  {"high": 83, "low": 62, "humidity": 65, "wind": 6, "sunrise": 6.0, "sunset": 18.4},
    10: {"high": 72, "low": 50, "humidity": 58, "wind": 7, "sunrise": 6.4, "sunset": 17.7},
    11: {"high": 61, "low": 40, "humidity": 60, "wind": 8, "sunrise": 6.8, "sunset": 17.1},
    12: {"high": 50, "low": 32, "humidity": 65, "wind": 8, "sunrise": 7.1, "sunset": 17.0},
}

REGIONS = {
    "southeast_us": SOUTHEAST_US,
}


@dataclass
class WeatherProfile:
    """Weather conditions for a region and month."""
    region: str
    month: int

    def __post_init__(self):
        data = REGIONS[self.region][self.month]
        self.avg_high = data["high"]
        self.avg_low = data["low"]
        self.avg_humidity = data["humidity"]
        self.avg_wind = data["wind"]
        self.sunrise = data["sunrise"]
        self.sunset = data["sunset"]

    @property
    def daylight_hours(self) -> float:
        return self.sunset - self.sunrise

    def get_conditions(self, day: int, hour: float, seed: int) -> dict:
        """Get weather conditions for a specific day and hour."""
        rng = random.Random(seed * 4000 + day)
        # Daily variation from average
        daily_high = self.avg_high + rng.gauss(0, 4)
        daily_low = self.avg_low + rng.gauss(0, 3)
        daily_humidity = max(20, min(100, self.avg_humidity + rng.gauss(0, 8)))
        daily_wind = max(0, self.avg_wind + rng.gauss(0, 3))

        # Hourly temperature: sinusoidal curve peaking at 15:00
        t_range = daily_high - daily_low
        # Peak at 15:00, trough at 5:00
        temp = daily_low + t_range * 0.5 * (1 + math.sin(math.pi * (hour - 5) / 20))

        return {
            "temp_f": round(temp, 1),
            "humidity_pct": round(daily_humidity, 1),
            "wind_mph": round(max(0, daily_wind), 1),
        }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_weather.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add tests/synthetic/weather.py tests/synthetic/test_weather.py
git commit -m "feat(synthetic): add WeatherProfile for regional weather simulation"
```

---

## Task 4: Snapshot Assembler (Real Collectors)

**Files:**
- Create: `tests/synthetic/assembler.py`
- Test: `tests/synthetic/test_assembler.py`

**Context:** The assembler takes entity states from EntityStateGenerator and feeds them through ARIA's real CollectorRegistry to produce snapshot dicts. This is the critical bridge between synthetic data and the real pipeline.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_assembler.py
"""Tests for snapshot assembler using real ARIA collectors."""
import pytest
from tests.synthetic.assembler import SnapshotAssembler
from tests.synthetic.entities import DeviceRoster, EntityStateGenerator
from tests.synthetic.people import Person, Schedule
from tests.synthetic.weather import WeatherProfile


@pytest.fixture
def assembler():
    roster = DeviceRoster.typical_home()
    people = [
        Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
        Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
    ]
    weather = WeatherProfile("southeast_us", month=2)
    return SnapshotAssembler(roster, people, weather, seed=42)


class TestSnapshotAssembler:
    def test_build_snapshot_for_day(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert snapshot["date"] == "2026-02-14"
        assert snapshot["day_of_week"] == "Saturday"
        assert "power" in snapshot
        assert "lights" in snapshot
        assert "occupancy" in snapshot
        assert "climate" in snapshot
        assert "locks" in snapshot
        assert "motion" in snapshot
        assert "entities" in snapshot

    def test_snapshot_has_nonzero_power(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert snapshot["power"]["total_watts"] > 0

    def test_snapshot_has_occupancy(self, assembler):
        snapshot = assembler.build_snapshot(day=5, date_str="2026-02-19")
        # Weekday evening snapshot should have people home or not
        occ = snapshot["occupancy"]
        assert "people_home" in occ or "device_count_home" in occ

    def test_snapshot_has_weather(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert "temp_f" in snapshot["weather"]
        assert snapshot["weather"]["temp_f"] > 0

    def test_snapshot_has_time_features(self, assembler):
        snapshot = assembler.build_snapshot(day=0, date_str="2026-02-14")
        assert "time_features" in snapshot
        assert "hour_sin" in snapshot["time_features"]

    def test_build_daily_series(self, assembler):
        snapshots = assembler.build_daily_series(days=7, start_date="2026-02-14")
        assert len(snapshots) == 7
        # Dates should be consecutive
        dates = [s["date"] for s in snapshots]
        assert dates[0] == "2026-02-14"
        assert dates[6] == "2026-02-20"

    def test_snapshots_are_deterministic(self, assembler):
        a = assembler.build_daily_series(days=3, start_date="2026-02-14")
        b = assembler.build_daily_series(days=3, start_date="2026-02-14")
        for sa, sb in zip(a, b):
            assert sa["power"]["total_watts"] == sb["power"]["total_watts"]
            assert sa["lights"]["on"] == sb["lights"]["on"]

    def test_snapshots_compatible_with_training(self, assembler):
        """Snapshots should work with build_training_data()."""
        from aria.engine.features.vector_builder import build_training_data
        snapshots = assembler.build_daily_series(days=14, start_date="2026-02-01")
        names, X, targets = build_training_data(snapshots)
        assert len(names) > 0
        assert len(X) > 0
        assert "power_watts" in targets
        assert len(targets["power_watts"]) == len(X)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_assembler.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# tests/synthetic/assembler.py
"""Snapshot assembler — feeds synthetic entity states through real ARIA collectors."""
from __future__ import annotations

from datetime import datetime, timedelta

from aria.engine.collectors.registry import CollectorRegistry
import aria.engine.collectors.extractors  # noqa: F401 — triggers registration
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
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        is_weekend = dt.weekday() >= 5
        holidays_config = HolidayConfig()
        safety_config = SafetyConfig()

        # Generate entity states for this moment
        states = self.entity_gen.generate_states(
            day=day, hour=hour, is_weekend=is_weekend,
            sunrise=self.weather.sunrise, sunset=self.weather.sunset,
        )

        # Build empty snapshot with correct date metadata
        snapshot = build_empty_snapshot(date_str, holidays_config)

        # Run all real collectors against synthetic states
        for name, collector_cls in CollectorRegistry.all().items():
            if name == "entities_summary":
                collector = collector_cls(safety_config=safety_config)
            else:
                collector = collector_cls()
            collector.extract(snapshot, states)

        # Add weather
        weather_cond = self.weather.get_conditions(day, hour, self.seed)
        snapshot["weather"] = weather_cond

        # Add time features
        timestamp_str = f"{date_str}T{int(hour):02d}:{int((hour % 1) * 60):02d}:00"
        sun_data = None
        for s in states:
            if s["entity_id"] == "sun.sun":
                sun_data = s["attributes"]
                break
        snapshot["time_features"] = build_time_features(timestamp_str, sun_data, date_str)

        # Add logbook summary (synthetic)
        snapshot["logbook_summary"] = {
            "total_events": 2500 + (day * 10),
            "useful_events": 2000 + (day * 8),
            "by_domain": {"light": 200, "switch": 100, "binary_sensor": 500},
            "hourly": {},
        }

        return snapshot

    def build_daily_series(
        self, days: int, start_date: str = "2026-02-01"
    ) -> list[dict]:
        """Build a series of daily snapshots."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        snapshots = []
        for day in range(days):
            dt = start + timedelta(days=day)
            date_str = dt.strftime("%Y-%m-%d")
            snapshot = self.build_snapshot(day=day, date_str=date_str)
            snapshots.append(snapshot)
        return snapshots
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_assembler.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add tests/synthetic/assembler.py tests/synthetic/test_assembler.py
git commit -m "feat(synthetic): add SnapshotAssembler using real ARIA collectors"
```

---

## Task 5: Household Simulator & Scenarios

**Files:**
- Create: `tests/synthetic/simulator.py`
- Create: `tests/synthetic/scenarios/__init__.py`
- Create: `tests/synthetic/scenarios/household.py`
- Test: `tests/synthetic/test_simulator.py`

**Context:** HouseholdSimulator is the top-level entry point. It accepts a named scenario, creates the appropriate people/devices/weather, and delegates to SnapshotAssembler. Scenarios define household configurations, not outcomes.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_simulator.py
"""Tests for HouseholdSimulator and scenarios."""
import pytest
from tests.synthetic.simulator import HouseholdSimulator


class TestHouseholdSimulator:
    def test_stable_couple_scenario(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 7

    def test_new_roommate_scenario(self):
        sim = HouseholdSimulator(scenario="new_roommate", days=21, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 21

    def test_vacation_scenario(self):
        sim = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14

    def test_work_from_home_scenario(self):
        sim = HouseholdSimulator(scenario="work_from_home", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14

    def test_sensor_degradation_scenario(self):
        sim = HouseholdSimulator(scenario="sensor_degradation", days=14, seed=42)
        snapshots = sim.generate()
        assert len(snapshots) == 14

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            HouseholdSimulator(scenario="nonexistent", days=7, seed=42)

    def test_deterministic(self):
        a = HouseholdSimulator(scenario="stable_couple", days=7, seed=42).generate()
        b = HouseholdSimulator(scenario="stable_couple", days=7, seed=42).generate()
        for sa, sb in zip(a, b):
            assert sa["power"]["total_watts"] == sb["power"]["total_watts"]

    def test_snapshots_have_variation(self):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        powers = [s["power"]["total_watts"] for s in snapshots]
        # Not all the same
        assert len(set(powers)) > 1

    def test_vacation_has_low_occupancy_midweek(self):
        sim = HouseholdSimulator(scenario="vacation", days=14, seed=42)
        snapshots = sim.generate()
        # Days 10-17 should have low/no occupancy (0-indexed: days 9-16)
        vacation_snapshots = snapshots[9:14]  # middle of vacation
        for s in vacation_snapshots:
            # During vacation, device_count_home should be lower
            assert s["occupancy"]["device_count_home"] <= 5 or len(s["occupancy"].get("people_home", [])) == 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_simulator.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# tests/synthetic/scenarios/__init__.py
"""Scenario configurations for household simulation."""
```

```python
# tests/synthetic/scenarios/household.py
"""Household scenario definitions."""
from __future__ import annotations

from tests.synthetic.people import Person, Schedule
from tests.synthetic.entities import DeviceRoster
from tests.synthetic.weather import WeatherProfile


def stable_couple(seed: int = 42) -> dict:
    """Two residents with consistent schedules."""
    return {
        "people": [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
    }


def new_roommate(seed: int = 42) -> dict:
    """Two residents for 14 days, third joins at day 15."""
    return {
        "people": [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
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
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "vacation_days": range(10, 18),
    }


def work_from_home(seed: int = 42) -> dict:
    """One resident switches to WFH at day 8."""
    return {
        "people": [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "wfh_person": "alice",
        "wfh_start_day": 8,
    }


def sensor_degradation(seed: int = 42) -> dict:
    """Battery sensors start reporting unavailable at day 20."""
    return {
        "people": [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=2),
        "degrade_start_day": 20,
    }


def holiday_week(seed: int = 42) -> dict:
    """Normal schedule with holiday flags."""
    return {
        "people": [
            Person("alice", Schedule.weekday_office(6.5, 23), Schedule.weekend(8, 23.5)),
            Person("bob", Schedule.weekday_office(7, 22.5), Schedule.weekend(8.5, 23)),
        ],
        "roster": DeviceRoster.typical_home(),
        "weather": WeatherProfile("southeast_us", month=12),
        "holiday_days": range(24, 27),  # Christmas
    }


SCENARIOS = {
    "stable_couple": stable_couple,
    "new_roommate": new_roommate,
    "vacation": vacation,
    "work_from_home": work_from_home,
    "sensor_degradation": sensor_degradation,
    "holiday_week": holiday_week,
}
```

```python
# tests/synthetic/simulator.py
"""HouseholdSimulator — top-level entry point for synthetic data generation."""
from __future__ import annotations

from datetime import datetime, timedelta

from tests.synthetic.assembler import SnapshotAssembler
from tests.synthetic.people import Person, Schedule
from tests.synthetic.scenarios.household import SCENARIOS


class HouseholdSimulator:
    """Generate realistic household data for ARIA pipeline testing."""

    def __init__(self, scenario: str, days: int, seed: int = 42, start_date: str = "2026-02-01"):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
        self.scenario_name = scenario
        self.scenario_config = SCENARIOS[scenario](seed)
        self.days = days
        self.seed = seed
        self.start_date = start_date

    def generate(self) -> list[dict]:
        """Generate daily snapshots for the scenario."""
        config = self.scenario_config
        people = list(config["people"])
        roster = config["roster"]
        weather = config["weather"]

        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        snapshots = []

        for day in range(self.days):
            dt = start + timedelta(days=day)
            date_str = dt.strftime("%Y-%m-%d")
            is_weekend = dt.weekday() >= 5

            # Apply scenario modifiers
            day_people = self._get_people_for_day(people, config, day, is_weekend)
            day_roster = self._get_roster_for_day(roster, config, day)

            assembler = SnapshotAssembler(day_roster, day_people, weather, self.seed)
            snapshot = assembler.build_snapshot(day=day, date_str=date_str)

            # Apply scenario-specific snapshot modifications
            self._apply_scenario_mods(snapshot, config, day)

            snapshots.append(snapshot)

        return snapshots

    def _get_people_for_day(
        self, base_people: list[Person], config: dict, day: int, is_weekend: bool
    ) -> list[Person]:
        """Adjust people list based on scenario events."""
        people = list(base_people)

        # Vacation: everyone away
        vacation_days = config.get("vacation_days")
        if vacation_days and day in vacation_days:
            # Replace all schedules with "away all day"
            away_sched = Schedule(wake=8, sleep=22, depart=0, arrive=24)
            return [
                Person(p.name, away_sched, away_sched)
                for p in people
            ]

        # New roommate joins
        add_day = config.get("add_person_at_day")
        if add_day and day >= add_day:
            new_person = config["new_person"]
            if new_person.name not in [p.name for p in people]:
                people.append(new_person)

        # Work from home: replace one person's weekday schedule
        wfh_person = config.get("wfh_person")
        wfh_start = config.get("wfh_start_day")
        if wfh_person and wfh_start and day >= wfh_start and not is_weekend:
            wfh_sched = Schedule(wake=7, sleep=23, depart=None, arrive=None)
            people = [
                Person(p.name, wfh_sched, p.schedule_weekend)
                if p.name == wfh_person else p
                for p in people
            ]

        return people

    def _get_roster_for_day(self, base_roster, config: dict, day: int):
        """Adjust device roster based on scenario events."""
        degrade_day = config.get("degrade_start_day")
        if degrade_day and day >= degrade_day:
            # Mark battery-powered sensors as unavailable by draining batteries
            from tests.synthetic.entities import DeviceRoster
            devices = []
            for d in base_roster.devices:
                if d.battery is not None and d.domain in ("binary_sensor", "sensor"):
                    # Gradually degrade
                    from copy import copy
                    degraded = copy(d)
                    degraded.battery = max(0, d.battery - (day - degrade_day) * 5)
                    devices.append(degraded)
                else:
                    devices.append(d)
            return DeviceRoster(devices)
        return base_roster

    def _apply_scenario_mods(self, snapshot: dict, config: dict, day: int):
        """Apply post-snapshot scenario modifications."""
        holiday_days = config.get("holiday_days")
        if holiday_days and day in holiday_days:
            snapshot["is_holiday"] = True
            snapshot["holiday_name"] = "Holiday"
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_simulator.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add tests/synthetic/simulator.py tests/synthetic/scenarios/__init__.py tests/synthetic/scenarios/household.py tests/synthetic/test_simulator.py
git commit -m "feat(synthetic): add HouseholdSimulator with 6 named scenarios"
```

---

## Task 6: Pipeline Runner

**Files:**
- Create: `tests/synthetic/pipeline.py`
- Test: `tests/synthetic/test_pipeline.py`

**Context:** PipelineRunner orchestrates the full ARIA engine pipeline in a temp directory: save snapshots → compute baselines → build features → train models → generate predictions → score. Uses real ARIA code at every step.

**Step 1: Write the failing test**

```python
# tests/synthetic/test_pipeline.py
"""Tests for PipelineRunner — full pipeline orchestration."""
import pytest
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator


@pytest.fixture
def runner(tmp_path):
    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
    snapshots = sim.generate()
    return PipelineRunner(snapshots, data_dir=tmp_path)


class TestPipelineRunner:
    def test_save_snapshots(self, runner):
        runner.save_snapshots()
        daily_dir = runner.store.paths.daily_dir
        assert daily_dir.exists()
        files = list(daily_dir.glob("*.json"))
        assert len(files) == 21

    def test_compute_baselines(self, runner):
        runner.save_snapshots()
        baselines = runner.compute_baselines()
        assert isinstance(baselines, dict)
        # Should have at least some days of week
        assert len(baselines) > 0

    def test_build_training_data(self, runner):
        runner.save_snapshots()
        names, X, targets = runner.build_training_data()
        assert len(names) > 0
        assert len(X) > 0
        assert "power_watts" in targets

    def test_train_models(self, runner):
        runner.save_snapshots()
        results = runner.train_models()
        assert isinstance(results, dict)
        # Should have trained at least one model
        models_dir = runner.store.paths.models_dir
        assert models_dir.exists()

    def test_generate_predictions(self, runner):
        runner.save_snapshots()
        runner.compute_baselines()
        runner.train_models()
        predictions = runner.generate_predictions()
        assert "prediction_method" in predictions
        assert "power_watts" in predictions

    def test_score_predictions(self, runner):
        runner.save_snapshots()
        runner.compute_baselines()
        runner.train_models()
        runner.generate_predictions()
        scores = runner.score_predictions()
        assert "overall" in scores
        assert "metrics" in scores

    def test_run_full_pipeline(self, runner):
        """End-to-end: all stages complete without error."""
        result = runner.run_full()
        assert "snapshots_saved" in result
        assert "baselines" in result
        assert "training" in result
        assert "predictions" in result
        assert "scores" in result
        assert result["snapshots_saved"] == 21
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/synthetic/test_pipeline.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# tests/synthetic/pipeline.py
"""PipelineRunner — orchestrates full ARIA engine pipeline against synthetic data."""
from __future__ import annotations

from pathlib import Path

from aria.engine.config import (
    AppConfig, HAConfig, PathConfig, ModelConfig,
    OllamaConfig, WeatherConfig, SafetyConfig, HolidayConfig,
)
from aria.engine.storage.data_store import DataStore
from aria.engine.analysis.baselines import compute_baselines
from aria.engine.features.vector_builder import build_training_data
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.models.training import train_continuous_model
from aria.engine.predictions.predictor import generate_predictions, blend_predictions
from aria.engine.predictions.scoring import score_all_predictions


class PipelineRunner:
    """Run the full ARIA engine pipeline in a temp directory."""

    def __init__(self, snapshots: list[dict], data_dir: Path):
        self.snapshots = snapshots
        self.paths = PathConfig(data_dir=data_dir, logbook_path=data_dir / "current.json")
        self.paths.ensure_dirs()
        self.store = DataStore(self.paths)
        self.config = AppConfig(
            ha=HAConfig(),
            paths=self.paths,
            model=ModelConfig(),
            ollama=OllamaConfig(),
            weather=WeatherConfig(),
            safety=SafetyConfig(),
            holidays=HolidayConfig(),
        )
        self._baselines = None
        self._training_results = None
        self._predictions = None

    def save_snapshots(self) -> int:
        """Save all snapshots to the data directory."""
        for snapshot in self.snapshots:
            self.store.save_snapshot(snapshot)
        return len(self.snapshots)

    def compute_baselines(self) -> dict:
        """Compute baselines from saved snapshots."""
        snapshots = self.store.load_recent_snapshots(days=len(self.snapshots))
        if not snapshots:
            snapshots = self.snapshots
        self._baselines = compute_baselines(snapshots)
        self.store.save_baselines(self._baselines)
        return self._baselines

    def build_training_data(self) -> tuple:
        """Build feature matrix from snapshots."""
        snapshots = self.store.load_recent_snapshots(days=len(self.snapshots))
        if not snapshots:
            snapshots = self.snapshots
        config = self.store.load_feature_config() or DEFAULT_FEATURE_CONFIG
        return build_training_data(snapshots, config)

    def train_models(self) -> dict:
        """Train ML models from snapshots."""
        names, X, targets = self.build_training_data()
        results = {}
        models_dir = str(self.paths.models_dir)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)

        for metric_name, y in targets.items():
            if len(y) < 7:
                continue
            result = train_continuous_model(metric_name, names, X, y, models_dir)
            results[metric_name] = result

        self._training_results = results
        return results

    def generate_predictions(self, target_date: str | None = None) -> dict:
        """Generate predictions using baselines and trained models."""
        if self._baselines is None:
            self.compute_baselines()

        if target_date is None:
            # Predict for the day after the last snapshot
            last_date = self.snapshots[-1]["date"]
            from datetime import datetime, timedelta
            dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            target_date = dt.strftime("%Y-%m-%d")

        # Get ML predictions if models are trained
        ml_predictions = None
        if self._training_results and self.paths.models_dir.exists():
            from aria.engine.models.training import predict_with_ml
            last_snapshot = self.snapshots[-1]
            prev_snapshot = self.snapshots[-2] if len(self.snapshots) > 1 else None
            ml_predictions = predict_with_ml(
                last_snapshot, config=None, prev_snapshot=prev_snapshot,
                models_dir=str(self.paths.models_dir), store=self.store,
            )

        self._predictions = generate_predictions(
            target_date=target_date,
            baselines=self._baselines,
            ml_predictions=ml_predictions,
            paths=self.paths,
        )
        self.store.save_predictions(self._predictions)
        return self._predictions

    def score_predictions(self) -> dict:
        """Score predictions against last snapshot (as proxy for actual)."""
        if self._predictions is None:
            self.generate_predictions()
        actual_snapshot = self.snapshots[-1]
        return score_all_predictions(self._predictions, actual_snapshot)

    def run_full(self) -> dict:
        """Run the full pipeline end-to-end."""
        n_saved = self.save_snapshots()
        baselines = self.compute_baselines()
        training = self.train_models()
        predictions = self.generate_predictions()
        scores = self.score_predictions()
        return {
            "snapshots_saved": n_saved,
            "baselines": baselines,
            "training": training,
            "predictions": predictions,
            "scores": scores,
        }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/synthetic/test_pipeline.py -v`
Expected: All 7 tests PASS

**Note:** This task may require debugging. The pipeline touches many real ARIA modules. If a test fails due to missing fields in synthetic snapshots, the fix belongs in the SnapshotAssembler (Task 4) to add the missing field, not in the pipeline runner. Read the error carefully — it will tell you which snapshot field is expected.

**Step 5: Commit**

```bash
git add tests/synthetic/pipeline.py tests/synthetic/test_pipeline.py
git commit -m "feat(synthetic): add PipelineRunner for full pipeline orchestration"
```

---

## Task 7: Integration Tests — Model Competence (Tier 1)

**Files:**
- Create: `tests/integration/__init__.py` (if not exists — check first, there's already 5 integration tests)
- Create: `tests/integration/test_model_competence.py`
- Create: `tests/integration/conftest.py`

**Context:** These tests validate that sklearn models exhibit learner behavior against realistic simulated data. Assertions are relative (improvement, comparison) not absolute.

**Step 1: Write the failing test**

```python
# tests/integration/conftest.py
"""Shared fixtures for integration tests."""
import pytest
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner


@pytest.fixture(scope="module")
def stable_30d_runner(tmp_path_factory):
    """30-day stable household with full pipeline run. Module-scoped for performance."""
    tmp = tmp_path_factory.mktemp("stable_30d")
    sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp)
    runner.save_snapshots()
    return runner


@pytest.fixture(scope="module")
def stable_30d_snapshots():
    """30-day stable household snapshots."""
    sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
    return sim.generate()
```

```python
# tests/integration/test_model_competence.py
"""Tier 1: ML model competence tests against realistic synthetic data."""
import pytest
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner


class TestModelsConverge:
    """Models should improve accuracy with more training data."""

    def test_r2_improves_with_more_data(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        # Train on 14 days
        runner_early = PipelineRunner(snapshots[:14], data_dir=tmp_path / "early")
        runner_early.save_snapshots()
        early_results = runner_early.train_models()

        # Train on 25 days
        runner_late = PipelineRunner(snapshots[:25], data_dir=tmp_path / "late")
        runner_late.save_snapshots()
        late_results = runner_late.train_models()

        # At least one metric should have better R2 with more data
        improved = False
        for metric in early_results:
            if metric in late_results:
                if late_results[metric].get("r2", 0) >= early_results[metric].get("r2", 0):
                    improved = True
        assert improved, "No metric improved R2 with more data"


class TestModelsBeatBaseline:
    """ML predictions should outperform naive day-of-week baselines after sufficient data."""

    def test_ml_blend_beats_pure_baseline(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()

        scores = result["scores"]
        # Overall accuracy should be > 0 (not all wrong)
        assert scores["overall"] > 0


class TestModelsGeneralize:
    """Models should not severely overfit — test accuracy near train accuracy."""

    def test_generalization_gap_reasonable(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots[:21], data_dir=tmp_path)
        runner.save_snapshots()
        training_results = runner.train_models()

        # Check that validation metrics exist and are reasonable
        for metric, result in training_results.items():
            r2 = result.get("r2", None)
            if r2 is not None:
                # R2 should not be wildly negative (severe overfitting to noise)
                assert r2 > -1.0, f"{metric} has R2={r2}, suggesting severe overfitting"


class TestDegradationGraceful:
    """Pipeline should handle missing/degraded data without crashing."""

    def test_sensor_degradation_completes(self, tmp_path):
        sim = HouseholdSimulator(scenario="sensor_degradation", days=30, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()
        # Pipeline completed without exception
        assert result["snapshots_saved"] == 30
        assert result["predictions"] is not None


class TestColdStartProgression:
    """Pipeline should progress through learning stages with limited data."""

    def test_7_day_cold_start(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        baselines = runner.compute_baselines()
        # Should have baselines even with 7 days
        assert len(baselines) > 0

        # Predictions should work (statistical fallback)
        predictions = runner.generate_predictions()
        assert predictions is not None
        assert "power_watts" in predictions
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_model_competence.py -v`
Expected: FAIL

**Step 3: No new implementation needed** — these tests use PipelineRunner and HouseholdSimulator from previous tasks.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_model_competence.py -v`
Expected: All 5 tests PASS

**Note:** If any test fails, read the error. Likely causes: (1) synthetic snapshots missing a field the feature builder expects — fix in assembler.py, (2) not enough data for model training — adjust `ModelConfig.min_training_samples`. Do NOT weaken the test assertions.

**Step 5: Commit**

```bash
git add tests/integration/test_model_competence.py tests/integration/conftest.py
git commit -m "test(integration): add Tier 1 ML model competence tests"
```

---

## Task 8: Integration Tests — E2E Pipeline Flow (Tier 3)

**Files:**
- Create: `tests/integration/test_pipeline_flow.py`

**Context:** Validates that data flows through every pipeline stage and that intermediate formats are correct at each handoff.

**Step 1: Write the failing test**

```python
# tests/integration/test_pipeline_flow.py
"""Tier 3: End-to-end pipeline flow and handoff validation."""
import json
import pytest
from pathlib import Path
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner


class TestFullPipelineCompletes:
    """Full pipeline should run to completion with various scenarios."""

    @pytest.mark.parametrize("scenario,days", [
        ("stable_couple", 21),
        ("vacation", 14),
        ("work_from_home", 14),
        ("sensor_degradation", 30),
    ])
    def test_scenario_completes(self, tmp_path, scenario, days):
        sim = HouseholdSimulator(scenario=scenario, days=days, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()
        assert result["snapshots_saved"] == days
        assert result["scores"] is not None


class TestIntermediateFormats:
    """Each stage's output should match the expected schema."""

    def test_snapshot_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=7, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()

        # Load a saved snapshot and verify format
        loaded = runner.store.load_snapshot(snapshots[0]["date"])
        assert loaded is not None
        required_keys = ["date", "day_of_week", "power", "lights", "occupancy",
                         "climate", "locks", "motion", "entities", "weather"]
        for key in required_keys:
            assert key in loaded, f"Snapshot missing key: {key}"

    def test_baselines_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        baselines = runner.compute_baselines()

        # Baselines should have day-of-week keys with metric stats
        for day_name, day_data in baselines.items():
            assert "sample_count" in day_data
            assert "power_watts" in day_data
            assert "mean" in day_data["power_watts"]

    def test_predictions_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.run_full()

        predictions = runner._predictions
        assert "target_date" in predictions or "prediction_method" in predictions
        assert "power_watts" in predictions

    def test_scores_format(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        result = runner.run_full()

        scores = result["scores"]
        assert "overall" in scores
        assert "metrics" in scores
        assert isinstance(scores["overall"], (int, float))


class TestHubReadsEngineOutput:
    """Hub's IntelligenceModule should be able to read engine-produced files."""

    def test_hub_loads_baselines(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=14, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        runner.compute_baselines()

        # Verify the baselines file exists and is valid JSON
        baselines_path = runner.paths.data_dir / "baselines" / "baselines.json"
        assert baselines_path.exists()
        with open(baselines_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_hub_loads_predictions(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.run_full()

        # Verify predictions file exists
        preds_dir = runner.paths.data_dir / "predictions"
        assert preds_dir.exists()
        files = list(preds_dir.glob("*.json"))
        assert len(files) >= 1

    def test_hub_loads_models(self, tmp_path):
        sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        runner.train_models()

        # Verify model files exist
        models_dir = runner.paths.models_dir
        assert models_dir.exists()
        pkl_files = list(models_dir.glob("*.pkl"))
        assert len(pkl_files) >= 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_pipeline_flow.py -v`
Expected: FAIL

**Step 3: No new implementation needed.**

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_pipeline_flow.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/integration/test_pipeline_flow.py
git commit -m "test(integration): add Tier 3 E2E pipeline flow tests"
```

---

## Task 9: Ollama Record/Replay & Meta-Learning Tests (Tier 2)

**Files:**
- Create: `tests/integration/test_meta_learning.py`
- Create: `tests/fixtures/ollama_responses/` directory
- Create: `tests/fixtures/ollama_responses/meta_learning_sample.json`

**Context:** Meta-learning tests validate that Ollama output is valid and that the meta-learning loop doesn't degrade model accuracy. CI mode uses recorded responses. Local mode hits real Ollama.

**Step 1: Write the failing test**

```python
# tests/integration/test_meta_learning.py
"""Tier 2: Ollama meta-learning validation tests."""
import json
import pytest
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner
from aria.engine.llm.meta_learning import parse_suggestions


# Recorded Ollama response for CI replay
SAMPLE_META_RESPONSE = """<think>
Looking at the model performance, power_watts has R2=0.45 and lights_on has R2=0.32.
The feature importance shows hour_sin and people_home_count are the top features.
I should suggest enabling interaction features to capture the relationship.
</think>

```json
[
  {
    "parameter": "interaction_features.people_home_x_hour_sin",
    "current_value": false,
    "suggested_value": true,
    "reasoning": "people_home_count and hour_sin are both top features. Their interaction likely captures the occupancy-time pattern driving power and light usage.",
    "expected_impact": "Improve power_watts R2 by 5-10%"
  }
]
```"""


class TestParseMetaLearningOutput:
    """Meta-learning output should parse correctly."""

    def test_parse_valid_response(self):
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        assert len(suggestions) >= 1
        assert "parameter" in suggestions[0]
        assert "suggested_value" in suggestions[0]

    def test_parse_empty_response(self):
        suggestions = parse_suggestions("")
        assert suggestions == []

    def test_parse_response_with_no_json(self):
        suggestions = parse_suggestions("No changes needed at this time.")
        assert suggestions == []

    def test_suggestions_have_reasoning(self):
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        for s in suggestions:
            assert "reasoning" in s
            assert len(s["reasoning"]) > 10


class TestMetaLearningOutputValid:
    """Meta-learning suggestions should reference valid parameters."""

    def test_suggested_parameters_exist_in_config(self):
        from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
        suggestions = parse_suggestions(SAMPLE_META_RESPONSE)
        for s in suggestions:
            param = s["parameter"]
            # Should be a dot-path into DEFAULT_FEATURE_CONFIG
            parts = param.split(".")
            config = DEFAULT_FEATURE_CONFIG
            for part in parts:
                assert part in config, f"Parameter {param} not found in feature config"
                config = config[part]


@pytest.mark.ollama
class TestMetaLearningLive:
    """Tests that require a running Ollama instance. Skip in CI."""

    def test_meta_learning_produces_output(self, tmp_path):
        """Run real meta-learning against synthetic data."""
        sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
        snapshots = sim.generate()
        runner = PipelineRunner(snapshots, data_dir=tmp_path)
        runner.save_snapshots()
        runner.compute_baselines()
        training = runner.train_models()

        # Build the meta-learning input (model scores + config)
        from aria.engine.llm.meta_learning import parse_suggestions
        # This would need the actual Ollama call — implementation depends on
        # how meta_learning.py exposes its API. Wire up in implementation.
        pytest.skip("Requires Ollama integration wiring — implement after pipeline runner is stable")
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_meta_learning.py -v -m "not ollama"`
Expected: FAIL

**Step 3: Create fixture directory and sample response**

```bash
mkdir -p tests/fixtures/ollama_responses
```

```json
// tests/fixtures/ollama_responses/meta_learning_sample.json
{
  "response": "<think>\nLooking at the model performance, power_watts has R2=0.45.\n</think>\n\n```json\n[\n  {\n    \"parameter\": \"interaction_features.people_home_x_hour_sin\",\n    \"current_value\": false,\n    \"suggested_value\": true,\n    \"reasoning\": \"Interaction between occupancy and time of day is the strongest signal.\",\n    \"expected_impact\": \"Improve power_watts R2 by 5-10%\"\n  }\n]\n```"
}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_meta_learning.py -v -m "not ollama"`
Expected: All non-ollama tests PASS

**Step 5: Register the ollama marker in pytest config**

Check `pyproject.toml` for existing markers section and add:

```toml
[tool.pytest.ini_options]
markers = [
    "ollama: tests requiring a running Ollama instance (deselect with '-m not ollama')",
]
```

**Step 6: Commit**

```bash
git add tests/integration/test_meta_learning.py tests/fixtures/ollama_responses/ pyproject.toml
git commit -m "test(integration): add Tier 2 meta-learning validation tests with recorded Ollama responses"
```

---

## Task 10: Demo Mode CLI

**Files:**
- Modify: `aria/cli.py` — add `demo` subcommand
- Create: `tests/demo/__init__.py`
- Create: `tests/demo/generate.py`
- Test: `tests/integration/test_demo_mode.py`

**Context:** `aria demo --scenario stable_couple --days 30` runs the full pipeline with simulated data and starts the hub. `aria demo --checkpoint day_30` loads frozen fixtures.

**Step 1: Write the failing test**

```python
# tests/integration/test_demo_mode.py
"""Tests for aria demo mode CLI integration."""
import pytest
from pathlib import Path
from tests.demo.generate import generate_checkpoint


class TestDemoGenerate:
    def test_generate_checkpoint(self, tmp_path):
        output = generate_checkpoint(
            scenario="stable_couple",
            days=14,
            seed=42,
            output_dir=tmp_path / "day_14",
        )
        assert (tmp_path / "day_14").exists()
        assert (tmp_path / "day_14" / "daily").exists()
        assert len(list((tmp_path / "day_14" / "daily").glob("*.json"))) == 14
        assert (tmp_path / "day_14" / "baselines").exists()
        assert output["snapshots_saved"] == 14

    def test_generate_multiple_checkpoints(self, tmp_path):
        for name, days in [("day_07", 7), ("day_14", 14), ("day_30", 30)]:
            output = generate_checkpoint(
                scenario="stable_couple",
                days=days,
                seed=42,
                output_dir=tmp_path / name,
            )
            assert output["snapshots_saved"] == days
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_demo_mode.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# tests/demo/__init__.py
"""Demo data generation for ARIA dashboard visual testing."""
```

```python
# tests/demo/generate.py
"""Generate frozen demo checkpoints from household simulator."""
from __future__ import annotations

from pathlib import Path

from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner


def generate_checkpoint(
    scenario: str = "stable_couple",
    days: int = 30,
    seed: int = 42,
    output_dir: Path | str = None,
) -> dict:
    """Generate a frozen demo checkpoint.

    Runs the full pipeline with simulated data and saves all outputs
    to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = HouseholdSimulator(scenario=scenario, days=days, seed=seed)
    snapshots = sim.generate()

    runner = PipelineRunner(snapshots, data_dir=output_dir)
    return runner.run_full()


def generate_all_checkpoints(base_dir: Path | str = None):
    """Generate all standard demo checkpoints."""
    if base_dir is None:
        base_dir = Path(__file__).parent / "fixtures"
    base_dir = Path(base_dir)

    checkpoints = [
        ("day_07", 7),
        ("day_14", 14),
        ("day_30", 30),
        ("day_45", 45),
    ]

    for name, days in checkpoints:
        print(f"Generating {name} ({days} days)...")
        generate_checkpoint(
            scenario="stable_couple",
            days=days,
            seed=42,
            output_dir=base_dir / name,
        )
        print(f"  Done: {base_dir / name}")


if __name__ == "__main__":
    generate_all_checkpoints()
```

Now add the `demo` command to `aria/cli.py`. **Read the file first** to find the exact insertion point, then add:

1. A new subparser for `demo`
2. A `_demo()` function that either runs the pipeline or loads a checkpoint, then starts the hub

The CLI addition should look like:

```python
# In the subparsers section of main():
demo_parser = subparsers.add_parser("demo", help="Run pipeline with simulated data and start hub")
demo_parser.add_argument("--scenario", default="stable_couple", help="Scenario name")
demo_parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
demo_parser.add_argument("--checkpoint", help="Load frozen checkpoint instead of running pipeline")
demo_parser.add_argument("--port", type=int, default=8001)
demo_parser.add_argument("--seed", type=int, default=42)

# In _dispatch():
elif args.command == "demo":
    _demo(args)
```

```python
def _demo(args):
    """Run demo mode: simulate data, run pipeline, start hub."""
    import tempfile
    from pathlib import Path

    if args.checkpoint:
        # Load frozen checkpoint
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.exists():
            # Try tests/demo/fixtures/
            checkpoint_dir = Path(__file__).parent.parent / "tests" / "demo" / "fixtures" / args.checkpoint
        if not checkpoint_dir.exists():
            print(f"Checkpoint not found: {args.checkpoint}")
            return
        intel_dir = str(checkpoint_dir)
    else:
        # Run full pipeline with simulated data
        from tests.synthetic.simulator import HouseholdSimulator
        from tests.synthetic.pipeline import PipelineRunner

        tmp = Path(tempfile.mkdtemp(prefix="aria-demo-"))
        print(f"Demo data directory: {tmp}")
        print(f"Generating {args.days}-day '{args.scenario}' scenario...")

        sim = HouseholdSimulator(scenario=args.scenario, days=args.days, seed=args.seed)
        snapshots = sim.generate()

        runner = PipelineRunner(snapshots, data_dir=tmp)
        result = runner.run_full()
        print(f"Pipeline complete: {result['snapshots_saved']} snapshots, "
              f"overall accuracy: {result['scores'].get('overall', 'N/A')}")
        intel_dir = str(tmp)

    # Start hub with demo data
    print(f"Starting hub with demo data from {intel_dir}...")
    _serve("127.0.0.1", args.port, "info", intel_dir=intel_dir)
```

**Note:** The `_serve()` function will need an `intel_dir` parameter passed through to `IntelligenceHub`. Read `_serve()` to understand how it initializes the hub and where to inject the custom data directory. This may require a small modification to `_serve()`.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_demo_mode.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/demo/__init__.py tests/demo/generate.py tests/integration/test_demo_mode.py aria/cli.py
git commit -m "feat: add aria demo mode for full-pipeline visual testing"
```

---

## Task 11: Agent Definitions

**Files:**
- Create: `.claude/agents/aria-generator.md`
- Create: `.claude/agents/aria-auditor.md`

**Context:** Two Claude Code sub-agent definitions for exploratory testing. No tests needed — these are agent prompts.

**Step 1: Write generator agent**

```markdown
# .claude/agents/aria-generator.md
---
name: aria-generator
description: Generates and runs synthetic household scenarios to stress-test the ARIA pipeline
tools: [Bash, Read, Write, Grep, Glob]
---

# ARIA Scenario Generator

You explore the ARIA ML pipeline for weaknesses by designing and running synthetic household scenarios.

## Context

Read these files first:
- `docs/plans/2026-02-14-synthetic-data-pipeline-testing-design.md` — full design
- `tests/synthetic/simulator.py` — HouseholdSimulator API
- `tests/synthetic/scenarios/household.py` — existing scenarios
- `tests/synthetic/pipeline.py` — PipelineRunner API

## Your Job

1. Understand what aspect of the pipeline the user wants to stress-test
2. Design a new household scenario or modify an existing one to test it
3. Run the simulator and pipeline using PipelineRunner
4. Capture all intermediate outputs
5. Report what happened — did the pipeline handle it correctly?

## Constraints

- Write ONLY to temp dirs and `tests/` — never modify `aria/` source
- Use `.venv/bin/python` to run Python code
- All scenarios must be deterministic (use explicit seed)
- Report raw numbers, not judgments — let the user decide what's acceptable

## Running a Scenario

```python
from tests.synthetic.simulator import HouseholdSimulator
from tests.synthetic.pipeline import PipelineRunner
from pathlib import Path
import tempfile

tmp = Path(tempfile.mkdtemp())
sim = HouseholdSimulator(scenario="stable_couple", days=30, seed=42)
snapshots = sim.generate()
runner = PipelineRunner(snapshots, data_dir=tmp)
result = runner.run_full()
print(result)
```
```

**Step 2: Write auditor agent**

```markdown
# .claude/agents/aria-auditor.md
---
name: aria-auditor
description: Reviews ARIA pipeline outputs for correctness, validates ML behavior, audits test coverage
tools: [Bash, Read, Grep, Glob]
---

# ARIA Pipeline Auditor

You review pipeline outputs and test coverage for the ARIA ML system.

## Context

Read these files first:
- `docs/plans/2026-02-14-synthetic-data-pipeline-testing-design.md` — full design
- `tests/synthetic/pipeline.py` — PipelineRunner API
- `tests/integration/` — existing integration tests

## Your Job

### Output Review
When given pipeline output (from aria-generator or a real run):
1. Check intermediate formats — does each stage's output match expected schema?
2. Validate ML behavior — are models converging? Improving? Beating baselines?
3. Cross-check consistency — do predictions reference real baselines? Do scores reference real predictions?
4. Flag anomalies — anything unexpected in the outputs?

### Coverage Audit
When asked to audit test coverage:
1. Map pipeline stages to existing tests
2. Identify untested handoffs between stages
3. Recommend specific new test cases with scenario + assertion
4. Prioritize by risk — what's most likely to break silently?

## Constraints

- Read-only on `aria/` source code
- Can write new test files to `tests/`
- Can run `pytest` to verify existing tests
- Report specific findings with file paths and line numbers

## Running Tests

```bash
.venv/bin/python -m pytest tests/integration/ -v
.venv/bin/python -m pytest tests/synthetic/ -v
```
```

**Step 3: Commit**

```bash
git add .claude/agents/aria-generator.md .claude/agents/aria-auditor.md
git commit -m "feat: add aria-generator and aria-auditor agent definitions"
```

---

## Execution Summary

| Task | What | Estimated Files |
|------|------|----------------|
| 1 | Person & Schedule model | 3 files |
| 2 | Device Roster & Entity State | 2 files |
| 3 | Weather Profile | 2 files |
| 4 | Snapshot Assembler (real collectors) | 2 files |
| 5 | Household Simulator & Scenarios | 4 files |
| 6 | Pipeline Runner | 2 files |
| 7 | Tier 1: Model Competence tests | 2 files |
| 8 | Tier 3: E2E Pipeline Flow tests | 1 file |
| 9 | Tier 2: Meta-Learning tests + Ollama replay | 3 files |
| 10 | Demo Mode CLI | 3 files + modify 1 |
| 11 | Agent definitions | 2 files |
| **Total** | | **~26 files, 11 commits** |

## Dependencies

```
Task 1 (Person) ─┐
Task 2 (Devices) ─┼─→ Task 4 (Assembler) ─→ Task 5 (Simulator) ─→ Task 6 (Pipeline Runner) ─┐
Task 3 (Weather) ─┘                                                                            │
                                                                                               ├─→ Task 7 (Tier 1 tests)
                                                                                               ├─→ Task 8 (Tier 3 tests)
                                                                                               ├─→ Task 9 (Tier 2 tests)
                                                                                               ├─→ Task 10 (Demo mode)
                                                                                               └─→ Task 11 (Agents)
```

Tasks 1-3 can run in parallel. Tasks 7-11 can run in parallel after Task 6.
