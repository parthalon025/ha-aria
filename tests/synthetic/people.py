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
    _jitter_std: float = 0.3

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

        wake = times["wake"]
        transitions.append((wake, "bedroom"))
        transitions.append((wake + 0.1 + rng.random() * 0.2, "bathroom"))
        transitions.append((wake + 0.4 + rng.random() * 0.2, "kitchen"))

        if "depart" in times:
            transitions.append((times["depart"], "away"))
            transitions.append((times["arrive"], "kitchen"))
            transitions.append((times["arrive"] + 0.5 + rng.random() * 0.5, "living_room"))
        else:
            hour = wake + 1.5
            for room in rng.sample(self.rooms, min(3, len(self.rooms))):
                transitions.append((hour, room))
                hour += 1.0 + rng.random() * 1.5

        sleep = times["sleep"]
        transitions.append((sleep - 1.5, "living_room"))
        transitions.append((sleep - 0.3, "bathroom"))
        transitions.append((sleep, "bedroom"))

        transitions.sort(key=lambda t: t[0])
        return transitions
