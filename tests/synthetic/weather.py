"""Weather profile generation for household simulation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

SOUTHEAST_US = {
    1: {"high": 48, "low": 30, "humidity": 65, "wind": 8, "sunrise": 7.1, "sunset": 17.3},
    2: {"high": 53, "low": 33, "humidity": 62, "wind": 9, "sunrise": 6.8, "sunset": 17.8},
    3: {"high": 62, "low": 40, "humidity": 58, "wind": 10, "sunrise": 6.2, "sunset": 18.3},
    4: {"high": 72, "low": 49, "humidity": 55, "wind": 9, "sunrise": 5.5, "sunset": 18.8},
    5: {"high": 80, "low": 58, "humidity": 60, "wind": 7, "sunrise": 5.1, "sunset": 19.3},
    6: {"high": 87, "low": 66, "humidity": 65, "wind": 6, "sunrise": 5.0, "sunset": 19.6},
    7: {"high": 90, "low": 70, "humidity": 70, "wind": 5, "sunrise": 5.2, "sunset": 19.5},
    8: {"high": 89, "low": 69, "humidity": 70, "wind": 5, "sunrise": 5.5, "sunset": 19.1},
    9: {"high": 83, "low": 62, "humidity": 65, "wind": 6, "sunrise": 6.0, "sunset": 18.4},
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
        daily_high = self.avg_high + rng.gauss(0, 4)
        daily_low = self.avg_low + rng.gauss(0, 3)
        daily_humidity = max(20, min(100, self.avg_humidity + rng.gauss(0, 8)))
        daily_wind = max(0, self.avg_wind + rng.gauss(0, 3))

        t_range = daily_high - daily_low
        temp = daily_low + t_range * 0.5 * (1 + math.sin(math.pi * (hour - 5) / 20))

        return {
            "temp_f": round(temp, 1),
            "humidity_pct": round(daily_humidity, 1),
            "wind_mph": round(max(0, daily_wind), 1),
        }
