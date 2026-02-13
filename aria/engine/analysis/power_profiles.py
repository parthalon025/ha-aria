"""Appliance power profiling and cycle detection.

Learns power consumption profiles for individual smart plug outlets.
Detects cycle start/stop, matches against reference profiles, and
tracks appliance health via degradation indicators.

Inspired by HA WashData — learns appliance fingerprints from power data.
Operates on per-outlet wattage from USP PDU Pro smart plugs.
"""

from collections import defaultdict
from datetime import datetime

import numpy as np


# Default thresholds for cycle detection
DEFAULT_ON_THRESHOLD = 5.0     # Watts — outlet is "active" above this
DEFAULT_OFF_THRESHOLD = 2.0    # Watts — outlet is "idle" below this
DEFAULT_MIN_CYCLE_MINUTES = 1  # Minimum cycle duration
DEFAULT_MAX_CYCLE_HOURS = 24   # Maximum cycle duration


class ApplianceProfile:
    """Learned power profile for a single outlet/appliance."""

    def __init__(self, name, reference_watts=None, typical_duration_minutes=None,
                 peak_watts=None, idle_watts=None):
        self.name = name
        self.reference_watts = reference_watts or []
        self.typical_duration_minutes = typical_duration_minutes
        self.peak_watts = peak_watts or 0.0
        self.idle_watts = idle_watts or 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "reference_watts": [round(w, 1) for w in self.reference_watts],
            "typical_duration_minutes": self.typical_duration_minutes,
            "peak_watts": round(self.peak_watts, 1),
            "idle_watts": round(self.idle_watts, 1),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            reference_watts=data.get("reference_watts", []),
            typical_duration_minutes=data.get("typical_duration_minutes"),
            peak_watts=data.get("peak_watts", 0),
            idle_watts=data.get("idle_watts", 0),
        )


class ApplianceProfiler:
    """Learns and manages power profiles for smart plug outlets.

    Tracks per-outlet power consumption over time, detects active cycles,
    and builds reference profiles for appliance identification and health
    monitoring.
    """

    def __init__(self, on_threshold=DEFAULT_ON_THRESHOLD,
                 off_threshold=DEFAULT_OFF_THRESHOLD,
                 min_cycle_minutes=DEFAULT_MIN_CYCLE_MINUTES):
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_cycle_minutes = min_cycle_minutes
        self.profiles = {}  # name -> ApplianceProfile

    def detect_cycles(self, power_series):
        """Detect on/off cycles from a time series of power readings.

        Args:
            power_series: List of (timestamp_str, watts) tuples sorted by time.

        Returns:
            List of cycle dicts with start, end, duration_minutes,
            peak_watts, avg_watts, energy_wh.
        """
        if len(power_series) < 2:
            return []

        cycles = []
        in_cycle = False
        cycle_start = None
        cycle_readings = []

        for ts_str, watts in power_series:
            ts = self._parse_ts(ts_str)
            if ts is None:
                continue

            if not in_cycle and watts >= self.on_threshold:
                # Cycle starts
                in_cycle = True
                cycle_start = ts
                cycle_readings = [watts]
            elif in_cycle and watts < self.off_threshold:
                # Cycle ends
                in_cycle = False
                duration = (ts - cycle_start).total_seconds() / 60
                if duration >= self.min_cycle_minutes and cycle_readings:
                    cycles.append(self._build_cycle(
                        cycle_start, ts, cycle_readings, duration
                    ))
                cycle_readings = []
            elif in_cycle:
                cycle_readings.append(watts)

        return cycles

    def learn_profile(self, outlet_name, cycles):
        """Learn or update an appliance profile from observed cycles.

        Args:
            outlet_name: Name of the outlet/appliance.
            cycles: List of cycle dicts from detect_cycles().

        Returns:
            ApplianceProfile or None if insufficient cycles.
        """
        if len(cycles) < 2:
            return None

        durations = [c["duration_minutes"] for c in cycles]
        peaks = [c["peak_watts"] for c in cycles]
        avgs = [c["avg_watts"] for c in cycles]

        # Build reference profile from median cycle
        median_idx = len(durations) // 2
        sorted_by_dur = sorted(range(len(cycles)), key=lambda i: durations[i])
        reference_idx = sorted_by_dur[median_idx]

        # Normalize reference to fixed number of points for comparison
        reference = self._normalize_series(cycles[reference_idx].get("readings", avgs), 20)

        profile = ApplianceProfile(
            name=outlet_name,
            reference_watts=reference,
            typical_duration_minutes=round(float(np.median(durations)), 1),
            peak_watts=round(float(np.median(peaks)), 1),
            idle_watts=round(min(c.get("min_watts", 0) for c in cycles), 1),
        )

        self.profiles[outlet_name] = profile
        return profile

    def assess_health(self, outlet_name, recent_cycles, n_baseline=5):
        """Assess appliance health by comparing recent cycles to profile.

        Args:
            outlet_name: Name of the outlet.
            recent_cycles: Recent cycle dicts to evaluate.
            n_baseline: Number of recent cycles to use for assessment.

        Returns:
            Health assessment dict with score, indicators, and alerts.
        """
        profile = self.profiles.get(outlet_name)
        if profile is None or len(recent_cycles) < 2:
            return {"score": None, "reason": "insufficient data"}

        recent = recent_cycles[-n_baseline:]
        indicators = {}
        alerts = []

        # Duration stability
        durations = [c["duration_minutes"] for c in recent]
        if profile.typical_duration_minutes and profile.typical_duration_minutes > 0:
            dur_ratio = np.mean(durations) / profile.typical_duration_minutes
            indicators["duration_ratio"] = round(float(dur_ratio), 2)
            if dur_ratio > 1.3:
                alerts.append(f"cycles {int((dur_ratio - 1) * 100)}% longer than typical")
            elif dur_ratio < 0.7:
                alerts.append(f"cycles {int((1 - dur_ratio) * 100)}% shorter than typical")

        # Peak power stability
        peaks = [c["peak_watts"] for c in recent]
        if profile.peak_watts > 0:
            peak_ratio = np.mean(peaks) / profile.peak_watts
            indicators["peak_power_ratio"] = round(float(peak_ratio), 2)
            if peak_ratio > 1.2:
                alerts.append(f"peak power {int((peak_ratio - 1) * 100)}% above baseline")
            elif peak_ratio < 0.8:
                alerts.append(f"peak power {int((1 - peak_ratio) * 100)}% below baseline")

        # Energy consistency
        energies = [c.get("energy_wh", 0) for c in recent]
        if len(energies) >= 3 and np.mean(energies) > 0:
            cv = float(np.std(energies) / np.mean(energies))
            indicators["energy_cv"] = round(cv, 3)
            if cv > 0.3:
                alerts.append(f"energy consumption highly variable (CV={cv:.2f})")

        # Overall health score (0-100)
        score = 100
        for key, ratio in indicators.items():
            if key.endswith("_ratio"):
                deviation = abs(ratio - 1.0)
                score -= deviation * 50  # -50 per 100% deviation

        if indicators.get("energy_cv", 0) > 0.3:
            score -= 15

        score = max(0, min(100, round(score)))

        return {
            "score": score,
            "indicators": indicators,
            "alerts": alerts,
            "cycles_analyzed": len(recent),
        }

    def analyze_snapshot_outlets(self, snapshots):
        """Analyze per-outlet power data from multiple snapshots.

        Args:
            snapshots: List of (date_str, snapshot_dict) tuples.

        Returns:
            Dict with per-outlet summaries, active outlets, and profiles.
        """
        # Collect per-outlet time series
        outlet_series = defaultdict(list)
        for date_str, snap in snapshots:
            outlets = snap.get("power", {}).get("outlets", {})
            for name, watts in outlets.items():
                try:
                    outlet_series[name].append((date_str, float(watts)))
                except (ValueError, TypeError):
                    continue

        results = {"outlets": {}, "active_count": 0, "profiles_learned": 0}

        for name, series in outlet_series.items():
            if len(series) < 2:
                continue

            # Detect cycles
            cycles = self.detect_cycles(series)

            # Compute summary stats
            watts_values = [w for _, w in series]
            avg_watts = float(np.mean(watts_values))
            max_watts = float(np.max(watts_values))
            is_active = avg_watts > self.on_threshold

            outlet_info = {
                "avg_watts": round(avg_watts, 1),
                "max_watts": round(max_watts, 1),
                "is_active": is_active,
                "cycles_detected": len(cycles),
                "readings": len(series),
            }

            # Learn profile if enough cycles
            profile = self.learn_profile(name, cycles)
            if profile:
                outlet_info["profile"] = profile.to_dict()
                results["profiles_learned"] += 1

            # Health assessment if we have a profile
            if profile and len(cycles) >= 3:
                health = self.assess_health(name, cycles)
                outlet_info["health"] = health

            results["outlets"][name] = outlet_info
            if is_active:
                results["active_count"] += 1

        return results

    def _build_cycle(self, start, end, readings, duration_minutes):
        """Build a cycle dict from raw data."""
        return {
            "start": start.isoformat() if isinstance(start, datetime) else str(start),
            "end": end.isoformat() if isinstance(end, datetime) else str(end),
            "duration_minutes": round(duration_minutes, 1),
            "peak_watts": round(float(max(readings)), 1),
            "avg_watts": round(float(np.mean(readings)), 1),
            "min_watts": round(float(min(readings)), 1),
            "energy_wh": round(float(np.mean(readings) * duration_minutes / 60), 2),
            "readings": readings,
        }

    @staticmethod
    def _normalize_series(values, n_points):
        """Normalize a time series to a fixed number of points via interpolation."""
        if not values:
            return [0.0] * n_points
        if len(values) == n_points:
            return [float(v) for v in values]

        x_orig = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, n_points)
        interpolated = np.interp(x_new, x_orig, values)
        return [round(float(v), 1) for v in interpolated]

    @staticmethod
    def _parse_ts(ts_str):
        """Parse a timestamp string."""
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(ts_str, fmt)
            except (ValueError, TypeError):
                continue
        return None


def profile_correlation(profile_a, profile_b):
    """Compute correlation between two appliance profiles.

    Returns Pearson correlation coefficient (-1 to 1).
    Useful for detecting when outlets have similar consumption patterns.
    """
    ref_a = profile_a.reference_watts
    ref_b = profile_b.reference_watts

    if not ref_a or not ref_b or len(ref_a) != len(ref_b):
        return 0.0

    a = np.array(ref_a)
    b = np.array(ref_b)

    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0

    return round(float(np.corrcoef(a, b)[0, 1]), 3)
