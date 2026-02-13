"""Bayesian occupancy estimation from multi-sensor fusion.

Replaces binary people_home with probabilistic per-area occupancy scores.
Combines motion sensors (fast decay), power draw (slow decay), and
device tracker data into Bayesian posterior probabilities.

Inspired by Area Occupancy Detection's sensor fusion approach.
Degrades gracefully — works with whatever sensors are available.
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta


# Sensor type weights and decay rates (seconds)
SENSOR_CONFIG = {
    "motion": {"weight": 0.9, "decay_seconds": 300},     # High confidence, 5 min decay
    "door": {"weight": 0.6, "decay_seconds": 600},        # Medium, 10 min decay
    "media": {"weight": 0.4, "decay_seconds": 1800},      # Low, 30 min decay (TV stays on)
    "power": {"weight": 0.3, "decay_seconds": 3600},       # Low, 1 hour decay
    "device_tracker": {"weight": 0.5, "decay_seconds": 0}, # Binary, no decay
}

# Prior probability of occupancy when no sensor data available
DEFAULT_PRIOR = 0.3


class BayesianOccupancy:
    """Multi-sensor Bayesian occupancy estimator.

    Maintains per-area occupancy probability by fusing signals from
    motion sensors, door sensors, media players, power draw, and
    device trackers. Learns hourly/daily priors from historical data.
    """

    def __init__(self, area_sensors=None, priors=None):
        """Initialize occupancy estimator.

        Args:
            area_sensors: Optional dict mapping area names to sensor configs.
                Format: {"living_room": {"motion": ["binary_sensor.motion_lr"], ...}}
                If None, uses auto-discovery from snapshot data.
            priors: Optional dict of learned priors by (day_of_week, hour).
                Format: {("Monday", 14): 0.85, ...}
        """
        self.area_sensors = area_sensors or {}
        self.priors = priors or {}

    def estimate(self, snapshot, timestamp=None):
        """Estimate per-area occupancy from current snapshot.

        Args:
            snapshot: Current HA snapshot dict with motion, power, media, occupancy sections.
            timestamp: Optional datetime for time-based priors. Defaults to now.

        Returns:
            Dict mapping area names to occupancy info:
            {
                "overall": {"probability": 0.92, "confidence": "high", "signals": [...]},
                "living_room": {"probability": 0.85, ...},
            }
        """
        if timestamp is None:
            timestamp = datetime.now()

        results = {}

        # Overall home occupancy (from device trackers + global signals)
        overall = self._estimate_overall(snapshot, timestamp)
        results["overall"] = overall

        # Per-area estimates (if area_sensors configured)
        for area, sensors in self.area_sensors.items():
            area_result = self._estimate_area(area, sensors, snapshot, timestamp)
            results[area] = area_result

        return results

    def _estimate_overall(self, snapshot, timestamp):
        """Estimate overall home occupancy probability."""
        signals = []

        # Device tracker signal (strongest for home/away)
        occ = snapshot.get("occupancy", {})
        people_home = occ.get("people_home", [])
        device_count = occ.get("device_count_home", 0)

        if people_home:
            signals.append(("device_tracker", 1.0, f"{len(people_home)} people home"))
        elif device_count > 5:
            signals.append(("device_tracker", 0.7, f"{device_count} devices home"))
        else:
            signals.append(("device_tracker", 0.1, "no people/devices"))

        # Motion signal (any active motion sensor)
        motion = snapshot.get("motion", {}).get("sensors", {})
        active_motion = sum(1 for v in motion.values() if v == "on")
        if active_motion > 0:
            signals.append(("motion", 1.0, f"{active_motion} motion sensor(s) active"))
        elif motion:
            signals.append(("motion", 0.2, "motion sensors quiet"))

        # Power signal (high power draw suggests activity)
        power = snapshot.get("power", {}).get("total_watts", 0)
        if power > 100:
            power_signal = min(1.0, power / 500)
            signals.append(("power", power_signal, f"{power:.0f}W draw"))
        else:
            signals.append(("power", 0.1, f"{power:.0f}W (standby)"))

        # Media signal
        media = snapshot.get("media", {})
        active_media = sum(1 for v in media.values()
                          if isinstance(v, str) and v in ("playing", "on"))
        if active_media > 0:
            signals.append(("media", 0.9, f"{active_media} media playing"))

        # Fuse signals using Bayesian update
        prior = self._get_prior(timestamp)
        probability = self._bayesian_fuse(prior, signals)

        return {
            "probability": round(probability, 3),
            "confidence": self._classify_confidence(probability, len(signals)),
            "signals": [{"type": t, "value": round(v, 2), "detail": d}
                        for t, v, d in signals],
            "prior": round(prior, 3),
        }

    def _estimate_area(self, area, sensors, snapshot, timestamp):
        """Estimate occupancy for a specific area."""
        signals = []

        # Motion sensors for this area
        motion_data = snapshot.get("motion", {}).get("sensors", {})
        for sensor_name in sensors.get("motion", []):
            for name, state in motion_data.items():
                if sensor_name in name or name in sensor_name:
                    value = 1.0 if state == "on" else 0.1
                    signals.append(("motion", value, f"{name}: {state}"))

        # Power sensors for this area
        outlets = snapshot.get("power", {}).get("outlets", {})
        for outlet_name in sensors.get("power", []):
            for name, watts in outlets.items():
                if outlet_name in name or name in outlet_name:
                    value = min(1.0, watts / 50) if watts > 5 else 0.1
                    signals.append(("power", value, f"{name}: {watts:.1f}W"))

        if not signals:
            return {
                "probability": DEFAULT_PRIOR,
                "confidence": "none",
                "signals": [],
                "prior": DEFAULT_PRIOR,
            }

        prior = self._get_prior(timestamp, area)
        probability = self._bayesian_fuse(prior, signals)

        return {
            "probability": round(probability, 3),
            "confidence": self._classify_confidence(probability, len(signals)),
            "signals": [{"type": t, "value": round(v, 2), "detail": d}
                        for t, v, d in signals],
            "prior": round(prior, 3),
        }

    def _bayesian_fuse(self, prior, signals):
        """Fuse multiple sensor signals using Bayesian updating.

        Each signal contributes evidence for/against occupancy based on
        its type's configured weight. Uses log-odds for numerical stability.
        """
        if not signals:
            return prior

        # Convert prior to log-odds
        prior = max(0.01, min(0.99, prior))
        log_odds = math.log(prior / (1 - prior))

        for sensor_type, value, _ in signals:
            config = SENSOR_CONFIG.get(sensor_type, {"weight": 0.3})
            weight = config["weight"]

            # Convert sensor value to evidence (log-likelihood ratio)
            # value=1.0 → strong positive evidence, value=0.0 → strong negative
            value = max(0.01, min(0.99, value))
            evidence = weight * math.log(value / (1 - value))
            log_odds += evidence

        # Convert back to probability
        probability = 1.0 / (1.0 + math.exp(-log_odds))
        return max(0.0, min(1.0, probability))

    def _get_prior(self, timestamp, area=None):
        """Get learned prior for this time slot.

        Falls back to DEFAULT_PRIOR if no learned priors available.
        """
        dow = timestamp.strftime("%A")
        hour = timestamp.hour
        key = (dow, hour, area) if area else (dow, hour)

        return self.priors.get(key, DEFAULT_PRIOR)

    @staticmethod
    def _classify_confidence(probability, signal_count):
        """Classify confidence based on probability strength and signal count."""
        if signal_count == 0:
            return "none"
        if signal_count < 2:
            return "low"
        if 0.3 < probability < 0.7:
            return "low"
        if signal_count >= 3 and (probability > 0.8 or probability < 0.2):
            return "high"
        return "medium"


def learn_occupancy_priors(daily_snapshots, timestamps=None):
    """Learn hourly occupancy priors from historical snapshot data.

    Args:
        daily_snapshots: List of snapshot dicts with occupancy data.
        timestamps: Optional list of (day_of_week, hour) tuples corresponding
            to each snapshot. If None, uses snapshot metadata.

    Returns:
        Dict mapping (day_of_week, hour) to average occupancy probability.
    """
    slot_values = defaultdict(list)

    for i, snap in enumerate(daily_snapshots):
        # Determine time slot
        if timestamps and i < len(timestamps):
            dow, hour = timestamps[i]
        else:
            # Try to extract from snapshot metadata
            meta_date = snap.get("metadata", {}).get("date", "")
            meta_time = snap.get("metadata", {}).get("time", "12:00")
            if meta_date:
                try:
                    dt = datetime.strptime(f"{meta_date} {meta_time}", "%Y-%m-%d %H:%M")
                    dow = dt.strftime("%A")
                    hour = dt.hour
                except ValueError:
                    continue
            else:
                continue

        # Extract occupancy signal
        occ = snap.get("occupancy", {})
        people = occ.get("people_home", [])
        is_occupied = 1.0 if people else 0.0

        slot_values[(dow, hour)].append(is_occupied)

    # Average per slot
    priors = {}
    for key, values in slot_values.items():
        if len(values) >= 3:  # Minimum samples for a reliable prior
            priors[key] = round(sum(values) / len(values), 3)

    return priors


def occupancy_to_features(occupancy_result):
    """Convert occupancy estimation to feature vector values.

    Returns dict of feature names and values suitable for ML feature vectors.
    Replaces the binary people_home with continuous probability.
    """
    overall = occupancy_result.get("overall", {})
    features = {
        "occupancy_probability": overall.get("probability", DEFAULT_PRIOR),
        "occupancy_signal_count": len(overall.get("signals", [])),
    }

    # Per-area probabilities (if available)
    for area, data in occupancy_result.items():
        if area == "overall":
            continue
        features[f"occupancy_{area}"] = data.get("probability", DEFAULT_PRIOR)

    return features
