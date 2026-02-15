# Home Assistant Intelligence Engine v2 — ML Prediction & Learning Design

> **Status: IMPLEMENTED** (2026-02-10). 78 tests passing. Script at `~/.local/bin/ha-intelligence` (2,373 lines).

## In Plain English

Version 2 upgrades ARIA's brain from simple averages to real machine learning. Instead of just saying "this is above average," it now *predicts* what your home will do tomorrow — which rooms will be active, what temperature patterns to expect, which devices are likely to fail. It takes snapshots 7 times per day, trains ML models on the accumulated data, and uses a meta-learning loop where an AI reviews the predictions' accuracy and tunes the system to get better over time.

## Why This Exists

v1's statistical approach (averages and z-scores) works for simple anomaly detection but can't predict future behavior or learn from its mistakes. Real ML models (GradientBoosting, RandomForest, IsolationForest) can find complex, non-linear patterns: "When it's cold outside AND a weekday AND after sunset, the living room lights come on 15 minutes earlier than usual." The meta-learning loop is the key differentiator — it means the system automatically gets smarter without manual tuning.

**Goal:** Upgrade the existing statistical HA intelligence engine with sklearn ML models, comprehensive intra-day data capture, and a deepseek-r1:8b meta-learning loop that self-improves prediction accuracy over time — all running locally.

**Key decisions:**
- **Single LLM:** deepseek-r1:8b for all Ollama tasks (reports, meta-learning, anomaly interpretation)
- **sklearn + numpy:** GradientBoosting for continuous predictions, RandomForest for device failure, IsolationForest for contextual anomalies
- **Intra-day snapshots:** 7/day (every 4 hours + 11pm full pipeline)
- **~80-100 features** per snapshot (raw + derived + cyclical time encoding)
- **Auto-apply meta-learning** with guardrails (validate before applying, max 3 changes/week)
- **Gradual activation:** Day 1-7 stats, Day 8+ LLM reports, Day 14+ sklearn + meta-learning

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ha-intelligence v2                          │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   Layer 1     │   │   Layer 2     │   │    Layer 3          │  │
│  │  Statistical  │   │    sklearn    │   │  LLM Meta-Learning  │  │
│  │  Baselines    │   │   ML Models   │   │  (deepseek-r1:8b)   │  │
│  │              │   │              │   │                    │  │
│  │ - mean/stdev │   │ - GradBoost  │   │ - accuracy review  │  │
│  │ - EWMA       │   │ - RandomFor. │   │ - error analysis   │  │
│  │ - z-score    │   │ - IsoForest  │   │ - feature suggest  │  │
│  │              │   │              │   │ - auto-apply       │  │
│  │ Day 1+       │   │ Day 14+      │   │ Day 14+            │  │
│  └──────┬───────┘   └──────┬───────┘   └────────┬───────────┘  │
│         │                  │                     │              │
│         └──────┬───────────┴─────────────────────┘              │
│                │                                                │
│         ┌──────▼───────┐                                        │
│         │  Prediction   │  Weighted blend:                      │
│         │  Combiner     │  Day 14-60: 70% stats / 30% ML       │
│         │              │  Day 60-90: 50/50                     │
│         │              │  Day 90+:   30% stats / 70% ML        │
│         └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Outputs          │
                    │ - predictions.json │
                    │ - anomalies        │
                    │ - insight reports  │
                    │ - telegram brief   │
                    │ - /ha-predict      │
                    └───────────────────┘
```

**Activation timeline:**
```
Day 1-7:    Layer 1 only (statistical baselines accumulating)
Day 8-13:   Layer 1 + daily deepseek reports
Day 14+:    Layer 1 + Layer 2 (sklearn) + Layer 3 (meta-learning)
             (14 days × 7 snapshots/day = 98 samples — enough for initial training)
Day 60+:    ML predictions weighted higher than statistical
Day 90+:    ML dominates, statistical serves as sanity check
```

---

## File Structure

```
~/.local/bin/ha-intelligence              # Main script (existing, enhanced)
~/ha-logs/intelligence/
  ├── daily/YYYY-MM-DD.json               # End-of-day summary snapshot (existing)
  ├── intraday/                           # NEW — intra-day snapshots
  │   └── YYYY-MM-DD/
  │       ├── 00.json                     # Midnight
  │       ├── 04.json                     # 4am
  │       ├── 08.json                     # 8am
  │       ├── 12.json                     # Noon
  │       ├── 16.json                     # 4pm
  │       ├── 20.json                     # 8pm
  │       └── 23.json                     # 11pm
  ├── baselines.json                      # Statistical baselines (existing, enhanced)
  ├── predictions.json                    # Predictions (existing, enhanced with ML)
  ├── accuracy.json                       # Accuracy history (existing, enhanced per-model)
  ├── correlations.json                   # Correlations (existing)
  ├── insights/YYYY-MM-DD.json            # Daily deepseek reports (existing, model change)
  ├── models/                             # NEW — serialized sklearn models
  │   ├── power_watts.pkl
  │   ├── lights_on.pkl
  │   ├── devices_home.pkl
  │   ├── unavailable.pkl
  │   ├── useful_events.pkl
  │   ├── device_failure.pkl              # RandomForest classifier
  │   ├── anomaly_detector.pkl            # IsolationForest
  │   └── training_log.json              # Training metadata & metrics
  ├── meta-learning/                      # NEW — LLM meta-learning artifacts
  │   ├── suggestions.json                # Current suggestions queue
  │   ├── applied.json                    # History of applied changes
  │   └── weekly/YYYY-WNN.json            # Weekly analysis reports
  └── feature_config.json                 # NEW — feature engineering config
~/Documents/tests/test_ha_intelligence.py # Tests (existing, extended)
```

---

## Comprehensive Data Capture

### Intra-Day Snapshot Schema

Each intra-day snapshot captures the full state at that moment:

```json
{
  "date": "2026-02-10",
  "hour": 16,
  "timestamp": "2026-02-10T16:00:05",

  "time_features": {
    "hour": 16,
    "hour_sin": 0.866,
    "hour_cos": -0.5,
    "dow": 1,
    "dow_sin": 0.782,
    "dow_cos": 0.623,
    "month": 2,
    "month_sin": 0.866,
    "month_cos": 0.5,
    "day_of_year": 41,
    "day_of_year_sin": 0.643,
    "day_of_year_cos": 0.766,
    "is_weekend": false,
    "is_holiday": false,
    "is_night": false,
    "is_work_hours": true,
    "minutes_since_midnight": 960,
    "minutes_since_sunrise": 558,
    "minutes_until_sunset": 118,
    "daylight_remaining_pct": 17.5,
    "week_of_year": 7
  },

  "weather": {
    "raw": "Partly cloudy +72°F 60% →8mph",
    "condition": "Partly cloudy",
    "temp_f": 72,
    "feels_like_f": 74,
    "humidity_pct": 60,
    "wind_mph": 8,
    "pressure_mb": null,
    "visibility_mi": null,
    "cloud_cover_pct": null,
    "uv_index": null
  },

  "sun": {
    "sunrise": "06:42",
    "sunset": "17:58",
    "daylight_hours": 11.27,
    "solar_elevation": 32.5
  },

  "entities": {
    "total": 3065,
    "unavailable": 904,
    "by_domain": {"sensor": 1535, "device_tracker": 557, "...": "..."},
    "unavailable_list": ["sensor.flaky_device", "..."]
  },

  "power": {
    "total_watts": 245.3,
    "outlets": {
      "USP PDU Outlet 1 Power": 12.5,
      "USP PDU Outlet 2 Power": 0.0,
      "...": "..."
    }
  },

  "occupancy": {
    "people_home": ["Justin"],
    "people_away": ["Lisa"],
    "people_home_count": 1,
    "device_count_home": 62,
    "arrival_times_today": {"Justin": "07:30"},
    "departure_times_today": {"Lisa": "08:15"}
  },

  "climate": [
    {
      "name": "Bedroom",
      "state": "cool",
      "current_temp": 73,
      "target_temp": 68,
      "temp_delta": 5,
      "hvac_action": "cooling"
    }
  ],

  "locks": [
    {"name": "Back Door", "state": "locked", "battery": 58}
  ],

  "lights": {
    "on": 8,
    "off": 52,
    "unavailable": 13,
    "total_brightness": 980,
    "avg_brightness": 122.5,
    "rooms_lit": ["atrium", "living_room", "kitchen"]
  },

  "motion": {
    "sensors": {
      "Closet motion": "off",
      "Front door motion": "on",
      "...": "..."
    },
    "active_count": 1,
    "events_since_last_snapshot": 12
  },

  "doors_windows": {
    "front_door": {"state": "closed", "open_count_today": 8},
    "garage_door": {"state": "closed", "open_count_today": 2}
  },

  "batteries": {
    "lock.back_door": {"level": 58, "prev_snapshot_level": 58, "entity_type": "lock"},
    "sensor.hue_motion_2": {"level": 82, "prev_snapshot_level": 82, "entity_type": "sensor"}
  },

  "network": {
    "devices_home": 62,
    "devices_away": 322,
    "devices_unavailable": 169
  },

  "media": {
    "active_players": ["media_player.living_room"],
    "total_active": 1
  },

  "automations": {
    "on": 13,
    "off": 14,
    "unavailable": 4,
    "triggered_since_last_snapshot": ["automation.arrive_justin"]
  },

  "ev": {
    "TARS": {
      "battery_pct": 71,
      "charger_power_kw": 0.0,
      "range_miles": 199.3,
      "is_charging": false,
      "is_home": true
    }
  },

  "vacuum": {
    "status": "docked",
    "battery": 100
  },

  "logbook_summary": {
    "events_since_last_snapshot": 340,
    "useful_events_since_last_snapshot": 85,
    "by_domain": {"device_tracker": 120, "switch": 45, "light": 30, "...": "..."}
  }
}
```

### End-of-Day Summary (enhanced existing daily snapshot)

The existing `daily/YYYY-MM-DD.json` is enhanced with aggregated intra-day data:

```json
{
  "date": "2026-02-10",
  "day_of_week": "Tuesday",
  "is_weekend": false,
  "is_holiday": false,

  "intraday_curves": {
    "power_curve": [80, 75, 120, 180, 245, 200, 150],
    "occupancy_curve": [2, 2, 1, 0, 1, 2, 2],
    "lights_curve": [0, 0, 3, 2, 5, 12, 8],
    "motion_events_curve": [2, 0, 15, 5, 12, 25, 10]
  },

  "daily_aggregates": {
    "power_mean": 150.0,
    "power_max": 245.3,
    "power_min": 75.0,
    "power_std": 60.2,
    "lights_mean": 4.3,
    "lights_max": 12,
    "occupancy_mean_people": 1.4,
    "total_motion_events": 69,
    "total_door_opens": 15,
    "ev_miles_driven": 23.5,
    "ev_energy_consumed_kwh": 8.2,
    "hvac_runtime_hours": 6.5,
    "activity_window_start": "06:45",
    "activity_window_end": "23:20"
  },

  "derived_features": {
    "watts_per_person_home": 107.1,
    "watts_per_degree_delta": 49.1,
    "lights_per_person": 3.1,
    "power_7d_trend": 2.3,
    "occupancy_7d_trend": -0.1
  },

  "batteries_snapshot": {
    "lock.back_door": {"level": 58, "drain_rate_per_day": 0.8, "days_to_empty": 72},
    "...": "..."
  },

  "device_reliability": {
    "newly_unavailable": ["sensor.outdoor_temp"],
    "recovered": ["switch.garage_light"],
    "chronic_unavailable": ["sensor.flaky_device"]
  }
}
```

---

## Feature Engineering

### Feature Config File (`feature_config.json`)

This file defines which features are active. The LLM meta-learner can modify it (with guardrails).

```json
{
  "version": 1,
  "last_modified": "2026-02-10T23:45:00",
  "modified_by": "initial",

  "time_features": {
    "hour_sin_cos": true,
    "dow_sin_cos": true,
    "month_sin_cos": true,
    "day_of_year_sin_cos": true,
    "is_weekend": true,
    "is_holiday": true,
    "is_night": true,
    "is_work_hours": true,
    "minutes_since_sunrise": true,
    "minutes_until_sunset": true,
    "daylight_remaining_pct": true
  },

  "weather_features": {
    "temp_f": true,
    "humidity_pct": true,
    "wind_mph": true,
    "feels_like_f": false,
    "pressure_mb": false,
    "cloud_cover_pct": false
  },

  "home_features": {
    "people_home_count": true,
    "device_count_home": true,
    "lights_on": true,
    "total_brightness": true,
    "doors_open_today": true,
    "motion_events_since_last": true,
    "active_media_players": true,
    "ev_battery_pct": true,
    "ev_is_charging": true
  },

  "lag_features": {
    "prev_snapshot_power": true,
    "prev_snapshot_lights": true,
    "prev_snapshot_occupancy": true,
    "rolling_7d_power_mean": true,
    "rolling_7d_lights_mean": true
  },

  "interaction_features": {
    "is_weekend_x_temp": false,
    "people_home_x_hour_sin": false,
    "daylight_x_lights": false
  },

  "target_metrics": [
    "power_watts",
    "lights_on",
    "devices_home",
    "unavailable",
    "useful_events"
  ]
}
```

### Cyclical Encoding Implementation

```python
import math

def cyclical_encode(value, max_value):
    """Encode a cyclical feature as sin/cos pair."""
    angle = 2 * math.pi * value / max_value
    return math.sin(angle), math.cos(angle)

def build_time_features(timestamp, sun_data):
    """Build all time features from a timestamp and sun data."""
    dt = datetime.fromisoformat(timestamp)
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()
    month = dt.month
    doy = dt.timetuple().tm_yday

    h_sin, h_cos = cyclical_encode(hour, 24)
    d_sin, d_cos = cyclical_encode(dow, 7)
    m_sin, m_cos = cyclical_encode(month, 12)
    y_sin, y_cos = cyclical_encode(doy, 365)

    sunrise_minutes = _time_to_minutes(sun_data.get("sunrise", "06:00"))
    sunset_minutes = _time_to_minutes(sun_data.get("sunset", "18:00"))
    current_minutes = dt.hour * 60 + dt.minute
    daylight_total = sunset_minutes - sunrise_minutes

    return {
        "hour_sin": h_sin, "hour_cos": h_cos,
        "dow_sin": d_sin, "dow_cos": d_cos,
        "month_sin": m_sin, "month_cos": m_cos,
        "day_of_year_sin": y_sin, "day_of_year_cos": y_cos,
        "is_weekend": dow >= 5,
        "is_holiday": ...,  # from holidays lib
        "is_night": current_minutes < sunrise_minutes or current_minutes > sunset_minutes,
        "is_work_hours": not (dow >= 5) and 8 * 60 <= current_minutes <= 17 * 60,
        "minutes_since_sunrise": max(0, current_minutes - sunrise_minutes),
        "minutes_until_sunset": max(0, sunset_minutes - current_minutes),
        "daylight_remaining_pct": max(0, (sunset_minutes - current_minutes) / daylight_total * 100) if daylight_total > 0 else 0,
        "minutes_since_midnight": current_minutes,
        "week_of_year": dt.isocalendar()[1],
    }
```

### Feature Vector Assembly

```python
def build_feature_vector(snapshot, feature_config, prev_snapshot=None, rolling_stats=None):
    """Build the complete feature vector for one snapshot."""
    features = {}

    # Time features
    tf = snapshot.get("time_features", {})
    for key, enabled in feature_config.get("time_features", {}).items():
        if enabled and key in tf:
            features[key] = tf[key]

    # Weather features
    weather = snapshot.get("weather", {})
    for key, enabled in feature_config.get("weather_features", {}).items():
        if enabled:
            features[f"weather_{key}"] = weather.get(key, 0) or 0

    # Home state features
    home_map = {
        "people_home_count": snapshot.get("occupancy", {}).get("people_home_count", 0),
        "device_count_home": snapshot.get("occupancy", {}).get("device_count_home", 0),
        "lights_on": snapshot.get("lights", {}).get("on", 0),
        "total_brightness": snapshot.get("lights", {}).get("total_brightness", 0),
        "doors_open_today": sum(
            d.get("open_count_today", 0)
            for d in snapshot.get("doors_windows", {}).values()
        ),
        "motion_events_since_last": snapshot.get("motion", {}).get("events_since_last_snapshot", 0),
        "active_media_players": snapshot.get("media", {}).get("total_active", 0),
        "ev_battery_pct": snapshot.get("ev", {}).get("TARS", {}).get("battery_pct", 0),
        "ev_is_charging": 1 if snapshot.get("ev", {}).get("TARS", {}).get("is_charging") else 0,
    }
    for key, enabled in feature_config.get("home_features", {}).items():
        if enabled:
            features[key] = home_map.get(key, 0)

    # Lag features
    if prev_snapshot and feature_config.get("lag_features", {}).get("prev_snapshot_power"):
        features["prev_power"] = prev_snapshot.get("power", {}).get("total_watts", 0)
    if prev_snapshot and feature_config.get("lag_features", {}).get("prev_snapshot_lights"):
        features["prev_lights"] = prev_snapshot.get("lights", {}).get("on", 0)
    if rolling_stats:
        if feature_config.get("lag_features", {}).get("rolling_7d_power_mean"):
            features["rolling_7d_power"] = rolling_stats.get("power_mean_7d", 0)

    # Interaction features (LLM meta-learner may enable these)
    for key, enabled in feature_config.get("interaction_features", {}).items():
        if enabled:
            if key == "is_weekend_x_temp":
                features[key] = features.get("is_weekend", 0) * features.get("weather_temp_f", 0)
            elif key == "people_home_x_hour_sin":
                features[key] = features.get("people_home_count", 0) * features.get("hour_sin", 0)
            elif key == "daylight_x_lights":
                features[key] = features.get("daylight_remaining_pct", 0) * features.get("lights_on", 0)

    return features
```

---

## ML Models (scikit-learn)

### Model 1: Continuous Predictions (GradientBoostingRegressor)

One model per target metric: `power_watts`, `lights_on`, `devices_home`, `unavailable`, `useful_events`.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import numpy as np

def train_continuous_model(metric_name, features_list, targets, model_dir):
    """Train a GradientBoosting model for a continuous metric."""
    X = np.array(features_list)
    y = np.array(targets)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Save
    model_path = os.path.join(model_dir, f"{metric_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "metric": metric_name,
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "samples_train": len(X_train),
        "samples_val": len(X_val),
        "feature_importance": dict(zip(feature_names, model.feature_importances_.tolist())),
    }
```

### Model 2: Device Failure Prediction (RandomForestClassifier)

Predicts probability of a device going unavailable within 7 days.

```python
from sklearn.ensemble import RandomForestClassifier

def build_device_failure_features(device_id, snapshots):
    """Build feature vector for device failure prediction."""
    outage_days_7d = 0
    outage_days_30d = 0
    days_since_outage = 999
    battery = -1
    domain = device_id.split(".")[0]

    for i, snap in enumerate(reversed(snapshots)):
        unavail_list = snap.get("entities", {}).get("unavailable_list", [])
        is_unavail = device_id in unavail_list
        days_ago = i

        if is_unavail:
            if days_ago < 7:
                outage_days_7d += 1
            if days_ago < 30:
                outage_days_30d += 1
            days_since_outage = min(days_since_outage, days_ago)

        # Battery from most recent snapshot
        if i == 0:
            batteries = snap.get("batteries", {})
            if device_id in batteries:
                battery = batteries[device_id].get("level", -1) or -1

    # Trend: compare first half to second half of 30-day window
    mid = min(len(snapshots), 30) // 2
    early_outages = sum(
        1 for s in snapshots[:mid]
        if device_id in s.get("entities", {}).get("unavailable_list", [])
    )
    late_outages = sum(
        1 for s in snapshots[mid:min(len(snapshots), 30)]
        if device_id in s.get("entities", {}).get("unavailable_list", [])
    )
    trend = 1 if late_outages > early_outages else (-1 if late_outages < early_outages else 0)

    domain_map = {"sensor": 0, "switch": 1, "light": 2, "binary_sensor": 3,
                  "device_tracker": 4, "lock": 5, "climate": 6}

    return {
        "outage_count_7d": outage_days_7d,
        "outage_count_30d": outage_days_30d,
        "days_since_outage": min(days_since_outage, 365),
        "outage_trend": trend,
        "battery_level": battery,
        "domain_encoded": domain_map.get(domain, 7),
    }

def train_device_failure_model(snapshots, model_dir):
    """Train RandomForest classifier for device failure prediction."""
    # Build training data: for each device that was ever unavailable,
    # create samples from each day with label = "went unavailable within 7 days?"
    # ... (implementation in task)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )
    # ... fit, evaluate, save
```

### Model 3: Contextual Anomaly Detection (IsolationForest)

Unsupervised — learns "normal" and flags multi-dimensional outliers.

```python
from sklearn.ensemble import IsolationForest

def train_anomaly_detector(features_list, model_dir):
    """Train IsolationForest on normal data."""
    X = np.array(features_list)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,  # Expect ~5% of data to be anomalous
        random_state=42,
    )
    model.fit(X)

    model_path = os.path.join(model_dir, "anomaly_detector.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {"samples": len(X), "contamination": 0.05}

def detect_contextual_anomalies(snapshot_features, model_path):
    """Score a snapshot for multi-dimensional anomalies."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = np.array([snapshot_features])
    score = model.decision_function(X)[0]  # Negative = more anomalous
    is_anomaly = model.predict(X)[0] == -1

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(score, 4),
        "severity": "high" if score < -0.3 else "medium" if score < -0.1 else "low",
    }
```

### Prediction Blending

```python
def blend_predictions(stat_pred, ml_pred, days_of_data):
    """Blend statistical and ML predictions based on data maturity."""
    if days_of_data < 14:
        return stat_pred  # ML not ready yet

    if days_of_data < 60:
        ml_weight = 0.3
    elif days_of_data < 90:
        ml_weight = 0.5
    else:
        ml_weight = 0.7

    stat_weight = 1.0 - ml_weight
    blended = stat_pred * stat_weight + ml_pred * ml_weight

    return round(blended, 1)
```

---

## LLM Meta-Learning Loop (deepseek-r1:8b)

### Weekly Analysis Prompt

```python
META_LEARNING_PROMPT = """You are a data scientist analyzing a home automation prediction system.
Your job is to find patterns in prediction errors and suggest improvements.

## Current System Performance (last 7 days)
{accuracy_data}

## Feature Importance (from current sklearn models)
{feature_importance}

## Current Feature Configuration
{feature_config}

## Available Data Fields (from daily snapshots)
{available_fields}

## Known Correlations
{correlations}

## Previous Suggestions and Outcomes
{previous_suggestions}

## Task
Analyze the prediction accuracy data and suggest specific improvements.

For each suggestion, provide:
1. "action": "enable_feature" | "disable_feature" | "add_interaction" | "adjust_hyperparameter"
2. "target": the specific feature or parameter name
3. "reason": evidence from the accuracy data (cite specific numbers)
4. "expected_impact": which metric should improve and by roughly how much
5. "confidence": "high" | "medium" | "low"

Rules:
- Maximum 3 suggestions per analysis
- Only suggest changes with clear evidence from the data
- Do NOT suggest changes to safety-critical features
- Prefer enabling existing disabled features over creating new ones
- If accuracy is already >85%, focus on the weakest metric only

Output as a JSON array of suggestion objects.
"""
```

### Auto-Apply Guardrail Implementation

```python
def process_meta_learning_suggestions(suggestions, snapshots, current_models, feature_config):
    """Validate and apply meta-learning suggestions."""
    results = []
    applied_count = 0
    MAX_CHANGES_PER_WEEK = 3

    for suggestion in suggestions:
        if applied_count >= MAX_CHANGES_PER_WEEK:
            results.append({"suggestion": suggestion, "applied": False, "reason": "weekly limit reached"})
            continue

        # Create modified feature config
        modified_config = deep_copy(feature_config)
        apply_suggestion_to_config(suggestion, modified_config)

        # Retrain with modified config
        modified_features = build_all_feature_vectors(snapshots, modified_config)
        train_split = modified_features[:int(len(modified_features) * 0.8)]
        val_split = modified_features[int(len(modified_features) * 0.8):]

        current_accuracy = evaluate_models(current_models, val_split)
        modified_model = retrain_with_config(train_split, modified_config)
        modified_accuracy = evaluate_models(modified_model, val_split)

        improvement = modified_accuracy - current_accuracy

        if improvement >= 2.0:  # >=2% improvement required
            # Apply permanently
            save_feature_config(modified_config)
            save_models(modified_model)
            applied_count += 1
            results.append({
                "suggestion": suggestion,
                "applied": True,
                "improvement": round(improvement, 2),
                "new_accuracy": round(modified_accuracy, 2),
            })
        else:
            results.append({
                "suggestion": suggestion,
                "applied": False,
                "reason": f"improvement {improvement:.1f}% < 2% threshold",
                "accuracy_delta": round(improvement, 2),
            })

    return results
```

### Meta-Learning Schedule

```
Sunday 23:45 (after daily pipeline at 23:30):
  1. Load last 7 days of accuracy scores
  2. Load feature importance from current models
  3. Load current feature_config.json
  4. Prompt deepseek-r1:8b for analysis
  5. Parse suggestions from LLM output
  6. Validate each suggestion (retrain + evaluate)
  7. Apply suggestions that pass guardrails
  8. Save weekly report to meta-learning/weekly/
  9. Update applied.json with history
```

---

## Ollama Configuration

### Switch from qwen2.5:7b to deepseek-r1:8b

```python
# In ha-intelligence config section, change:
OLLAMA_MODEL = "deepseek-r1:8b"  # was "qwen2.5:7b"
```

deepseek-r1:8b generates chain-of-thought reasoning before answering, which produces better analytical output for:
- Anomaly interpretation ("why is this unusual?")
- Prediction narratives ("what to expect tomorrow and why")
- Meta-learning analysis ("what patterns explain prediction errors?")

The thinking tokens add ~10-30 seconds of latency but this runs via cron at night — latency is irrelevant.

---

## Cron Schedule (Updated)

```cron
# Intra-day snapshots (lightweight — capture only, no analysis)
0 0,4,8,12,16,20 * * * . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --snapshot-intraday >> /home/justin/.local/log/ha-intelligence.log 2>&1

# Daily: end-of-day snapshot (existing)
0 23 * * * . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --snapshot >> /home/justin/.local/log/ha-intelligence.log 2>&1

# Daily: full pipeline — score yesterday, analyze, predict, report (existing, enhanced)
30 23 * * * . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --full >> /home/justin/.local/log/ha-intelligence.log 2>&1

# Weekly: sklearn retrain + meta-learning (Sunday only)
45 23 * * 0 . /home/justin/.env && /home/justin/.local/bin/ha-intelligence --retrain --meta-learn >> /home/justin/.local/log/ha-intelligence.log 2>&1
```

---

## Implementation Tasks

### Task 1: Install Dependencies

**Files:** None (system setup)

**Steps:**
1. Install scikit-learn and numpy: `pip3 install scikit-learn numpy --break-system-packages`
2. Verify: `python3 -c "from sklearn.ensemble import GradientBoostingRegressor; print('OK')"`

---

### Task 2: Intra-Day Snapshot Command

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Add the `--snapshot-intraday` command that captures a lightweight snapshot at the current hour. Reuse existing extraction functions but save to `intraday/YYYY-MM-DD/HH.json`. Add the new data categories: doors_windows, batteries, network, media, sun, vacuum.

**Test:** Write test for `build_intraday_snapshot()` that verifies all new data categories are present. Test cyclical time encoding (hour_sin/cos should be periodic). Test sun position extraction from `sun.sun` entity.

---

### Task 3: Enhanced Data Extraction Functions

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Add extraction functions for new data categories:
- `extract_doors_windows(snapshot, states)` — binary_sensor with device_class door/window
- `extract_batteries(snapshot, states)` — all entities with battery_level attribute
- `extract_network_summary(snapshot, states)` — device_tracker domain summary
- `extract_media(snapshot, states)` — media_player domain
- `extract_sun(snapshot, states)` — sun.sun entity for sunrise/sunset
- `extract_vacuum(snapshot, states)` — vacuum domain
- `build_time_features(timestamp, sun_data)` — cyclical encoding + solar-relative features

**Test:** Write test for each extraction function with sample entity data. Test cyclical encoding properties: `hour_sin(0) ≈ hour_sin(24)`, `hour_sin(12) ≈ -hour_sin(0)`.

---

### Task 4: Feature Vector Builder

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Implement `build_feature_vector(snapshot, feature_config, prev_snapshot, rolling_stats)`. Reads `feature_config.json` to determine which features are active. Returns a dict of feature_name → float. Add `load_feature_config()` and `save_feature_config()`.

**Test:** Build feature vector from a sample snapshot. Verify feature count matches config. Verify lag features require prev_snapshot. Verify interaction features are computed correctly.

---

### Task 5: End-of-Day Summary Enhancement

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Enhance the daily snapshot with aggregated intra-day data: power/occupancy/lights curves, daily aggregates (mean, max, min, std), derived features (watts_per_person, activity_window), battery drain rates.

**Test:** Given 7 mock intra-day snapshots, verify end-of-day summary computes correct curves and aggregates.

---

### Task 6: sklearn Training Pipeline

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Implement `train_all_models(snapshots, feature_config)`:
- Load all intra-day snapshots
- Build feature vectors for each
- Train GradientBoostingRegressor for each target metric
- Train RandomForestClassifier for device failure
- Train IsolationForest for anomaly detection
- Save all models to `models/` directory
- Save training log with MAE, R2, feature importance

Add `--retrain` CLI command.

**Test:** Generate 50 synthetic snapshots with known patterns (power increases with temp). Train model. Verify R2 > 0.5 on the synthetic data. Verify model files are saved.

---

### Task 7: ML-Enhanced Prediction

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Modify `generate_predictions()` to:
1. Generate statistical prediction (existing)
2. Generate ML prediction (load model, build feature vector, predict)
3. Blend based on `days_of_data` using `blend_predictions()`
4. Include both individual predictions and blended in output

Add contextual anomaly detection: score current snapshot with IsolationForest.

**Test:** Verify blending weights match data maturity. Verify ML prediction is used when model exists. Verify fallback to statistical-only when no model.

---

### Task 8: Device Failure Prediction

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Implement device failure training data builder:
- For each device ever unavailable, create feature vectors from each day
- Label: did this device go unavailable within 7 days of this date?
- Train RandomForest
- At prediction time: score all devices, flag those with >50% failure probability

**Test:** Create synthetic device history where a device has increasing outages. Verify failure probability increases. Test battery drain detection.

---

### Task 9: LLM Meta-Learning Loop

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Implement:
- `run_meta_learning()` — orchestrates the weekly analysis
- `prompt_deepseek_meta_analysis()` — builds prompt, calls Ollama
- `parse_suggestions()` — extract JSON suggestions from LLM response
- `validate_suggestion()` — retrain with modification, compare accuracy
- `apply_suggestion()` — update feature_config.json and retrain

Add `--meta-learn` CLI command.

**Test:** Test suggestion parsing from mock LLM output. Test guardrail: suggestion that doesn't improve accuracy is rejected. Test max 3 changes/week limit.

---

### Task 10: Switch to deepseek-r1:8b

**Files:**
- Modify: `~/.local/bin/ha-intelligence`

Change `OLLAMA_MODEL` from `qwen2.5:7b` to `deepseek-r1:8b`. Test all LLM-dependent functions: `generate_insight_report()`, `generate_brief_line()`, `run_meta_learning()`.

Handle deepseek-r1 output format — it includes `<think>...</think>` blocks before the answer. Strip or preserve thinking tokens based on context (strip for brief, preserve for reports).

**Test:** Manual — run `ha-intelligence --report --dry-run` and verify output quality.

---

### Task 11: Enhanced Accuracy Tracking

**Files:**
- Modify: `~/.local/bin/ha-intelligence`
- Modify: `~/Documents/tests/test_ha_intelligence.py`

Enhance `accuracy.json` to track per-model accuracy:
```json
{
  "scores": [
    {
      "date": "2026-03-01",
      "overall": 78,
      "statistical": {"power_watts": 72, "lights_on": 85, "...": "..."},
      "ml": {"power_watts": 81, "lights_on": 88, "...": "..."},
      "blended": {"power_watts": 78, "lights_on": 87, "...": "..."}
    }
  ],
  "trend": "improving",
  "ml_vs_stat_delta": 6.2,
  "best_metric": "lights_on",
  "worst_metric": "power_watts"
}
```

---

### Task 12: Update Cron Schedule

**Files:**
- Modify: crontab

Add intra-day snapshot cron entries. Add weekly retrain + meta-learn entry. Verify all cron jobs run correctly.

---

### Task 13: Update Claude Code Skills

**Files:**
- Modify: `~/.claude/skills/ha-predict/SKILL.md`
- Modify: `~/.claude/skills/ha-learn/SKILL.md`

Update skills to surface ML predictions, feature importance, device failure alerts, and meta-learning suggestions.

---

### Task 14: Update Design Documentation

**Files:**
- Modify: `~/Documents/docs/plans/2026-02-10-ha-intelligence-engine.md`
- Modify: CLAUDE.md files (memory, project)

Update the original design doc with v2 changes. Update CLAUDE.md memory with new patterns and gotchas discovered during implementation.

---

## Testing Strategy

- **Unit tests:** Each extraction function, feature builder, training pipeline, prediction blending
- **Synthetic data tests:** Generate known patterns, verify models learn them
- **Integration test:** Full pipeline run with mock HA data
- **Manual test:** Run `--snapshot-intraday`, `--retrain`, `--meta-learn --dry-run` against live HA
- **Regression:** Existing 24 tests must continue to pass

---

## What This Does NOT Do

- **Real-time event streaming** — still snapshot-based (7/day), not event-driven
- **Control devices** — read-only, no write actions to HA
- **Cloud training** — all local (Ollama + sklearn on your machine)
- **GPU acceleration** — sklearn uses CPU; sufficient for this data volume
- **Deep learning / neural nets** — GradientBoosting + RandomForest are better for tabular data at this scale
- **Multi-home support** — single HA instance

---

## Success Metrics

- **Day 14:** sklearn models trained, initial predictions with ML blending
- **Day 30:** Blended accuracy ≥65% (better than pure statistical)
- **Day 60:** Blended accuracy ≥75%, meta-learning has applied ≥3 improvements
- **Day 90:** Blended accuracy ≥80%, device failure predictions catching real events
- **Day 180:** Accuracy ≥85%, meta-learning loop demonstrably improving predictions quarter-over-quarter
