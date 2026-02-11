"""ML Engine Module - Adaptive machine learning predictions.

Trains sklearn models based on discovered capabilities and generates predictions
for home automation metrics (power, lights, occupancy, etc.).

Architecture:
- Reads capabilities from hub cache to determine what to predict
- Trains separate models per capability (GradientBoosting, RandomForest, blend)
- Stores trained models and predictions back to hub cache
- Runs training on schedule (weekly) and prediction daily
"""

import os
import json
import logging
import pickle
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

from hub.core import Module, IntelligenceHub


logger = logging.getLogger(__name__)


class MLEngine(Module):
    """Machine learning prediction engine with adaptive capability mapping."""

    def __init__(
        self,
        hub: IntelligenceHub,
        models_dir: str,
        training_data_dir: str
    ):
        """Initialize ML engine.

        Args:
            hub: IntelligenceHub instance
            models_dir: Directory to store trained models
            training_data_dir: Directory with historical snapshots for training
        """
        super().__init__("ml_engine", hub)
        self.models_dir = Path(models_dir)
        self.training_data_dir = Path(training_data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Capability to prediction mapping
        # Maps discovered capabilities to what we should predict
        self.capability_predictions = {
            "power_monitoring": ["power_watts"],
            "lighting": ["lights_on", "total_brightness"],
            "occupancy": ["people_home", "devices_home"],
            "motion": ["motion_active_count"],
            "climate": ["temperature", "humidity"],
        }

        # Loaded models cache
        self.models: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize module - load existing models."""
        self.logger.info("ML Engine initializing...")

        # Load capabilities from hub cache
        capabilities_entry = await self.hub.get_cache("capabilities")
        if not capabilities_entry:
            self.logger.warning("No capabilities found in cache. Run discovery first.")
            return

        capabilities = capabilities_entry.get("data", {})
        self.logger.info(f"Found {len(capabilities)} capabilities in cache")

        # Load existing models
        await self._load_models()

        self.logger.info("ML Engine initialized")

    async def _load_models(self):
        """Load trained models from disk."""
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)

                model_name = model_file.stem
                self.models[model_name] = model_data
                self.logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                self.logger.error(f"Failed to load model {model_file}: {e}")

    async def train_models(self, days_history: int = 60):
        """Train models using historical data.

        Args:
            days_history: Number of days of historical data to use for training
        """
        self.logger.info(f"Training models with {days_history} days of history...")

        # Get capabilities to determine what to train
        capabilities_entry = await self.hub.get_cache("capabilities")
        if not capabilities_entry:
            self.logger.error("No capabilities in cache. Cannot train without discovery data.")
            return

        capabilities = capabilities_entry.get("data", {})

        # Load training data
        training_data = await self._load_training_data(days_history)
        if not training_data:
            self.logger.error("No training data available")
            return

        self.logger.info(f"Loaded {len(training_data)} snapshots for training")

        # Train models for each available capability
        for capability_name, capability_data in capabilities.items():
            if not capability_data.get("available"):
                continue

            # Check if we have predictions defined for this capability
            prediction_targets = self.capability_predictions.get(capability_name)
            if not prediction_targets:
                self.logger.debug(f"No prediction targets defined for {capability_name}")
                continue

            self.logger.info(f"Training models for capability: {capability_name}")

            for target in prediction_targets:
                try:
                    await self._train_model_for_target(
                        target,
                        training_data,
                        capability_name
                    )
                except Exception as e:
                    self.logger.error(f"Failed to train model for {target}: {e}")

        self.logger.info("Model training complete")

        # Store training metadata in cache
        await self.hub.set_cache(
            "ml_training_metadata",
            {
                "last_trained": datetime.now().isoformat(),
                "days_history": days_history,
                "num_snapshots": len(training_data),
                "capabilities_trained": list(capabilities.keys())
            }
        )

    async def _load_training_data(self, days: int) -> List[Dict[str, Any]]:
        """Load historical snapshots for training.

        Args:
            days: Number of days to load

        Returns:
            List of snapshot dictionaries
        """
        snapshots = []
        today = datetime.now()

        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            snapshot_file = self.training_data_dir / f"{date_str}.json"

            if snapshot_file.exists():
                try:
                    with open(snapshot_file) as f:
                        snapshot = json.load(f)
                        snapshots.append(snapshot)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")

        return snapshots

    async def _train_model_for_target(
        self,
        target: str,
        training_data: List[Dict[str, Any]],
        capability_name: str
    ):
        """Train a model for a specific prediction target.

        Args:
            target: Target metric to predict (e.g., "power_watts")
            training_data: List of historical snapshots
            capability_name: Capability this target belongs to
        """
        self.logger.info(f"Training model for target: {target}")

        # Extract features and target values
        X, y = self._build_training_dataset(training_data, target)

        if len(X) < 10:
            self.logger.warning(f"Insufficient training data for {target}: {len(X)} samples")
            return

        # Train GradientBoosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        gb_model.fit(X, y)

        # Train RandomForest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X, y)

        # Create scaler for feature normalization
        scaler = StandardScaler()
        scaler.fit(X)

        # Store model data
        config = self._get_feature_config()
        model_data = {
            "target": target,
            "capability": capability_name,
            "gb_model": gb_model,
            "rf_model": rf_model,
            "scaler": scaler,
            "trained_at": datetime.now().isoformat(),
            "num_samples": len(X),
            "feature_names": self._get_feature_names(config)
        }

        # Save to disk
        model_file = self.models_dir / f"{target}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        # Cache in memory
        self.models[target] = model_data

        self.logger.info(
            f"Model trained for {target}: "
            f"{len(X)} samples, {len(model_data['feature_names'])} features"
        )

    def _build_training_dataset(
        self,
        snapshots: List[Dict[str, Any]],
        target: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build training dataset from snapshots.

        Args:
            snapshots: List of historical snapshots
            target: Target metric to extract

        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        X_list = []
        y_list = []
        config = self._get_feature_config()

        for i, snapshot in enumerate(snapshots):
            # Get previous snapshot and rolling stats for lag features
            prev_snapshot = snapshots[i - 1] if i > 0 else None

            # Compute rolling stats for last 7 snapshots
            rolling_stats = {}
            if i >= 7:
                recent = snapshots[max(0, i - 7):i]
                rolling_stats["power_mean_7d"] = sum(
                    s.get("power", {}).get("total_watts", 0) for s in recent
                ) / len(recent)
                rolling_stats["lights_mean_7d"] = sum(
                    s.get("lights", {}).get("on", 0) for s in recent
                ) / len(recent)

            # Extract features
            features = self._extract_features(
                snapshot,
                config=config,
                prev_snapshot=prev_snapshot,
                rolling_stats=rolling_stats
            )
            if features is None:
                continue

            # Extract target value
            target_value = self._extract_target(snapshot, target)
            if target_value is None:
                continue

            X_list.append(list(features.values()))
            y_list.append(target_value)

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list), np.array(y_list)

    def _get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration from cache or return default.

        Returns:
            Feature configuration dictionary
        """
        # Default feature config matching ha-intelligence
        DEFAULT_FEATURE_CONFIG = {
            "version": 1,
            "last_modified": "",
            "modified_by": "ml_engine",
            "time_features": {
                "hour_sin_cos": True,
                "dow_sin_cos": True,
                "month_sin_cos": True,
                "day_of_year_sin_cos": True,
                "is_weekend": True,
                "is_holiday": True,
                "is_night": True,
                "is_work_hours": True,
                "minutes_since_sunrise": True,
                "minutes_until_sunset": True,
                "daylight_remaining_pct": True,
            },
            "weather_features": {
                "temp_f": True,
                "humidity_pct": True,
                "wind_mph": True,
            },
            "home_features": {
                "people_home_count": True,
                "device_count_home": True,
                "lights_on": True,
                "total_brightness": True,
                "motion_active_count": True,
                "active_media_players": True,
                "ev_battery_pct": True,
                "ev_is_charging": True,
            },
            "lag_features": {
                "prev_snapshot_power": True,
                "prev_snapshot_lights": True,
                "prev_snapshot_occupancy": True,
                "rolling_7d_power_mean": True,
                "rolling_7d_lights_mean": True,
            },
            "interaction_features": {
                "is_weekend_x_temp": False,
                "people_home_x_hour_sin": False,
                "daylight_x_lights": False,
            },
            "target_metrics": [
                "power_watts",
                "lights_on",
                "devices_home",
                "unavailable",
                "useful_events",
            ],
        }

        # TODO: Load from hub cache "feature_config" category with versioning
        # For now, return default
        return DEFAULT_FEATURE_CONFIG.copy()

    def _get_feature_names(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Return ordered list of feature names from config.

        Args:
            config: Feature configuration (uses default if None)

        Returns:
            List of feature names in order
        """
        if config is None:
            config = self._get_feature_config()

        names = []

        # Time features
        tc = config.get("time_features", {})
        if tc.get("hour_sin_cos"):
            names.extend(["hour_sin", "hour_cos"])
        if tc.get("dow_sin_cos"):
            names.extend(["dow_sin", "dow_cos"])
        if tc.get("month_sin_cos"):
            names.extend(["month_sin", "month_cos"])
        if tc.get("day_of_year_sin_cos"):
            names.extend(["day_of_year_sin", "day_of_year_cos"])
        for simple in ["is_weekend", "is_holiday", "is_night", "is_work_hours",
                       "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct"]:
            if tc.get(simple):
                names.append(simple)

        # Weather features
        for key in config.get("weather_features", {}):
            if config["weather_features"][key]:
                names.append(f"weather_{key}")

        # Home state features
        for key in config.get("home_features", {}):
            if config["home_features"][key]:
                names.append(key)

        # Lag features
        for key in config.get("lag_features", {}):
            if config["lag_features"][key]:
                names.append(key)

        # Interaction features
        for key in config.get("interaction_features", {}):
            if config["interaction_features"][key]:
                names.append(key)

        return names

    def _compute_time_features(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        """Compute time features from snapshot if not present.

        Args:
            snapshot: Snapshot dictionary (must have 'date' field)

        Returns:
            Dictionary of time features
        """
        # If snapshot already has time_features, return them
        if "time_features" in snapshot:
            return snapshot["time_features"]

        # Compute time features from date field
        date_str = snapshot.get("date")
        if not date_str:
            # Return zeros for all features if no date
            return {
                "hour_sin": 0, "hour_cos": 0,
                "dow_sin": 0, "dow_cos": 0,
                "month_sin": 0, "month_cos": 0,
                "day_of_year_sin": 0, "day_of_year_cos": 0,
                "is_weekend": 0, "is_holiday": 0,
                "is_night": 0, "is_work_hours": 0,
                "minutes_since_sunrise": 0,
                "minutes_until_sunset": 0,
                "daylight_remaining_pct": 0,
            }

        # Parse date (daily snapshots are at midnight)
        dt = datetime.fromisoformat(date_str) if "T" in date_str else datetime.strptime(date_str, "%Y-%m-%d")

        # Use noon for daily snapshots (more representative than midnight)
        if "T" not in date_str:
            dt = dt.replace(hour=12, minute=0, second=0)

        # Cyclic encoding helper
        def sin_cos_encode(value, period):
            angle = 2 * math.pi * value / period
            return round(math.sin(angle), 6), round(math.cos(angle), 6)

        # Hour features (24-hour cycle)
        h_sin, h_cos = sin_cos_encode(dt.hour, 24)

        # Day of week features (7-day cycle, 0=Monday)
        dow = dt.weekday()
        d_sin, d_cos = sin_cos_encode(dow, 7)

        # Month features (12-month cycle)
        month = dt.month
        m_sin, m_cos = sin_cos_encode(month - 1, 12)

        # Day of year features (365-day cycle)
        day_of_year = dt.timetuple().tm_yday
        doy_sin, doy_cos = sin_cos_encode(day_of_year - 1, 365)

        # Boolean features
        is_weekend = 1 if dow >= 5 else 0
        is_holiday = 1 if snapshot.get("is_holiday") else 0
        is_night = 1 if dt.hour < 6 or dt.hour >= 22 else 0
        is_work_hours = 1 if 9 <= dt.hour < 17 and dow < 5 else 0

        # Sun features (approximations - daily snapshots don't have precise sun data)
        # Assume sunrise ~6:30am (390 min), sunset ~6:30pm (1110 min)
        minutes_since_midnight = dt.hour * 60 + dt.minute
        minutes_since_sunrise = max(0, minutes_since_midnight - 390)
        minutes_until_sunset = max(0, 1110 - minutes_since_midnight)
        daylight_minutes = 1110 - 390
        daylight_remaining_pct = min(100, max(0, minutes_until_sunset / daylight_minutes * 100))

        return {
            "hour": dt.hour,
            "hour_sin": h_sin,
            "hour_cos": h_cos,
            "dow": dow,
            "dow_sin": d_sin,
            "dow_cos": d_cos,
            "month": month,
            "month_sin": m_sin,
            "month_cos": m_cos,
            "day_of_year": day_of_year,
            "day_of_year_sin": doy_sin,
            "day_of_year_cos": doy_cos,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "is_night": is_night,
            "is_work_hours": is_work_hours,
            "minutes_since_sunrise": minutes_since_sunrise,
            "minutes_until_sunset": minutes_until_sunset,
            "daylight_remaining_pct": daylight_remaining_pct,
        }

    def _extract_features(
        self,
        snapshot: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        prev_snapshot: Optional[Dict[str, Any]] = None,
        rolling_stats: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, float]]:
        """Extract feature vector from snapshot using feature config.

        Args:
            snapshot: Snapshot dictionary
            config: Feature configuration (uses default if None)
            prev_snapshot: Previous snapshot for lag features (optional)
            rolling_stats: Rolling statistics dict (optional)

        Returns:
            Dictionary of feature_name -> float value
        """
        if config is None:
            config = self._get_feature_config()

        features = {}

        # Compute or retrieve time features
        tf = self._compute_time_features(snapshot)
        tc = config.get("time_features", {})

        # Time features - sin/cos pairs for cyclic encoding
        if tc.get("hour_sin_cos"):
            features["hour_sin"] = tf.get("hour_sin", 0)
            features["hour_cos"] = tf.get("hour_cos", 0)
        if tc.get("dow_sin_cos"):
            features["dow_sin"] = tf.get("dow_sin", 0)
            features["dow_cos"] = tf.get("dow_cos", 0)
        if tc.get("month_sin_cos"):
            features["month_sin"] = tf.get("month_sin", 0)
            features["month_cos"] = tf.get("month_cos", 0)
        if tc.get("day_of_year_sin_cos"):
            features["day_of_year_sin"] = tf.get("day_of_year_sin", 0)
            features["day_of_year_cos"] = tf.get("day_of_year_cos", 0)

        # Time features - simple boolean/numeric
        for simple in ["is_weekend", "is_holiday", "is_night", "is_work_hours",
                       "minutes_since_sunrise", "minutes_until_sunset", "daylight_remaining_pct"]:
            if tc.get(simple):
                val = tf.get(simple, 0)
                features[simple] = 1 if val is True else (0 if val is False else (val or 0))

        # Weather features
        weather = snapshot.get("weather", {})
        for key, enabled in config.get("weather_features", {}).items():
            if enabled:
                features[f"weather_{key}"] = weather.get(key) or 0

        # Home state features
        hc = config.get("home_features", {})
        if hc.get("people_home_count"):
            features["people_home_count"] = snapshot.get("occupancy", {}).get(
                "people_home_count",
                len(snapshot.get("occupancy", {}).get("people_home", []))
            )
        if hc.get("device_count_home"):
            features["device_count_home"] = snapshot.get("occupancy", {}).get("device_count_home", 0)
        if hc.get("lights_on"):
            features["lights_on"] = snapshot.get("lights", {}).get("on", 0)
        if hc.get("total_brightness"):
            features["total_brightness"] = snapshot.get("lights", {}).get("total_brightness", 0)
        if hc.get("motion_active_count"):
            features["motion_active_count"] = snapshot.get("motion", {}).get("active_count", 0)
        if hc.get("active_media_players"):
            features["active_media_players"] = snapshot.get("media", {}).get("total_active", 0)
        if hc.get("ev_battery_pct"):
            features["ev_battery_pct"] = snapshot.get("ev", {}).get("TARS", {}).get("battery_pct", 0)
        if hc.get("ev_is_charging"):
            features["ev_is_charging"] = 1 if snapshot.get("ev", {}).get("TARS", {}).get("is_charging") else 0

        # Lag features - previous snapshot and rolling stats
        lc = config.get("lag_features", {})
        if lc.get("prev_snapshot_power") and prev_snapshot:
            features["prev_snapshot_power"] = prev_snapshot.get("power", {}).get("total_watts", 0)
        elif lc.get("prev_snapshot_power"):
            features["prev_snapshot_power"] = 0
        if lc.get("prev_snapshot_lights") and prev_snapshot:
            features["prev_snapshot_lights"] = prev_snapshot.get("lights", {}).get("on", 0)
        elif lc.get("prev_snapshot_lights"):
            features["prev_snapshot_lights"] = 0
        if lc.get("prev_snapshot_occupancy") and prev_snapshot:
            features["prev_snapshot_occupancy"] = prev_snapshot.get("occupancy", {}).get("device_count_home", 0)
        elif lc.get("prev_snapshot_occupancy"):
            features["prev_snapshot_occupancy"] = 0
        if lc.get("rolling_7d_power_mean"):
            features["rolling_7d_power_mean"] = (rolling_stats or {}).get("power_mean_7d", 0)
        if lc.get("rolling_7d_lights_mean"):
            features["rolling_7d_lights_mean"] = (rolling_stats or {}).get("lights_mean_7d", 0)

        # Interaction features - derived from other features
        ic = config.get("interaction_features", {})
        if ic.get("is_weekend_x_temp"):
            features["is_weekend_x_temp"] = features.get("is_weekend", 0) * features.get("weather_temp_f", 0)
        if ic.get("people_home_x_hour_sin"):
            features["people_home_x_hour_sin"] = features.get("people_home_count", 0) * features.get("hour_sin", 0)
        if ic.get("daylight_x_lights"):
            features["daylight_x_lights"] = features.get("daylight_remaining_pct", 0) * features.get("lights_on", 0)

        return features

    def _extract_target(self, snapshot: Dict[str, Any], target: str) -> Optional[float]:
        """Extract target value from snapshot.

        Args:
            snapshot: Snapshot dictionary
            target: Target metric name

        Returns:
            Target value or None if not available
        """
        # Map target names to snapshot locations
        target_map = {
            "power_watts": ("power", "total_watts"),
            "lights_on": ("lights", "on"),
            "total_brightness": ("lights", "total_brightness"),
            "people_home": ("occupancy", "people_home_count"),
            "devices_home": ("occupancy", "device_count_home"),
            "motion_active_count": ("motion", "active_count"),
        }

        if target not in target_map:
            return None

        section, key = target_map[target]
        value = snapshot.get(section, {}).get(key)

        return float(value) if value is not None else None

    async def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for tomorrow using trained models.

        Returns:
            Dictionary of predictions by target
        """
        self.logger.info("Generating predictions...")

        if not self.models:
            self.logger.warning("No models loaded. Train models first.")
            return {}

        # Get current state from cache to use as features
        # In real implementation, this would fetch the latest snapshot
        # For now, return placeholder
        predictions = {
            "timestamp": datetime.now().isoformat(),
            "predictions": {},
            "model_count": len(self.models)
        }

        self.logger.info(f"Generated predictions for {len(self.models)} targets")

        # Store predictions in cache
        await self.hub.set_cache("ml_predictions", predictions)

        return predictions

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        """Handle hub events.

        Args:
            event_type: Type of event
            data: Event data
        """
        # ML module could respond to cache updates
        # e.g., when new discovery data available, retrain models
        if event_type == "cache_updated" and data.get("category") == "capabilities":
            self.logger.info("Capabilities updated - models may need retraining")

    async def schedule_periodic_training(self, interval_days: int = 7):
        """Schedule periodic model retraining.

        Args:
            interval_days: Days between training runs
        """
        async def training_task():
            try:
                await self.train_models(days_history=60)
            except Exception as e:
                self.logger.error(f"Scheduled training failed: {e}")

        await self.hub.schedule_task(
            task_id="ml_training_periodic",
            coro=training_task,
            interval=timedelta(days=interval_days),
            run_immediately=False
        )

        self.logger.info(f"Scheduled periodic training every {interval_days} days")
