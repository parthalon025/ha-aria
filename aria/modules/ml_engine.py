"""ML Engine Module - Adaptive machine learning predictions.

Trains ML models based on discovered capabilities and generates predictions
for home automation metrics (power, lights, occupancy, etc.).

Architecture:
- Reads capabilities from hub cache to determine what to predict
- Trains separate models per capability (GradientBoosting, RandomForest, LightGBM, blend)
- Stores trained models and predictions back to hub cache
- Runs training on schedule (weekly) and prediction daily
- Model selection is configurable: any subset of {gb, rf, lgbm} can be enabled
"""

import json
import logging
import pickle
import math
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb

# Suppress sklearn warning about feature names when using numpy arrays.
# Our feature pipeline guarantees alignment between training and prediction —
# the same _extract_features() dict order is used for both paths.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG as _ENGINE_FEATURE_CONFIG
from aria.hub.core import Module, IntelligenceHub


logger = logging.getLogger(__name__)

# Feature engineering constants — will move to config store in Phase 2
DECAY_HALF_LIFE_DAYS = 7
WEEKDAY_ALIGNMENT_BONUS = 1.5
ROLLING_WINDOWS_HOURS = [1, 3, 6]


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

        # Model configuration — which model types to train and their blend weights.
        # Keys: "gb" (GradientBoosting), "rf" (RandomForest), "lgbm" (LightGBM)
        # Weights are normalized at prediction time so they always sum to 1.0.
        self.enabled_models: Dict[str, bool] = {
            "gb": True,
            "rf": True,
            "lgbm": True,
        }
        self.model_weights: Dict[str, float] = {
            "gb": 0.35,
            "rf": 0.25,
            "lgbm": 0.40,
        }

        # Loaded models cache
        self.models: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """Initialize module - load existing models."""
        self.logger.info("ML Engine initializing...")

        # Load capabilities from hub cache (warn if stale)
        capabilities_entry = await self.hub.get_cache_fresh(
            "capabilities", timedelta(hours=48), caller="ml_engine.init"
        )
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

        # Get capabilities to determine what to train (warn if stale)
        capabilities_entry = await self.hub.get_cache_fresh(
            "capabilities", timedelta(hours=48), caller="ml_engine.train"
        )
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

        # Train global anomaly detector on all features
        await self._train_anomaly_detector(training_data)

        self.logger.info("Model training complete")

        # Collect training summary from all trained models
        trained_targets = []
        accuracy_summary = {}
        for target, model_data in self.models.items():
            if target == "anomaly_detector":
                continue
            if "accuracy_scores" in model_data:
                trained_targets.append(target)
                accuracy_summary[target] = model_data["accuracy_scores"]

        # Store training metadata in cache
        await self.hub.set_cache(
            "ml_training_metadata",
            {
                "last_trained": datetime.now().isoformat(),
                "days_history": days_history,
                "num_snapshots": len(training_data),
                "capabilities_trained": list(capabilities.keys()),
                "targets_trained": trained_targets,
                "accuracy_summary": accuracy_summary,
                "has_anomaly_detector": "anomaly_detector" in self.models
            }
        )

        # Store feature configuration for reuse across restarts
        config = await self._get_feature_config()
        config["last_modified"] = datetime.now().isoformat()
        config["modified_by"] = "ml_engine"
        await self.hub.set_cache("feature_config", config)

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

        # Extract features, target values, and decay-based sample weights
        X, y, sample_weights = await self._build_training_dataset(training_data, target)

        if len(X) < 14:
            self.logger.warning(f"Insufficient training data for {target}: {len(X)} samples (need 14+)")
            return

        # Sort snapshots chronologically for proper train/validation split
        # (already chronological from _load_training_data, but ensure it)

        # 80/20 chronological split (no shuffle - time series data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train = sample_weights[:split_idx] if len(sample_weights) > 0 else None

        # Train GradientBoosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=max(3, len(X_train) // 20),
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(X_train, y_train, sample_weight=w_train)

        # Train RandomForest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train, sample_weight=w_train)

        # Train LightGBM model (always trained even if disabled in enabled_models,
        # so toggling a model on doesn't require a full retrain cycle)
        lgbm_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8,
            random_state=42,
            verbosity=-1,  # Suppress LightGBM info logs
            importance_type='gain',  # Gain-based importance (reduction in loss)
        )
        lgbm_model.fit(X_train, y_train, sample_weight=w_train)

        # Train IsolationForest for anomaly detection
        iso_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        iso_model.fit(X_train)

        # Compute validation metrics
        gb_pred = gb_model.predict(X_val)
        rf_pred = rf_model.predict(X_val)
        lgbm_pred = lgbm_model.predict(X_val)

        gb_mae = mean_absolute_error(y_val, gb_pred)
        gb_r2 = r2_score(y_val, gb_pred) if len(y_val) > 1 else 0.0

        rf_mae = mean_absolute_error(y_val, rf_pred)
        rf_r2 = r2_score(y_val, rf_pred) if len(y_val) > 1 else 0.0

        lgbm_mae = mean_absolute_error(y_val, lgbm_pred)
        lgbm_r2 = r2_score(y_val, lgbm_pred) if len(y_val) > 1 else 0.0

        # Extract feature importance from RandomForest
        config = await self._get_feature_config()
        feature_names = await self._get_feature_names(config)
        feature_importance = {
            name: round(float(importance), 4)
            for name, importance in zip(feature_names, rf_model.feature_importances_)
        }

        # Extract LightGBM feature importance (gain-based)
        lgbm_feature_importance = {
            name: round(float(importance), 4)
            for name, importance in zip(feature_names, lgbm_model.feature_importances_)
        }

        # Create scaler for feature normalization
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Store model data with complete metadata
        model_data = {
            "target": target,
            "capability": capability_name,
            "gb_model": gb_model,
            "rf_model": rf_model,
            "lgbm_model": lgbm_model,
            "iso_model": iso_model,
            "scaler": scaler,
            "trained_at": datetime.now().isoformat(),
            "num_samples": len(X),
            "num_train": len(X_train),
            "num_val": len(X_val),
            "feature_names": feature_names,
            "feature_importance": feature_importance,
            "lgbm_feature_importance": lgbm_feature_importance,
            "accuracy_scores": {
                "gb_mae": round(gb_mae, 3),
                "gb_r2": round(gb_r2, 3),
                "rf_mae": round(rf_mae, 3),
                "rf_r2": round(rf_r2, 3),
                "lgbm_mae": round(lgbm_mae, 3),
                "lgbm_r2": round(lgbm_r2, 3),
            }
        }

        # Save to disk
        model_file = self.models_dir / f"{target}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        # Cache in memory
        self.models[target] = model_data

        self.logger.info(
            f"Model trained for {target}: "
            f"{len(X)} samples ({len(X_train)} train, {len(X_val)} val), "
            f"{len(feature_names)} features, "
            f"GB MAE={gb_mae:.2f} R²={gb_r2:.3f}, "
            f"RF MAE={rf_mae:.2f} R²={rf_r2:.3f}, "
            f"LGBM MAE={lgbm_mae:.2f} R²={lgbm_r2:.3f}"
        )

    async def _train_anomaly_detector(self, training_data: List[Dict[str, Any]]):
        """Train global anomaly detector on all features.

        Args:
            training_data: List of historical snapshots
        """
        self.logger.info("Training anomaly detector...")

        if len(training_data) < 14:
            self.logger.warning(f"Insufficient data for anomaly detector ({len(training_data)} < 14)")
            return

        # Build feature matrix from all snapshots
        config = await self._get_feature_config()
        feature_names = await self._get_feature_names(config)
        X_list = []

        for i, snapshot in enumerate(training_data):
            prev_snapshot = training_data[i - 1] if i > 0 else None

            # Compute rolling stats
            rolling_stats = {}
            if i >= 7:
                recent = training_data[max(0, i - 7):i]
                rolling_stats["power_mean_7d"] = sum(
                    s.get("power", {}).get("total_watts", 0) for s in recent
                ) / len(recent)
                rolling_stats["lights_mean_7d"] = sum(
                    s.get("lights", {}).get("on", 0) for s in recent
                ) / len(recent)

            features = await self._extract_features(
                snapshot,
                config=config,
                prev_snapshot=prev_snapshot,
                rolling_stats=rolling_stats
            )

            if features:
                X_list.append([features.get(name, 0) for name in feature_names])

        if len(X_list) < 14:
            self.logger.warning(f"Insufficient feature vectors for anomaly detector ({len(X_list)} < 14)")
            return

        X = np.array(X_list, dtype=float)

        # Train IsolationForest
        model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Assume 5% of training data is anomalous
            random_state=42
        )
        model.fit(X)

        # Save anomaly detector
        model_data = {
            "model": model,
            "trained_at": datetime.now().isoformat(),
            "num_samples": len(X),
            "contamination": 0.05
        }

        model_file = self.models_dir / "anomaly_detector.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        # Cache in memory
        self.models["anomaly_detector"] = model_data

        self.logger.info(
            f"Anomaly detector trained: {len(X)} samples, contamination=0.05"
        )

    async def _build_training_dataset(
        self,
        snapshots: List[Dict[str, Any]],
        target: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build training dataset from snapshots.

        Args:
            snapshots: List of historical snapshots
            target: Target metric to extract

        Returns:
            Tuple of (features, targets, sample_weights) as numpy arrays.
            Sample weights are decay-based: recent data and same-weekday
            data receive higher weight.
        """
        X_list = []
        y_list = []
        included_snapshots = []
        config = await self._get_feature_config()
        feature_names = await self._get_feature_names(config)

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
            features = await self._extract_features(
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

            X_list.append([features.get(name, 0) for name in feature_names])
            y_list.append(target_value)
            included_snapshots.append(snapshot)

        if not X_list:
            return np.array([]), np.array([]), np.array([])

        # Compute decay-based sample weights
        sample_weights = self._compute_decay_weights(included_snapshots)

        return np.array(X_list), np.array(y_list), sample_weights

    async def _get_feature_config(self) -> Dict[str, Any]:
        """Get feature configuration from cache or return default.

        Returns:
            Feature configuration dictionary
        """
        # Use canonical engine config as default — hub extends with rolling
        # window features in _get_feature_names(), not in the config dict.
        import copy
        default = copy.deepcopy(_ENGINE_FEATURE_CONFIG)
        default["modified_by"] = "ml_engine"

        # Load from hub cache with fallback to default
        config_entry = await self.hub.get_cache("feature_config")
        if config_entry:
            self.logger.debug("Loaded feature config from cache")
            return config_entry.get("data", default)

        self.logger.debug("Using default feature config (no cache found)")
        return default

    async def _get_feature_names(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Return ordered list of feature names from config.

        Args:
            config: Feature configuration (uses default if None)

        Returns:
            List of feature names in order
        """
        if config is None:
            config = await self._get_feature_config()

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

        # Rolling window features (always included)
        for hours in ROLLING_WINDOWS_HOURS:
            names.append(f"rolling_{hours}h_event_count")
            names.append(f"rolling_{hours}h_domain_entropy")
            names.append(f"rolling_{hours}h_dominant_domain_pct")
            names.append(f"rolling_{hours}h_trend")

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
        is_work_hours = 1 if 8 <= dt.hour < 17 and dow < 5 else 0

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

    def _compute_decay_weights(
        self,
        snapshots: List[Dict[str, Any]],
        reference_date: Optional[datetime] = None
    ) -> np.ndarray:
        """Compute decay-based sample weights for training data.

        Recent snapshots get higher weight. Same-weekday snapshots get a bonus.

        Args:
            snapshots: List of snapshots (must have 'date' field)
            reference_date: Reference date for computing age (defaults to now)

        Returns:
            Array of weights, one per snapshot
        """
        if reference_date is None:
            reference_date = datetime.now()

        ref_weekday = reference_date.weekday()
        weights = []

        for snapshot in snapshots:
            date_str = snapshot.get("date", "")
            if not date_str:
                weights.append(0.0)
                continue

            try:
                if "T" in date_str:
                    snap_dt = datetime.fromisoformat(date_str)
                else:
                    snap_dt = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                weights.append(0.0)
                continue

            days_ago = max(0, (reference_date - snap_dt).total_seconds() / 86400)
            recency_decay = math.exp(-days_ago / DECAY_HALF_LIFE_DAYS)

            same_weekday = snap_dt.weekday() == ref_weekday
            weekday_bonus = WEEKDAY_ALIGNMENT_BONUS if same_weekday else 1.0

            weights.append(recency_decay * weekday_bonus)

        return np.array(weights, dtype=float)

    async def _compute_rolling_window_stats(
        self,
        activity_log: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Compute rolling window statistics from activity log.

        For each window size (1h, 3h, 6h), computes:
        - Event count per domain
        - Domain entropy (spread of activity across domains)
        - Dominant domain (most active domain)
        - Activity trend (increasing/decreasing/stable)

        Args:
            activity_log: Activity log data (from cache). If None, attempts
                to read from hub cache.

        Returns:
            Dictionary of rolling window features
        """
        if activity_log is None:
            cache_entry = await self.hub.get_cache("activity_log")
            if cache_entry and cache_entry.get("data"):
                activity_log = cache_entry["data"]

        stats: Dict[str, float] = {}

        if not activity_log:
            # Return zeros for all rolling window features
            for hours in ROLLING_WINDOWS_HOURS:
                stats[f"rolling_{hours}h_event_count"] = 0
                stats[f"rolling_{hours}h_domain_entropy"] = 0
                stats[f"rolling_{hours}h_dominant_domain_pct"] = 0
                stats[f"rolling_{hours}h_trend"] = 0  # 0=stable
            return stats

        windows = activity_log.get("windows", [])
        if not windows:
            for hours in ROLLING_WINDOWS_HOURS:
                stats[f"rolling_{hours}h_event_count"] = 0
                stats[f"rolling_{hours}h_domain_entropy"] = 0
                stats[f"rolling_{hours}h_dominant_domain_pct"] = 0
                stats[f"rolling_{hours}h_trend"] = 0
            return stats

        now = datetime.now()

        for hours in ROLLING_WINDOWS_HOURS:
            cutoff = (now - timedelta(hours=hours)).isoformat()

            # Filter windows within this time range
            # String comparison works because activity_monitor generates ISO 8601
            # timestamps in consistent local timezone format (lexicographic order)
            relevant = [w for w in windows if w.get("window_start", "") >= cutoff]

            if not relevant:
                stats[f"rolling_{hours}h_event_count"] = 0
                stats[f"rolling_{hours}h_domain_entropy"] = 0
                stats[f"rolling_{hours}h_dominant_domain_pct"] = 0
                stats[f"rolling_{hours}h_trend"] = 0
                continue

            # Aggregate domain counts across windows
            total_events = 0
            domain_counts: Dict[str, int] = {}
            for w in relevant:
                total_events += w.get("event_count", 0)
                for domain, count in w.get("by_domain", {}).items():
                    domain_counts[domain] = domain_counts.get(domain, 0) + count

            stats[f"rolling_{hours}h_event_count"] = total_events

            # Domain entropy: -sum(p * log2(p)) for each domain
            if total_events > 0 and domain_counts:
                entropy = 0.0
                for count in domain_counts.values():
                    p = count / total_events
                    if p > 0:
                        entropy -= p * math.log2(p)
                stats[f"rolling_{hours}h_domain_entropy"] = round(entropy, 4)

                # Dominant domain percentage
                max_count = max(domain_counts.values())
                stats[f"rolling_{hours}h_dominant_domain_pct"] = round(
                    max_count / total_events, 4
                )
            else:
                stats[f"rolling_{hours}h_domain_entropy"] = 0
                stats[f"rolling_{hours}h_dominant_domain_pct"] = 0

            # Activity trend: compare first half vs second half of window
            if len(relevant) >= 2:
                mid = len(relevant) // 2
                first_half = relevant[:mid]
                second_half = relevant[mid:]
                first_count = sum(w.get("event_count", 0) for w in first_half)
                second_count = sum(w.get("event_count", 0) for w in second_half)

                if first_count == 0 and second_count == 0:
                    trend = 0.0  # stable
                elif first_count == 0:
                    trend = 1.0  # increasing
                else:
                    ratio = second_count / first_count
                    if ratio > 1.2:
                        trend = 1.0   # increasing
                    elif ratio < 0.8:
                        trend = -1.0  # decreasing
                    else:
                        trend = 0.0   # stable
                stats[f"rolling_{hours}h_trend"] = trend
            else:
                stats[f"rolling_{hours}h_trend"] = 0.0

        return stats

    async def _extract_features(
        self,
        snapshot: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        prev_snapshot: Optional[Dict[str, Any]] = None,
        rolling_stats: Optional[Dict[str, float]] = None,
        rolling_window_stats: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, float]]:
        """Extract feature vector from snapshot using feature config.

        Args:
            snapshot: Snapshot dictionary
            config: Feature configuration (uses default if None)
            prev_snapshot: Previous snapshot for lag features (optional)
            rolling_stats: Rolling statistics dict (optional)
            rolling_window_stats: Rolling window stats from activity log (optional)

        Returns:
            Dictionary of feature_name -> float value
        """
        if config is None:
            config = await self._get_feature_config()

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

        # Rolling window features (from activity log)
        rws = rolling_window_stats or {}
        for hours in ROLLING_WINDOWS_HOURS:
            features[f"rolling_{hours}h_event_count"] = rws.get(f"rolling_{hours}h_event_count", 0)
            features[f"rolling_{hours}h_domain_entropy"] = rws.get(f"rolling_{hours}h_domain_entropy", 0)
            features[f"rolling_{hours}h_dominant_domain_pct"] = rws.get(f"rolling_{hours}h_dominant_domain_pct", 0)
            features[f"rolling_{hours}h_trend"] = rws.get(f"rolling_{hours}h_trend", 0)

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

        Uses configurable model blending (GradientBoosting, RandomForest, LightGBM)
        and anomaly detection to generate predictions with confidence scores.

        Returns:
            Dictionary of predictions by target with confidence and anomaly info
        """
        self.logger.info("Generating predictions...")

        if not self.models:
            self.logger.warning("No models loaded. Train models first.")
            return {}

        # Get latest snapshot from cache or discovery data
        snapshot = await self._get_current_snapshot()
        if not snapshot:
            self.logger.error("No current snapshot available for prediction")
            return {}

        # Get previous snapshot for lag features
        prev_snapshot = await self._get_previous_snapshot()

        # Compute rolling stats (last 7 snapshots)
        rolling_stats = await self._compute_rolling_stats()

        # Compute rolling window stats from live activity log
        rolling_window_stats = await self._compute_rolling_window_stats()

        # Build feature config
        config = await self._get_feature_config()

        # Extract features from current state
        features = await self._extract_features(
            snapshot,
            config=config,
            prev_snapshot=prev_snapshot,
            rolling_stats=rolling_stats,
            rolling_window_stats=rolling_window_stats
        )

        if features is None:
            self.logger.error("Failed to extract features from snapshot")
            return {}

        # Generate predictions for each trained model
        predictions_dict = {}
        feature_names = await self._get_feature_names(config)

        # Build feature vector in correct order (same for all models)
        feature_vector = [features.get(name, 0) for name in feature_names]
        X = np.array([feature_vector], dtype=float)

        # Detect anomalies if model exists
        is_anomaly = False
        anomaly_score = None
        if "anomaly_detector" in self.models:
            try:
                anomaly_model = self.models["anomaly_detector"]["model"]
                anomaly_score = float(anomaly_model.decision_function(X)[0])
                # Negative score = anomaly (more negative = more anomalous)
                is_anomaly = bool(anomaly_model.predict(X)[0] == -1)
                self.logger.info(f"Anomaly detection: score={anomaly_score:.3f}, is_anomaly={is_anomaly}")
            except Exception as e:
                self.logger.error(f"Anomaly detection failed: {e}")

        for target, model_data in self.models.items():
            # Skip anomaly detector (already processed)
            if target == "anomaly_detector":
                continue

            try:
                # Scale features
                scaler = model_data["scaler"]
                X_scaled = scaler.transform(X)

                # Collect predictions from all enabled models
                individual_preds: Dict[str, float] = {}
                active_weights: Dict[str, float] = {}

                if self.enabled_models.get("gb") and "gb_model" in model_data:
                    individual_preds["gb"] = float(model_data["gb_model"].predict(X_scaled)[0])
                    active_weights["gb"] = self.model_weights.get("gb", 0.35)

                if self.enabled_models.get("rf") and "rf_model" in model_data:
                    individual_preds["rf"] = float(model_data["rf_model"].predict(X_scaled)[0])
                    active_weights["rf"] = self.model_weights.get("rf", 0.25)

                if self.enabled_models.get("lgbm") and "lgbm_model" in model_data:
                    individual_preds["lgbm"] = float(model_data["lgbm_model"].predict(X_scaled)[0])
                    active_weights["lgbm"] = self.model_weights.get("lgbm", 0.40)

                if not individual_preds:
                    self.logger.warning(f"No enabled models produced predictions for {target}")
                    continue

                # Normalize weights to sum to 1.0
                weight_sum = sum(active_weights.values())
                normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}

                # Blend predictions using normalized weights
                blended_pred = sum(
                    normalized_weights[k] * individual_preds[k]
                    for k in individual_preds
                )

                # Calculate confidence based on model agreement
                # Standard deviation of predictions relative to mean
                pred_values = list(individual_preds.values())
                avg_pred = sum(pred_values) / len(pred_values)

                if len(pred_values) > 1 and abs(avg_pred) > 1e-6:
                    max_diff = max(abs(p - avg_pred) for p in pred_values)
                    rel_diff = max_diff / abs(avg_pred)
                    confidence = max(0.0, min(1.0, 1.0 - rel_diff))
                elif len(pred_values) == 1:
                    # Single model — no agreement signal, moderate confidence
                    confidence = 0.7
                else:
                    # All predictions near zero
                    max_diff = max(abs(p - avg_pred) for p in pred_values) if pred_values else 0
                    confidence = 1.0 if max_diff < 0.1 else 0.5

                # Build prediction entry with per-model values
                pred_entry = {
                    "value": round(blended_pred, 2),
                    "confidence": round(confidence, 3),
                    "is_anomaly": is_anomaly,
                    "blend_weights": {k: round(v, 3) for k, v in normalized_weights.items()},
                }

                # Include individual model predictions
                if "gb" in individual_preds:
                    pred_entry["gb_prediction"] = round(individual_preds["gb"], 2)
                if "rf" in individual_preds:
                    pred_entry["rf_prediction"] = round(individual_preds["rf"], 2)
                if "lgbm" in individual_preds:
                    pred_entry["lgbm_prediction"] = round(individual_preds["lgbm"], 2)

                predictions_dict[target] = pred_entry

                model_details = ", ".join(
                    f"{k.upper()}={v:.2f}" for k, v in individual_preds.items()
                )
                self.logger.debug(
                    f"Prediction for {target}: {blended_pred:.2f} "
                    f"({model_details}, conf={confidence:.3f})"
                )

            except Exception as e:
                self.logger.error(f"Failed to predict {target}: {e}")
                continue

        # Build final result
        result = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions_dict,
            "anomaly_detected": is_anomaly,
            "anomaly_score": round(anomaly_score, 3) if anomaly_score is not None else None,
            "feature_count": len(feature_names),
            "model_count": len([k for k in self.models.keys() if k != "anomaly_detector"])
        }

        # Store in cache
        await self.hub.set_cache(
            "ml_predictions",
            result,
            category="predictions",
            ttl_seconds=86400  # 24 hours
        )

        self.logger.info(
            f"Generated {len(predictions_dict)} predictions "
            f"(anomaly_detected={is_anomaly})"
        )

        return result

    async def _get_current_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get latest snapshot from cache or build from discovery data.

        Returns:
            Snapshot dictionary or None if unavailable
        """
        # Try to get latest snapshot from cache
        snapshot_entry = await self.hub.get_cache("latest_snapshot")
        if snapshot_entry:
            return snapshot_entry.get("data")

        # Fall back to building from discovery data
        discovery_entry = await self.hub.get_cache("discovery")
        if not discovery_entry:
            return None

        # Build minimal snapshot from discovery data
        discovery = discovery_entry.get("data", {})
        snapshot = {
            "date": datetime.now().isoformat(),
            "power": discovery.get("power_monitoring", {}),
            "lights": discovery.get("lighting", {}),
            "occupancy": discovery.get("occupancy", {}),
            "motion": discovery.get("motion", {}),
            "climate": discovery.get("climate", {}),
            "weather": {},  # Would need separate weather API call
        }

        return snapshot

    async def _get_previous_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get previous snapshot for lag features.

        Returns:
            Previous snapshot or None if unavailable
        """
        # Look for historical snapshots in training data dir
        snapshot_files = sorted(self.training_data_dir.glob("*.json"))

        if len(snapshot_files) >= 2:
            # Return second-to-last (most recent historical)
            try:
                with open(snapshot_files[-2]) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load previous snapshot: {e}")

        return None

    async def _compute_rolling_stats(self) -> Dict[str, float]:
        """Compute rolling statistics from recent snapshots.

        Returns:
            Dictionary of rolling statistics
        """
        stats = {}

        # Load last 7 snapshots for rolling calculations
        snapshot_files = sorted(self.training_data_dir.glob("*.json"))
        if len(snapshot_files) < 7:
            return stats

        recent_snapshots = []
        for snapshot_file in snapshot_files[-7:]:
            try:
                with open(snapshot_file) as f:
                    recent_snapshots.append(json.load(f))
            except Exception as e:
                self.logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")
                continue

        if not recent_snapshots:
            return stats

        # Compute rolling means
        power_values = [s.get("power", {}).get("total_watts", 0) for s in recent_snapshots]
        lights_values = [s.get("lights", {}).get("on", 0) for s in recent_snapshots]

        stats["power_mean_7d"] = sum(power_values) / len(power_values) if power_values else 0
        stats["lights_mean_7d"] = sum(lights_values) / len(lights_values) if lights_values else 0

        return stats

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
