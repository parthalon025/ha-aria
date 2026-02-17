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
import math
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Suppress sklearn warning about feature names when using numpy arrays.
# Our feature pipeline guarantees alignment between training and prediction —
# the same _extract_features() dict order is used for both paths.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)

from aria.capabilities import Capability, DemandSignal  # noqa: E402
from aria.engine.fallback import FallbackTracker  # noqa: E402
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG as _ENGINE_FEATURE_CONFIG  # noqa: E402
from aria.engine.features.vector_builder import build_feature_vector as _engine_build_feature_vector  # noqa: E402
from aria.engine.hardware import recommend_tier, scan_hardware  # noqa: E402
from aria.engine.models.registry import TieredModelRegistry  # noqa: E402
from aria.engine.validation import validate_snapshot_batch  # noqa: E402
from aria.hub.core import IntelligenceHub, Module  # noqa: E402

logger = logging.getLogger(__name__)

# Feature engineering constants — will move to config store in Phase 2
DECAY_HALF_LIFE_DAYS = 7
WEEKDAY_ALIGNMENT_BONUS = 1.5
ROLLING_WINDOWS_HOURS = [1, 3, 6]


def _compute_trend(relevant: list[dict[str, Any]]) -> float:
    """Compute activity trend from first vs second half of relevant windows.

    Returns:
        1.0 (increasing), -1.0 (decreasing), or 0.0 (stable).
    """
    if len(relevant) < 2:
        return 0.0
    mid = len(relevant) // 2
    first_count = sum(w.get("event_count", 0) for w in relevant[:mid])
    second_count = sum(w.get("event_count", 0) for w in relevant[mid:])
    if first_count == 0 and second_count == 0:
        return 0.0
    if first_count == 0:
        return 1.0
    ratio = second_count / first_count
    if ratio > 1.2:
        return 1.0
    if ratio < 0.8:
        return -1.0
    return 0.0


def should_full_retrain(current_trees: int, max_trees: int = 500) -> bool:
    """Check if model has exceeded tree cap and needs full retrain.

    Phase 2: Wire into _train_model_for_target() to gate incremental vs full
    retrain based on config['incremental.max_total_trees'].
    """
    return current_trees > max_trees


class MLEngine(Module):
    """Machine learning prediction engine with adaptive capability mapping."""

    CAPABILITIES = [
        Capability(
            id="ml_realtime",
            name="Real-Time ML Predictions",
            description="Feature engineering, model training, and adaptive predictions for HA capabilities.",
            module="ml_engine",
            layer="hub",
            config_keys=["features.decay_half_life_days", "features.weekday_alignment_bonus"],
            test_paths=["tests/hub/test_ml_training.py", "tests/hub/test_reference_model.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
            demand_signals=[
                DemandSignal(
                    entity_domains=["sensor"],
                    device_classes=["power", "energy"],
                    min_entities=5,
                    description="Power/energy sensors for consumption prediction",
                ),
                DemandSignal(
                    entity_domains=["light", "switch"],
                    device_classes=[],
                    min_entities=3,
                    description="Controllable devices for usage pattern prediction",
                ),
                DemandSignal(
                    entity_domains=["binary_sensor"],
                    device_classes=["motion", "occupancy"],
                    min_entities=2,
                    description="Motion/occupancy sensors for presence prediction",
                ),
            ],
        ),
    ]

    def __init__(self, hub: IntelligenceHub, models_dir: str, training_data_dir: str):
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

        # Hardware-aware tiered model registry (Phase 1)
        self.registry = TieredModelRegistry.with_defaults()
        hw_profile = scan_hardware()
        self.current_tier = recommend_tier(hw_profile)
        self.fallback_tracker = FallbackTracker(ttl_days=7)
        logger.info(
            f"ML Engine tier: {self.current_tier} (hw: {hw_profile.ram_gb}GB RAM, {hw_profile.cpu_cores} cores)"
        )

        # Model configuration — derived from registry entries that match the
        # current training pipeline (gb, rf, lgbm). Full registry-driven
        # training will replace this when _fit_all_models is refactored.
        _TRAINING_MODELS = {"gb", "rf", "lgbm"}
        resolved = [e for e in self.registry.resolve("power_watts", self.current_tier) if e.name in _TRAINING_MODELS]
        if not resolved:
            # Tier 1 or dependency-missing: fall back to legacy defaults so
            # the training pipeline still works with hardcoded model creation.
            logger.warning(
                f"No registry models matched training pipeline at tier {self.current_tier} — using legacy defaults"
            )
            self.enabled_models: dict[str, bool] = {"gb": True, "rf": True, "lgbm": True}
            self.model_weights: dict[str, float] = {"gb": 0.35, "rf": 0.25, "lgbm": 0.40}
        else:
            self.enabled_models = {e.name: True for e in resolved}
            total_w = sum(e.weight for e in resolved)
            self.model_weights = {e.name: e.weight / total_w for e in resolved} if total_w else {}

        # Online prediction blend weight (Phase 2)
        self.online_blend_weight = 0.3

        # MAE-based ensemble weight auto-tuner (Phase 2)
        from aria.engine.weight_tuner import EnsembleWeightTuner

        self.weight_tuner = EnsembleWeightTuner(window_days=7)

        # Loaded models cache
        self.models: dict[str, dict[str, Any]] = {}

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

    async def _apply_auto_weights(self):
        """Recompute ensemble weights from tuner and apply."""
        weights = self.weight_tuner.compute_weights()
        if weights:
            # Only update weights for models that are in both the tuner and the engine
            for model_key in self.model_weights:
                if model_key in weights:
                    self.model_weights[model_key] = weights[model_key]
            await self.hub.set_cache("ml_ensemble_weights", weights)
            self.logger.info(f"Auto-tuned weights: {weights}")

    def _collect_training_result(self, target: str, capability_name: str) -> dict[str, Any] | None:
        """Collect training metrics for a single target if available."""
        if target not in self.models or "accuracy_scores" not in self.models[target]:
            return None
        scores = self.models[target]["accuracy_scores"]
        r2_values = [v for k, v in scores.items() if k.endswith("_r2")]
        mae_values = [v for k, v in scores.items() if k.endswith("_mae")]
        avg_r2 = sum(r2_values) / len(r2_values) if r2_values else 0.0
        avg_mae = sum(mae_values) / len(mae_values) if mae_values else 0.0
        importance = self.models[target].get("feature_importance", {})
        top5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "r2": round(avg_r2, 3),
            "mae": round(avg_mae, 3),
            "top_features": [name for name, _ in top5],
        }

    async def _train_capability_targets(
        self, capabilities: dict, training_data: list
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Train models for all capability targets and collect results."""
        training_results: dict[str, dict[str, dict[str, Any]]] = {}
        for capability_name, capability_data in capabilities.items():
            if not capability_data.get("available"):
                continue
            prediction_targets = self.capability_predictions.get(capability_name)
            if not prediction_targets:
                continue
            self.logger.info(f"Training models for capability: {capability_name}")
            for target in prediction_targets:
                try:
                    await self._train_model_for_target(target, training_data, capability_name)
                    result = self._collect_training_result(target, capability_name)
                    if result:
                        training_results.setdefault(capability_name, {})[target] = result
                except Exception as e:
                    self.logger.error(f"Failed to train model for {target}: {e}")
        return training_results

    async def train_models(self, days_history: int = 60):
        """Train models using historical data."""
        self.logger.info(f"Training models with {days_history} days of history...")

        capabilities_entry = await self.hub.get_cache_fresh(
            "capabilities", timedelta(hours=48), caller="ml_engine.train"
        )
        if not capabilities_entry:
            self.logger.error("No capabilities in cache. Cannot train without discovery data.")
            return

        capabilities = capabilities_entry.get("data", {})
        training_data = await self._load_training_data(days_history)
        if not training_data:
            self.logger.error("No training data available")
            return

        self.logger.info(f"Loaded {len(training_data)} snapshots for training")
        training_results = await self._train_capability_targets(capabilities, training_data)

        await self._train_anomaly_detector(training_data)
        self.logger.info("Model training complete")

        trained_targets = []
        accuracy_summary = {}
        for target, model_data in self.models.items():
            if target != "anomaly_detector" and "accuracy_scores" in model_data:
                trained_targets.append(target)
                accuracy_summary[target] = model_data["accuracy_scores"]

        await self.hub.set_cache(
            "ml_training_metadata",
            {
                "last_trained": datetime.now().isoformat(),
                "days_history": days_history,
                "num_snapshots": len(training_data),
                "capabilities_trained": list(capabilities.keys()),
                "targets_trained": trained_targets,
                "accuracy_summary": accuracy_summary,
                "has_anomaly_detector": "anomaly_detector" in self.models,
            },
        )

        if training_results:
            await self._write_feedback_to_capabilities(training_results)

        config = await self._get_feature_config()
        config["last_modified"] = datetime.now().isoformat()
        config["modified_by"] = "ml_engine"
        await self.hub.set_cache("feature_config", config)

    async def _write_feedback_to_capabilities(self, training_results: dict[str, dict[str, dict[str, Any]]]):
        """Write ML accuracy feedback back to the capabilities cache.

        Closes the loop between ML training and capability usefulness scoring
        by updating each capability's ``ml_accuracy`` and ``predictability``
        component based on actual model performance.

        Args:
            training_results: Per-capability, per-target training metrics.
                Structure: ``{cap_name: {target: {"r2": float, "mae": float, "top_features": list}}}``
        """
        caps_entry = await self.hub.get_cache("capabilities")
        if not caps_entry:
            self.logger.warning("Cannot write ML feedback: capabilities cache not found")
            return

        caps = caps_entry.get("data", {})
        now_iso = datetime.now().isoformat()
        updated_count = 0

        for cap_name, targets in training_results.items():
            if cap_name not in caps:
                continue

            cap = caps[cap_name]

            # Compute mean R² across all targets for this capability
            r2_values = [t["r2"] for t in targets.values()]
            mean_r2 = sum(r2_values) / len(r2_values) if r2_values else 0.0
            # Clamp to [0, 1] — negative R² means worse than baseline
            mean_r2_clamped = max(0.0, min(1.0, mean_r2))

            # Build per-target detail dict
            targets_detail = {}
            all_top_features = []
            for target_name, metrics in targets.items():
                targets_detail[target_name] = {
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                }
                all_top_features.extend(metrics.get("top_features", []))

            # Deduplicate top features while preserving order
            seen = set()
            unique_top_features = []
            for f in all_top_features:
                if f not in seen:
                    seen.add(f)
                    unique_top_features.append(f)

            # Write ml_accuracy block
            cap["ml_accuracy"] = {
                "mean_r2": round(mean_r2, 3),
                "targets": targets_detail,
                "last_trained": now_iso,
                "feature_importance_top5": unique_top_features[:5],
            }

            # Update predictability component in usefulness scoring
            if "usefulness_components" not in cap:
                cap["usefulness_components"] = {}
            cap["usefulness_components"]["predictability"] = round(mean_r2_clamped * 100)

            updated_count += 1

        if updated_count > 0:
            await self.hub.set_cache("capabilities", caps, {"source": "ml_feedback"})
            self.logger.info(f"ML feedback written to {updated_count} capabilities in cache")

    async def _load_training_data(self, days: int) -> list[dict[str, Any]]:
        """Load historical snapshots for training.

        Args:
            days: Number of days to load

        Returns:
            List of snapshot dictionaries
        """
        raw_snapshots = []
        today = datetime.now()

        for i in range(days):
            date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            snapshot_file = self.training_data_dir / f"{date_str}.json"

            if snapshot_file.exists():
                try:
                    with open(snapshot_file) as f:
                        snapshot = json.load(f)
                        raw_snapshots.append(snapshot)
                except (OSError, json.JSONDecodeError) as e:
                    self.logger.warning(f"Failed to load snapshot {snapshot_file}: {e}")

        valid, rejected = validate_snapshot_batch(raw_snapshots)
        if rejected:
            self.logger.warning(f"Rejected {len(rejected)} of {len(raw_snapshots)} snapshots (corrupt/incomplete data)")

        return valid

    @staticmethod
    def _fit_all_models(X_train, y_train, w_train):
        """Fit GB, RF, LightGBM, and IsolationForest models on training data."""
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=max(3, len(X_train) // 20),
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_train, y_train, sample_weight=w_train)

        rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(X_train, y_train, sample_weight=w_train)

        lgbm_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=15,
            min_child_samples=max(3, len(X_train) // 20),
            subsample=0.8,
            random_state=42,
            verbosity=-1,
            importance_type="gain",
        )
        lgbm_model.fit(X_train, y_train, sample_weight=w_train)

        iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso_model.fit(X_train)

        return gb_model, rf_model, lgbm_model, iso_model

    @staticmethod
    def _compute_validation_metrics(gb_model, rf_model, lgbm_model, X_val, y_val) -> dict[str, float]:
        """Compute MAE and R2 for each model on validation data."""
        gb_pred = gb_model.predict(X_val)
        rf_pred = rf_model.predict(X_val)
        lgbm_pred = lgbm_model.predict(X_val)
        has_multi = len(y_val) > 1
        return {
            "gb_mae": round(mean_absolute_error(y_val, gb_pred), 3),
            "gb_r2": round(r2_score(y_val, gb_pred), 3) if has_multi else 0.0,
            "rf_mae": round(mean_absolute_error(y_val, rf_pred), 3),
            "rf_r2": round(r2_score(y_val, rf_pred), 3) if has_multi else 0.0,
            "lgbm_mae": round(mean_absolute_error(y_val, lgbm_pred), 3),
            "lgbm_r2": round(r2_score(y_val, lgbm_pred), 3) if has_multi else 0.0,
        }

    async def _train_model_for_target(self, target: str, training_data: list[dict[str, Any]], capability_name: str):
        """Train a model for a specific prediction target."""
        self.logger.info(f"Training model for target: {target}")

        X, y, sample_weights = await self._build_training_dataset(training_data, target)

        if len(X) < 14:
            self.logger.warning(f"Insufficient training data for {target}: {len(X)} samples (need 14+)")
            return

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train = sample_weights[:split_idx] if len(sample_weights) > 0 else None

        gb_model, rf_model, lgbm_model, iso_model = self._fit_all_models(X_train, y_train, w_train)
        accuracy_scores = self._compute_validation_metrics(gb_model, rf_model, lgbm_model, X_val, y_val)

        config = await self._get_feature_config()
        feature_names = await self._get_feature_names(config)
        feature_importance = {
            name: round(float(imp), 4) for name, imp in zip(feature_names, rf_model.feature_importances_, strict=False)
        }
        lgbm_feature_importance = {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, lgbm_model.feature_importances_, strict=False)
        }

        scaler = StandardScaler()
        scaler.fit(X_train)

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
            "accuracy_scores": accuracy_scores,
        }

        model_file = self.models_dir / f"{target}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)
        self.models[target] = model_data

        self.logger.info(
            f"Model trained for {target}: "
            f"{len(X)} samples ({len(X_train)} train, {len(X_val)} val), "
            f"{len(feature_names)} features, "
            f"GB MAE={accuracy_scores['gb_mae']:.2f} R²={accuracy_scores['gb_r2']:.3f}, "
            f"RF MAE={accuracy_scores['rf_mae']:.2f} R²={accuracy_scores['rf_r2']:.3f}, "
            f"LGBM MAE={accuracy_scores['lgbm_mae']:.2f} R²={accuracy_scores['lgbm_r2']:.3f}"
        )

    async def _train_anomaly_detector(self, training_data: list[dict[str, Any]]):
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
                recent = training_data[max(0, i - 7) : i]
                rolling_stats["power_mean_7d"] = sum(s.get("power", {}).get("total_watts", 0) for s in recent) / len(
                    recent
                )
                rolling_stats["lights_mean_7d"] = sum(s.get("lights", {}).get("on", 0) for s in recent) / len(recent)

            features = await self._extract_features(
                snapshot, config=config, prev_snapshot=prev_snapshot, rolling_stats=rolling_stats
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
            random_state=42,
        )
        model.fit(X)

        # Save anomaly detector
        model_data = {
            "model": model,
            "trained_at": datetime.now().isoformat(),
            "num_samples": len(X),
            "contamination": 0.05,
        }

        model_file = self.models_dir / "anomaly_detector.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        # Cache in memory
        self.models["anomaly_detector"] = model_data

        self.logger.info(f"Anomaly detector trained: {len(X)} samples, contamination=0.05")

    async def _build_training_dataset(
        self, snapshots: list[dict[str, Any]], target: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                recent = snapshots[max(0, i - 7) : i]
                rolling_stats["power_mean_7d"] = sum(s.get("power", {}).get("total_watts", 0) for s in recent) / len(
                    recent
                )
                rolling_stats["lights_mean_7d"] = sum(s.get("lights", {}).get("on", 0) for s in recent) / len(recent)

            # Extract features
            features = await self._extract_features(
                snapshot, config=config, prev_snapshot=prev_snapshot, rolling_stats=rolling_stats
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

    async def _get_feature_config(self) -> dict[str, Any]:
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

    async def _get_feature_names(self, config: dict[str, Any] | None = None) -> list[str]:
        """Return ordered list of feature names from config.

        Args:
            config: Feature configuration (uses default if None)

        Returns:
            List of feature names in order
        """
        if config is None:
            config = await self._get_feature_config()

        names: list[str] = []
        self._collect_time_feature_names(config.get("time_features", {}), names)
        self._collect_dict_feature_names(config.get("weather_features", {}), names, prefix="weather_")
        self._collect_dict_feature_names(config.get("home_features", {}), names)
        self._collect_dict_feature_names(config.get("lag_features", {}), names)
        self._collect_dict_feature_names(config.get("interaction_features", {}), names)

        # Rolling window features (always included)
        for hours in ROLLING_WINDOWS_HOURS:
            names.extend(
                [
                    f"rolling_{hours}h_event_count",
                    f"rolling_{hours}h_domain_entropy",
                    f"rolling_{hours}h_dominant_domain_pct",
                    f"rolling_{hours}h_trend",
                ]
            )

        return names

    @staticmethod
    def _collect_time_feature_names(tc: dict[str, Any], names: list[str]) -> None:
        """Append time feature names from config section to names list."""
        _SIN_COS_PAIRS = {
            "hour_sin_cos": ("hour_sin", "hour_cos"),
            "dow_sin_cos": ("dow_sin", "dow_cos"),
            "month_sin_cos": ("month_sin", "month_cos"),
            "day_of_year_sin_cos": ("day_of_year_sin", "day_of_year_cos"),
        }
        for key, pair in _SIN_COS_PAIRS.items():
            if tc.get(key):
                names.extend(pair)
        for simple in [
            "is_weekend",
            "is_holiday",
            "is_night",
            "is_work_hours",
            "minutes_since_sunrise",
            "minutes_until_sunset",
            "daylight_remaining_pct",
        ]:
            if tc.get(simple):
                names.append(simple)

    @staticmethod
    def _collect_dict_feature_names(
        section: dict[str, Any],
        names: list[str],
        prefix: str = "",
    ) -> None:
        """Append enabled feature names from a config section to names list."""
        for key, enabled in section.items():
            if enabled:
                names.append(f"{prefix}{key}")

    def _compute_time_features(self, snapshot: dict[str, Any]) -> dict[str, float]:
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
                "hour_sin": 0,
                "hour_cos": 0,
                "dow_sin": 0,
                "dow_cos": 0,
                "month_sin": 0,
                "month_cos": 0,
                "day_of_year_sin": 0,
                "day_of_year_cos": 0,
                "is_weekend": 0,
                "is_holiday": 0,
                "is_night": 0,
                "is_work_hours": 0,
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
        self, snapshots: list[dict[str, Any]], reference_date: datetime | None = None
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

    async def _compute_rolling_window_stats(self, activity_log: dict[str, Any] | None = None) -> dict[str, float]:
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

        stats: dict[str, float] = {}

        if not activity_log:
            return self._empty_rolling_window_stats()

        windows = activity_log.get("windows", [])
        if not windows:
            return self._empty_rolling_window_stats()

        now = datetime.now()
        for hours in ROLLING_WINDOWS_HOURS:
            cutoff = (now - timedelta(hours=hours)).isoformat()
            relevant = [w for w in windows if w.get("window_start", "") >= cutoff]
            self._compute_single_window_stats(stats, hours, relevant)

        return stats

    @staticmethod
    def _empty_rolling_window_stats() -> dict[str, float]:
        """Return zero-valued rolling window stats for all window sizes."""
        stats: dict[str, float] = {}
        for hours in ROLLING_WINDOWS_HOURS:
            stats[f"rolling_{hours}h_event_count"] = 0
            stats[f"rolling_{hours}h_domain_entropy"] = 0
            stats[f"rolling_{hours}h_dominant_domain_pct"] = 0
            stats[f"rolling_{hours}h_trend"] = 0
        return stats

    @staticmethod
    def _compute_single_window_stats(
        stats: dict[str, float],
        hours: int,
        relevant: list[dict[str, Any]],
    ) -> None:
        """Compute event count, entropy, dominant domain pct, and trend for one window size."""
        prefix = f"rolling_{hours}h"
        if not relevant:
            stats[f"{prefix}_event_count"] = 0
            stats[f"{prefix}_domain_entropy"] = 0
            stats[f"{prefix}_dominant_domain_pct"] = 0
            stats[f"{prefix}_trend"] = 0
            return

        # Aggregate domain counts across windows
        total_events = 0
        domain_counts: dict[str, int] = {}
        for w in relevant:
            total_events += w.get("event_count", 0)
            for domain, count in w.get("by_domain", {}).items():
                domain_counts[domain] = domain_counts.get(domain, 0) + count

        stats[f"{prefix}_event_count"] = total_events

        # Domain entropy and dominant domain pct
        if total_events > 0 and domain_counts:
            entropy = 0.0
            for count in domain_counts.values():
                p = count / total_events
                if p > 0:
                    entropy -= p * math.log2(p)
            stats[f"{prefix}_domain_entropy"] = round(entropy, 4)
            max_count = max(domain_counts.values())
            stats[f"{prefix}_dominant_domain_pct"] = round(max_count / total_events, 4)
        else:
            stats[f"{prefix}_domain_entropy"] = 0
            stats[f"{prefix}_dominant_domain_pct"] = 0

        # Activity trend: compare first half vs second half
        stats[f"{prefix}_trend"] = _compute_trend(relevant)

    async def _extract_features(
        self,
        snapshot: dict[str, Any],
        config: dict[str, Any] | None = None,
        prev_snapshot: dict[str, Any] | None = None,
        rolling_stats: dict[str, float] | None = None,
        rolling_window_stats: dict[str, float] | None = None,
    ) -> dict[str, float] | None:
        """Extract feature vector from snapshot using shared vector_builder.

        Delegates base feature extraction to vector_builder.build_feature_vector()
        (single source of truth), then appends hub-specific rolling window features
        from the live activity log.
        """
        if config is None:
            config = await self._get_feature_config()

        # If snapshot lacks time_features, compute them (daily snapshots may not have them)
        if "time_features" not in snapshot:
            snapshot = {**snapshot, "time_features": self._compute_time_features(snapshot)}

        # Delegate base feature extraction to shared engine builder
        features = _engine_build_feature_vector(snapshot, config, prev_snapshot, rolling_stats)

        # Hub-only: append rolling window features from live activity log
        rws = rolling_window_stats or {}
        for hours in ROLLING_WINDOWS_HOURS:
            features[f"rolling_{hours}h_event_count"] = rws.get(f"rolling_{hours}h_event_count", 0)
            features[f"rolling_{hours}h_domain_entropy"] = rws.get(f"rolling_{hours}h_domain_entropy", 0)
            features[f"rolling_{hours}h_dominant_domain_pct"] = rws.get(f"rolling_{hours}h_dominant_domain_pct", 0)
            features[f"rolling_{hours}h_trend"] = rws.get(f"rolling_{hours}h_trend", 0)

        return features

    def _extract_target(self, snapshot: dict[str, Any], target: str) -> float | None:
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

    async def generate_predictions(self) -> dict[str, Any]:
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

        X, feature_names = await self._build_prediction_feature_vector()
        if X is None:
            return {}

        # Detect anomalies if model exists
        is_anomaly, anomaly_score = self._run_anomaly_detection(X)

        # Generate predictions for each target
        predictions_dict = self._predict_all_targets(X, is_anomaly)

        # Build and cache final result
        result = {
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions_dict,
            "anomaly_detected": is_anomaly,
            "anomaly_score": round(anomaly_score, 3) if anomaly_score is not None else None,
            "feature_count": len(feature_names),
            "model_count": len([k for k in self.models if k != "anomaly_detector"]),
        }

        await self.hub.set_cache(
            "ml_predictions",
            result,
            category="predictions",
            ttl_seconds=86400,
        )

        self.logger.info(f"Generated {len(predictions_dict)} predictions (anomaly_detected={is_anomaly})")
        return result

    async def _build_prediction_feature_vector(self) -> tuple[np.ndarray | None, list[str]]:
        """Build the feature vector for prediction from current state.

        Returns:
            Tuple of (X array, feature_names) or (None, []) if unavailable.
        """
        snapshot = await self._get_current_snapshot()
        if not snapshot:
            self.logger.error("No current snapshot available for prediction")
            return None, []

        prev_snapshot = await self._get_previous_snapshot()
        rolling_stats = await self._compute_rolling_stats()
        rolling_window_stats = await self._compute_rolling_window_stats()
        config = await self._get_feature_config()

        features = await self._extract_features(
            snapshot,
            config=config,
            prev_snapshot=prev_snapshot,
            rolling_stats=rolling_stats,
            rolling_window_stats=rolling_window_stats,
        )
        if features is None:
            self.logger.error("Failed to extract features from snapshot")
            return None, []

        feature_names = await self._get_feature_names(config)
        feature_vector = [features.get(name, 0) for name in feature_names]
        return np.array([feature_vector], dtype=float), feature_names

    def _run_anomaly_detection(self, X: np.ndarray) -> tuple[bool, float | None]:
        """Run anomaly detection on feature vector.

        Returns:
            (is_anomaly, anomaly_score) tuple.
        """
        if "anomaly_detector" not in self.models:
            return False, None
        try:
            anomaly_model = self.models["anomaly_detector"]["model"]
            score = float(anomaly_model.decision_function(X)[0])
            is_anomaly = bool(anomaly_model.predict(X)[0] == -1)
            self.logger.info(f"Anomaly detection: score={score:.3f}, is_anomaly={is_anomaly}")
            return is_anomaly, score
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return False, None

    def _predict_all_targets(self, X: np.ndarray, is_anomaly: bool) -> dict[str, Any]:
        """Generate blended predictions for all trained targets.

        Returns:
            Dict mapping target name to prediction entry.
        """
        predictions_dict: dict[str, Any] = {}
        for target, model_data in self.models.items():
            if target == "anomaly_detector":
                continue
            try:
                entry = self._predict_single_target(target, model_data, X, is_anomaly)
                if entry is not None:
                    predictions_dict[target] = entry
            except Exception as e:
                self.logger.error(f"Failed to predict {target}: {e}")
        return predictions_dict

    def _predict_single_target(
        self,
        target: str,
        model_data: dict,
        X: np.ndarray,
        is_anomaly: bool,
    ) -> dict[str, Any] | None:
        """Predict a single target using model blending.

        Returns:
            Prediction entry dict or None if no models produced predictions.
        """
        X_scaled = model_data["scaler"].transform(X)

        # Collect predictions from all enabled models
        individual_preds: dict[str, float] = {}
        active_weights: dict[str, float] = {}
        _MODEL_KEYS = [("gb", 0.35), ("rf", 0.25), ("lgbm", 0.40)]
        for model_key, default_weight in _MODEL_KEYS:
            if self.enabled_models.get(model_key) and f"{model_key}_model" in model_data:
                individual_preds[model_key] = float(model_data[f"{model_key}_model"].predict(X_scaled)[0])
                active_weights[model_key] = self.model_weights.get(model_key, default_weight)

        if not individual_preds:
            self.logger.warning(f"No enabled models produced predictions for {target}")
            return None

        # Blend predictions
        weight_sum = sum(active_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}
        blended_pred = sum(normalized_weights[k] * individual_preds[k] for k in individual_preds)

        confidence = self._compute_prediction_confidence(individual_preds)

        # Blend batch and online predictions (Phase 2)
        online_pred = self._get_online_prediction(target, model_data, X)
        final_pred = self._blend_batch_online(blended_pred, online_pred)

        pred_entry: dict[str, Any] = {
            "value": round(final_pred, 2),
            "confidence": round(confidence, 3),
            "is_anomaly": is_anomaly,
            "blend_weights": {k: round(v, 3) for k, v in normalized_weights.items()},
        }
        for model_key in ("gb", "rf", "lgbm"):
            if model_key in individual_preds:
                pred_entry[f"{model_key}_prediction"] = round(individual_preds[model_key], 2)

        # Include online prediction metadata when available
        if online_pred is not None:
            pred_entry["online_prediction"] = round(online_pred, 2)
            pred_entry["online_blend_weight"] = self.online_blend_weight

        model_details = ", ".join(f"{k.upper()}={v:.2f}" for k, v in individual_preds.items())
        online_detail = f", ONLINE={online_pred:.2f}" if online_pred is not None else ""
        self.logger.debug(
            f"Prediction for {target}: {final_pred:.2f} ({model_details}{online_detail}, conf={confidence:.3f})"
        )
        return pred_entry

    def _get_online_prediction(self, target: str, model_data: dict, X: np.ndarray) -> float | None:
        """Look up online learner prediction for a target.

        Reconstructs a features dict from the numpy array and model_data's
        feature_names, then queries the online learner module.

        Returns:
            Online prediction float, or None if unavailable.
        """
        # Access modules dict directly (sync) — hub.get_module() is async
        # but only does a dict lookup, and we're in a sync call chain.
        online_learner = getattr(self.hub, "modules", {}).get("online_learner")
        if online_learner is None:
            return None

        feature_names = model_data.get("feature_names", [])
        if feature_names and X.shape[1] == len(feature_names):
            features_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}
        else:
            features_dict = {}

        try:
            return online_learner.get_prediction(target, features_dict)
        except Exception as e:
            self.logger.debug(f"Online prediction failed for {target}: {e}")
            return None

    def _blend_batch_online(self, batch_pred: float, online_pred: float | None) -> float:
        """Blend batch and online predictions.

        Falls back to batch-only if online prediction is None or weight is zero.
        """
        if online_pred is None or self.online_blend_weight <= 0:
            return batch_pred
        batch_weight = 1.0 - self.online_blend_weight
        return batch_weight * batch_pred + self.online_blend_weight * online_pred

    @staticmethod
    def _compute_prediction_confidence(individual_preds: dict[str, float]) -> float:
        """Compute confidence score based on model agreement."""
        pred_values = list(individual_preds.values())
        avg_pred = sum(pred_values) / len(pred_values)
        if len(pred_values) > 1 and abs(avg_pred) > 1e-6:
            max_diff = max(abs(p - avg_pred) for p in pred_values)
            return max(0.0, min(1.0, 1.0 - max_diff / abs(avg_pred)))
        if len(pred_values) == 1:
            return 0.7
        max_diff = max(abs(p - avg_pred) for p in pred_values) if pred_values else 0
        return 1.0 if max_diff < 0.1 else 0.5

    async def _get_current_snapshot(self) -> dict[str, Any] | None:
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

    async def _get_previous_snapshot(self) -> dict[str, Any] | None:
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

    async def _compute_rolling_stats(self) -> dict[str, float]:
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

    async def _train_reference_model(self, features, targets):
        """Train clean reference model without meta-learner modifications.

        Maintains a parallel unmodified model to distinguish meta-learner
        errors from genuine behavioral drift.
        """
        pass  # Phase 2: Wire into retrain cycle to train parallel model with
        # frozen feature config.  compare_model_accuracy() in intelligence.py
        # provides the comparison logic once both model sets exist.

    async def on_event(self, event_type: str, data: dict[str, Any]):
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

        Checks ml_training_metadata cache to determine if training should run
        immediately (stale or missing) or wait for the next scheduled interval.

        Args:
            interval_days: Days between training runs
        """
        # Determine staleness from cache metadata
        run_immediately = False
        try:
            metadata = await self.hub.get_cache("ml_training_metadata")
            if metadata is None:
                run_immediately = True
                self.logger.info("No training metadata found — will train immediately on startup")
            else:
                # Handle both {"data": {"last_trained": ...}} and {"last_trained": ...}
                data = metadata.get("data", metadata) if isinstance(metadata, dict) else metadata
                last_trained_str = data.get("last_trained") if isinstance(data, dict) else None
                if not last_trained_str:
                    run_immediately = True
                    self.logger.info("Training metadata missing last_trained — will train immediately")
                else:
                    last_trained = datetime.fromisoformat(last_trained_str)
                    days_since = (datetime.now() - last_trained).total_seconds() / 86400
                    if days_since > interval_days:
                        run_immediately = True
                        self.logger.info(
                            f"Last training was {days_since:.1f} days ago (>{interval_days}) — will train immediately"
                        )
                    else:
                        self.logger.info(
                            f"Last training was {days_since:.1f} days ago — next scheduled run in "
                            f"{interval_days - days_since:.1f} days"
                        )
        except Exception as e:
            run_immediately = True
            self.logger.warning(f"Failed to check training metadata: {e} — will train immediately")

        async def training_task():
            try:
                await self.train_models(days_history=60)
            except Exception as e:
                self.logger.error(f"Scheduled training failed: {e}")

        await self.hub.schedule_task(
            task_id="ml_training_periodic",
            coro=training_task,
            interval=timedelta(days=interval_days),
            run_immediately=run_immediately,
        )

        self.logger.info(f"Scheduled periodic training every {interval_days} days")
