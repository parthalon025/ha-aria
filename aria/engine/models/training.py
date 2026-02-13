"""Model training orchestration and ML prediction pipeline."""

import json
import os
import pickle
from datetime import datetime

from aria.engine.config import AppConfig, PathConfig
from aria.engine.storage.data_store import DataStore

HAS_SKLEARN = True
try:
    import numpy as np
except ImportError:
    HAS_SKLEARN = False


def train_all_models(days=90, config=None, store=None):
    """Train all sklearn models from intra-day snapshots.

    Returns training results dict.
    """
    if not HAS_SKLEARN:
        print("sklearn not installed, skipping ML training")
        return {"error": "sklearn not installed"}

    if config is None:
        config = AppConfig()
    if store is None:
        store = DataStore(config.paths)

    models_dir = str(config.paths.models_dir)

    # Lazy imports — these modules may not be migrated yet
    from aria.engine.features.vector_builder import build_training_data
    from aria.engine.models.gradient_boosting import GradientBoostingModel
    from aria.engine.models.isolation_forest import IsolationForestModel
    from aria.engine.models.device_failure import train_device_failure_model

    # Load intra-day snapshots (or fall back to daily)
    snapshots = store.load_all_intraday_snapshots(days)
    if len(snapshots) < 14:
        # Fall back to daily snapshots
        snapshots = store.load_recent_snapshots(days)
    if len(snapshots) < 14:
        print(f"Insufficient training data ({len(snapshots)} snapshots, need 14+)")
        return {"error": f"insufficient data ({len(snapshots)} snapshots)"}

    feature_config = store.load_feature_config()
    feature_names, X, targets = build_training_data(snapshots, feature_config)

    results = {"trained_at": datetime.now().isoformat(), "models": {}}

    # Train continuous models
    gb_model = GradientBoostingModel()
    for metric in (feature_config or {}).get("target_metrics", []):
        if metric in targets:
            result = gb_model.train(metric, feature_names, X, targets[metric],
                                    models_dir, config.model)
            results["models"][metric] = result
            if "error" not in result:
                print(f"  {metric}: MAE={result['mae']}, R\u00b2={result['r2']}")

    # Train anomaly detector
    iso_model = IsolationForestModel()
    anomaly_result = iso_model.train(feature_names, X, models_dir)
    results["models"]["anomaly_detector"] = anomaly_result

    # Train device failure model (uses daily snapshots for longer history)
    daily_snaps = store.load_recent_snapshots(days)
    failure_result = train_device_failure_model(daily_snaps, models_dir)
    results["models"]["device_failure"] = failure_result
    if "error" not in failure_result:
        print(f"  device_failure: {failure_result['samples']} samples, positive_rate={failure_result['positive_rate']}")

    # Save training log
    os.makedirs(models_dir, exist_ok=True)
    log_path = os.path.join(models_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training log saved: {log_path}")

    return results


def predict_with_ml(snapshot, config=None, prev_snapshot=None, rolling_stats=None,
                    models_dir=None, store=None):
    """Generate ML predictions for a snapshot using trained models.

    Args:
        snapshot: Current snapshot dict.
        config: Feature config dict (from load_feature_config). Loaded from store if None.
        prev_snapshot: Previous snapshot for delta features.
        rolling_stats: Rolling statistics dict.
        models_dir: Path to saved models. Defaults to PathConfig().models_dir.
        store: DataStore instance for loading feature config.

    Returns dict of metric -> predicted value, or empty if no models.
    """
    if not HAS_SKLEARN:
        return {}

    from aria.engine.features.vector_builder import build_feature_vector, get_feature_names

    if config is None:
        if store is not None:
            config = store.load_feature_config()
        else:
            config = DataStore(PathConfig()).load_feature_config()
    if config is None:
        return {}

    if models_dir is None:
        models_dir = str(PathConfig().models_dir)

    feature_names = get_feature_names(config)
    fv = build_feature_vector(snapshot, config, prev_snapshot, rolling_stats)
    feature_row = [fv.get(name, 0) for name in feature_names]

    predictions = {}
    for metric in config.get("target_metrics", []):
        model_path = os.path.join(models_dir, f"{metric}.pkl")
        if not os.path.isfile(model_path):
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        X = np.array([feature_row], dtype=float)
        pred = float(model.predict(X)[0])
        predictions[metric] = round(pred, 1)

    return predictions


def train_continuous_model(metric_name, feature_names, X, y, model_dir, config=None):
    """Train a single GradientBoosting model for one metric.

    Convenience wrapper around GradientBoostingModel.train() for use by
    meta_learning validation (which needs to train in temp dirs).
    """
    from aria.engine.models.gradient_boosting import GradientBoostingModel
    model = GradientBoostingModel()
    return model.train(metric_name, feature_names, X, y, model_dir, config)


# Canonical home: aria.engine.predictions.predictor
from aria.engine.predictions.predictor import blend_predictions  # noqa: F401


def count_days_of_data(store_or_paths=None):
    """Count how many days of snapshot data exist.

    Accepts DataStore, PathConfig, or None (uses defaults).
    """
    if store_or_paths is None:
        paths = PathConfig()
    elif isinstance(store_or_paths, PathConfig):
        paths = store_or_paths
    elif isinstance(store_or_paths, DataStore):
        paths = store_or_paths.paths
    else:
        paths = PathConfig()
    daily_dir = paths.daily_dir
    if not daily_dir.is_dir():
        return 0
    return len([f for f in os.listdir(daily_dir) if f.endswith(".json")])
