"""Isolation Forest model for contextual anomaly detection."""

import logging
import os
import pickle

from aria.engine.models.registry import BaseModel, ModelRegistry

logger = logging.getLogger(__name__)

HAS_SKLEARN = True
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
except ImportError:
    HAS_SKLEARN = False


def detect_contextual_anomalies(snapshot_features, model_dir):
    """Score a snapshot for multi-dimensional anomalies using IsolationForest."""
    if not HAS_SKLEARN:
        return {"is_anomaly": False, "anomaly_score": 0}

    model_path = os.path.join(model_dir, "anomaly_detector.pkl")
    if not os.path.isfile(model_path):
        return {"is_anomaly": False, "anomaly_score": 0}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Check if autoencoder is enabled for this model
    config_path = os.path.join(model_dir, "anomaly_config.pkl")
    ae_enabled = False
    if os.path.isfile(config_path):
        with open(config_path, "rb") as f:
            anomaly_config = pickle.load(f)
        ae_enabled = anomaly_config.get("autoencoder_enabled", False)

    X = np.array([snapshot_features], dtype=float)

    X_original = X
    if ae_enabled:
        from aria.engine.models.autoencoder import Autoencoder

        ae = Autoencoder()
        errors = ae.reconstruction_errors(X, model_dir)
        if errors is not None:
            X = np.column_stack([X, errors])

    try:
        score = float(model.decision_function(X)[0])
        is_anomaly = model.predict(X)[0] == -1
    except ValueError:
        if X is not X_original:
            logger.warning(
                "Dimension mismatch with autoencoder features "
                "(model expects %d, got %d) — falling back to base features",
                model.n_features_in_,
                X.shape[1],
            )
            X = X_original
            score = float(model.decision_function(X)[0])
            is_anomaly = model.predict(X)[0] == -1
        else:
            raise

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(score, 4),
        "severity": "high" if score < -0.3 else ("medium" if score < -0.1 else "low"),
    }


@ModelRegistry.register("isolation_forest")
class IsolationForestModel(BaseModel):
    """IsolationForest for multi-dimensional anomaly detection."""

    def train(self, feature_names, X, model_dir, use_autoencoder=False, contamination=0.05):
        """Train IsolationForest for contextual anomaly detection.

        Args:
            feature_names: list of feature names
            X: training data array
            model_dir: directory to save model artifacts
            use_autoencoder: if True, train an Autoencoder first and append
                reconstruction error as an additional feature
            contamination: fraction of outliers in the dataset (default 0.05)

        Returns:
            dict with training metrics.
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed"}

        X_arr = np.array(X, dtype=float)
        if len(X_arr) < 14:
            return {"error": f"insufficient data ({len(X_arr)} samples)"}

        os.makedirs(model_dir, exist_ok=True)

        ae_enabled = False
        if use_autoencoder:
            from aria.engine.models.autoencoder import Autoencoder

            ae = Autoencoder()
            ae_result = ae.train(X_arr, model_dir)
            if "error" not in ae_result:
                errors = ae.reconstruction_errors(X_arr, model_dir)
                if errors is not None:
                    X_arr = np.column_stack([X_arr, errors])
                    ae_enabled = True

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        model.fit(X_arr)

        model_path = os.path.join(model_dir, "anomaly_detector.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save config so predict() knows whether to use autoencoder
        config_path = os.path.join(model_dir, "anomaly_config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump({"autoencoder_enabled": ae_enabled}, f)

        result = {
            "samples": len(X_arr),
            "contamination": contamination,
            "autoencoder_enabled": ae_enabled,
        }
        return result

    def predict(self, snapshot_features, model_dir):
        """Score a snapshot for multi-dimensional anomalies.

        Returns dict with is_anomaly, anomaly_score, severity.
        Delegates to module-level detect_contextual_anomalies().
        """
        return detect_contextual_anomalies(snapshot_features, model_dir)
