"""Isolation Forest model for contextual anomaly detection."""

import os
import pickle

from aria.engine.models.registry import ModelRegistry, BaseModel

HAS_SKLEARN = True
try:
    from sklearn.ensemble import IsolationForest
    import numpy as np
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

    X = np.array([snapshot_features], dtype=float)
    score = float(model.decision_function(X)[0])
    is_anomaly = model.predict(X)[0] == -1

    return {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(score, 4),
        "severity": "high" if score < -0.3 else ("medium" if score < -0.1 else "low"),
    }


@ModelRegistry.register("isolation_forest")
class IsolationForestModel(BaseModel):
    """IsolationForest for multi-dimensional anomaly detection."""

    def train(self, feature_names, X, model_dir):
        """Train IsolationForest for contextual anomaly detection.

        Returns training metrics dict.
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed"}

        X_arr = np.array(X, dtype=float)
        if len(X_arr) < 14:
            return {"error": f"insufficient data ({len(X_arr)} samples)"}

        model = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42,
        )
        model.fit(X_arr)

        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "anomaly_detector.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return {"samples": len(X_arr), "contamination": 0.05}

    def predict(self, snapshot_features, model_dir):
        """Score a snapshot for multi-dimensional anomalies.

        Returns dict with is_anomaly, anomaly_score, severity.
        Delegates to module-level detect_contextual_anomalies().
        """
        return detect_contextual_anomalies(snapshot_features, model_dir)
