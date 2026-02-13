"""Device failure prediction using RandomForest."""

import os
import pickle

HAS_SKLEARN = True
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
except ImportError:
    HAS_SKLEARN = False

# Domain encoding shared by train and predict
DOMAIN_MAP = {
    "sensor": 0, "switch": 1, "light": 2, "binary_sensor": 3,
    "device_tracker": 4, "lock": 5, "climate": 6,
}


def train_device_failure_model(snapshots, model_dir):
    """Train RandomForest for device failure prediction.

    Builds training data from historical snapshots: for each device that was
    ever unavailable, creates features from each day with label = went
    unavailable within 7 days.
    """
    if not HAS_SKLEARN or len(snapshots) < 14:
        return {"error": "insufficient data or sklearn not installed"}

    # Find all devices that were ever unavailable
    all_devices = set()
    for snap in snapshots:
        for eid in snap.get("entities", {}).get("unavailable_list", []):
            all_devices.add(eid)

    if not all_devices:
        return {"error": "no device outage data"}

    X = []
    y = []

    for device_id in all_devices:
        for i in range(len(snapshots) - 7):
            # Features: outage history up to this point
            history = snapshots[:i + 1]
            outage_7d = sum(1 for s in history[-7:] if device_id in s.get("entities", {}).get("unavailable_list", []))
            outage_30d = sum(1 for s in history[-30:] if device_id in s.get("entities", {}).get("unavailable_list", []))

            # Days since last outage
            days_since = 999
            for j, s in enumerate(reversed(history)):
                if device_id in s.get("entities", {}).get("unavailable_list", []):
                    days_since = j
                    break

            # Battery
            battery = -1
            batt_data = snapshots[i].get("batteries", {}).get(device_id, {})
            if batt_data:
                battery = batt_data.get("level", -1) or -1

            domain = device_id.split(".")[0]

            features = [
                outage_7d, outage_30d, min(days_since, 365),
                battery, DOMAIN_MAP.get(domain, 7),
            ]
            X.append(features)

            # Label: did this device go unavailable in the next 7 days?
            future = snapshots[i + 1:i + 8]
            went_unavailable = any(device_id in s.get("entities", {}).get("unavailable_list", []) for s in future)
            y.append(1 if went_unavailable else 0)

    if len(X) < 10 or sum(y) == 0:
        return {"error": "insufficient failure examples"}

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42,
    )
    model.fit(X_arr, y_arr)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "device_failure.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "samples": len(X_arr),
        "positive_rate": round(sum(y) / len(y), 3),
        "feature_names": ["outage_7d", "outage_30d", "days_since_outage", "battery", "domain"],
    }


def predict_device_failures(snapshots, model_dir):
    """Predict which devices are likely to fail in the next 7 days."""
    if not HAS_SKLEARN:
        return []

    model_path = os.path.join(model_dir, "device_failure.pkl")
    if not os.path.isfile(model_path):
        return []

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Find all devices that were ever unavailable
    all_devices = set()
    for snap in snapshots:
        for eid in snap.get("entities", {}).get("unavailable_list", []):
            all_devices.add(eid)

    predictions = []
    for device_id in all_devices:
        outage_7d = sum(1 for s in snapshots[-7:] if device_id in s.get("entities", {}).get("unavailable_list", []))
        outage_30d = sum(1 for s in snapshots[-30:] if device_id in s.get("entities", {}).get("unavailable_list", []))
        days_since = 999
        for j, s in enumerate(reversed(snapshots)):
            if device_id in s.get("entities", {}).get("unavailable_list", []):
                days_since = j
                break
        battery = -1
        if snapshots:
            batt_data = snapshots[-1].get("batteries", {}).get(device_id, {})
            if batt_data:
                battery = batt_data.get("level", -1) or -1
        domain = device_id.split(".")[0]

        features = np.array([[outage_7d, outage_30d, min(days_since, 365),
                               battery, DOMAIN_MAP.get(domain, 7)]], dtype=float)
        prob = model.predict_proba(features)[0]
        fail_prob = prob[1] if len(prob) > 1 else 0

        if fail_prob > 0.3:  # Only report if >30% chance
            predictions.append({
                "entity_id": device_id,
                "failure_probability": round(float(fail_prob), 3),
                "risk": "high" if fail_prob > 0.7 else ("medium" if fail_prob > 0.5 else "low"),
                "outages_last_7d": outage_7d,
                "battery": battery if battery >= 0 else None,
            })

    predictions.sort(key=lambda p: -p["failure_probability"])
    return predictions


# Canonical home: aria.engine.models.isolation_forest
from aria.engine.models.isolation_forest import detect_contextual_anomalies  # noqa: F401
