"""SHAP-based model explainability for faithful prediction narration.

Provides functions to explain individual predictions using SHAP values,
producing structured attribution reports that ground LLM narration in
actual model behavior rather than post-hoc rationalization.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def explain_prediction(model, scaler, feature_names, feature_vector, top_n=5):
    """Explain a single prediction using SHAP TreeExplainer.

    Args:
        model: A trained tree-based model (GradientBoosting, RandomForest, etc.).
        scaler: A fitted StandardScaler (or similar) used during training.
        feature_names: List of feature name strings matching the feature vector.
        feature_vector: 1-D array of raw (unscaled) feature values for one sample.
        top_n: Number of top contributing features to return.

    Returns:
        List of dicts sorted by absolute contribution (descending), each with:
            - feature: str — feature name
            - contribution: float — SHAP value (absolute magnitude)
            - direction: str — "positive" or "negative"
            - raw_value: float — the raw (unscaled) feature value

    Raises:
        RuntimeError: If shap is not installed.
    """
    if not HAS_SHAP:
        raise RuntimeError("shap package is required for explain_prediction")

    feature_vector = np.asarray(feature_vector).reshape(1, -1)
    scaled = scaler.transform(feature_vector)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)

    # shap_values shape: (1, n_features) for regression
    sv = shap_values[0] if shap_values.ndim == 2 else shap_values

    # Sort by absolute SHAP value descending
    indices = np.argsort(np.abs(sv))[::-1][:top_n]

    contributions = []
    for idx in indices:
        val = float(sv[idx])
        contributions.append(
            {
                "feature": feature_names[idx],
                "contribution": abs(val),
                "direction": "positive" if val >= 0 else "negative",
                "raw_value": float(feature_vector[0, idx]),
            }
        )

    return contributions


def build_attribution_report(metric, predicted, actual, contributions):
    """Build a structured attribution report for a single metric prediction.

    Args:
        metric: Name of the predicted metric (e.g. "power_watts").
        predicted: The model's predicted value.
        actual: The observed actual value.
        contributions: List of contribution dicts from explain_prediction().

    Returns:
        Dict with keys: metric, predicted, actual, delta, top_drivers.
    """
    return {
        "metric": metric,
        "predicted": predicted,
        "actual": actual,
        "delta": actual - predicted,
        "top_drivers": contributions,
    }
