"""Shared TypedDict schema definitions for engine-hub JSON contract.

Defines the structure that _read_intelligence_data() in intelligence.py
expects from the assembled cache payload. Used for runtime validation
(warn on missing keys) and contract testing.
"""

from typing import Any, TypedDict


class IntelligencePayload(TypedDict, total=False):
    """Top-level keys returned by IntelligenceModule._read_intelligence_data().

    All keys are optional (total=False) because some only appear after
    enough data has accumulated. The REQUIRED_INTELLIGENCE_KEYS set below
    defines which keys MUST always be present for the hub to function.
    """

    data_maturity: dict[str, Any]
    predictions: dict[str, Any] | None
    baselines: dict[str, Any] | None
    trend_data: list[dict[str, Any]]
    intraday_trend: list[dict[str, Any]]
    daily_insight: dict[str, Any] | None
    accuracy: dict[str, Any] | None
    correlations: list[Any]
    ml_models: dict[str, Any]
    meta_learning: dict[str, Any]
    run_log: list[dict[str, Any]]
    config: dict[str, Any]
    entity_correlations: dict[str, Any] | None
    sequence_anomalies: dict[str, Any] | None
    power_profiles: dict[str, Any] | None
    automation_suggestions: dict[str, Any] | None
    drift_status: dict[str, Any] | None
    feature_selection: dict[str, Any] | None
    reference_model: dict[str, Any] | None
    shap_attributions: dict[str, Any] | None
    autoencoder_status: dict[str, Any] | None
    isolation_forest_status: dict[str, Any] | None


# Keys that _read_intelligence_data() always sets (never conditionally omitted).
# These are structural keys the hub depends on for cache assembly and API responses.
REQUIRED_INTELLIGENCE_KEYS: set[str] = {
    "data_maturity",
    "predictions",
    "baselines",
    "trend_data",
    "intraday_trend",
    "daily_insight",
    "accuracy",
    "correlations",
    "ml_models",
    "meta_learning",
    "run_log",
    "config",
    "entity_correlations",
    "sequence_anomalies",
    "power_profiles",
    "automation_suggestions",
    "drift_status",
    "feature_selection",
    "reference_model",
    "shap_attributions",
    "autoencoder_status",
    "isolation_forest_status",
}


def validate_intelligence_payload(data: dict[str, Any]) -> list[str]:
    """Validate that intelligence payload contains all required keys.

    Returns list of missing key names. Empty list means valid.
    Does NOT raise — callers decide whether to warn or error.
    """
    missing = REQUIRED_INTELLIGENCE_KEYS - set(data.keys())
    return sorted(missing)
