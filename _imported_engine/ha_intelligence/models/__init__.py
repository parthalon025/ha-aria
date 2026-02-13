"""ML models — sklearn training, prediction, anomaly detection."""

from ha_intelligence.models.registry import ModelRegistry, BaseModel
from ha_intelligence.models.gradient_boosting import GradientBoostingModel
from ha_intelligence.models.isolation_forest import IsolationForestModel
from ha_intelligence.models.device_failure import (
    train_device_failure_model,
    predict_device_failures,
    detect_contextual_anomalies,
)
from ha_intelligence.models.training import (
    train_all_models,
    train_continuous_model,
    predict_with_ml,
    blend_predictions,
    count_days_of_data,
)

__all__ = [
    "ModelRegistry",
    "BaseModel",
    "GradientBoostingModel",
    "IsolationForestModel",
    "train_device_failure_model",
    "predict_device_failures",
    "detect_contextual_anomalies",
    "train_all_models",
    "train_continuous_model",
    "predict_with_ml",
    "blend_predictions",
    "count_days_of_data",
]
