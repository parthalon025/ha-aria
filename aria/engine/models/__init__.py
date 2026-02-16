"""ML models — sklearn training, prediction, anomaly detection."""

from aria.engine.models.device_failure import (
    detect_contextual_anomalies,
    predict_device_failures,
    train_device_failure_model,
)
from aria.engine.models.gradient_boosting import GradientBoostingModel
from aria.engine.models.isolation_forest import IsolationForestModel
from aria.engine.models.registry import BaseModel, ModelRegistry
from aria.engine.models.training import (
    blend_predictions,
    count_days_of_data,
    predict_with_ml,
    train_all_models,
    train_continuous_model,
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
