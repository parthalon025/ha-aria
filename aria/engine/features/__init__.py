"""Feature engineering — time encoding, vector building, feature config."""

from .feature_config import (
    DEFAULT_FEATURE_CONFIG,
    load_feature_config,
    save_feature_config,
    validate_feature_config,
)
from .time_features import build_time_features, cyclical_encode
from .vector_builder import (
    build_feature_vector,
    build_training_data,
    extract_target_values,
    get_feature_names,
)

__all__ = [
    "DEFAULT_FEATURE_CONFIG",
    "build_feature_vector",
    "build_time_features",
    "build_training_data",
    "cyclical_encode",
    "extract_target_values",
    "get_feature_names",
    "load_feature_config",
    "save_feature_config",
    "validate_feature_config",
]
