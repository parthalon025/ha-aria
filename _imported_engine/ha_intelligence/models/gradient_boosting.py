"""Gradient Boosting model for continuous metric prediction."""

import os
import pickle

from ha_intelligence.config import ModelConfig
from ha_intelligence.models.registry import ModelRegistry, BaseModel

HAS_SKLEARN = True
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
except ImportError:
    HAS_SKLEARN = False


@ModelRegistry.register("gradient_boosting")
class GradientBoostingModel(BaseModel):
    """GradientBoosting regressor for continuous Home Assistant metrics."""

    def train(self, metric_name, feature_names, X, y, model_dir, config=None):
        """Train a GradientBoosting model for a continuous metric.

        Returns training metrics dict.
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed"}

        if config is None:
            config = ModelConfig()

        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        if len(X_arr) < config.min_training_samples:
            return {"error": f"insufficient data ({len(X_arr)} samples, need {config.min_training_samples}+)"}

        # 80/20 chronological split (no shuffle — time series)
        split = int(len(X_arr) * config.validation_split)
        X_train, X_val = X_arr[:split], X_arr[split:]
        y_train, y_val = y_arr[:split], y_arr[split:]

        model = GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            min_samples_leaf=max(3, len(X_train) // config.min_samples_ratio),
            subsample=config.subsample,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred) if len(y_val) > 1 else 0.0

        # Feature importance
        importances = dict(zip(feature_names, [round(v, 4) for v in model.feature_importances_]))

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{metric_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return {
            "metric": metric_name,
            "mae": round(float(mae), 2),
            "r2": round(float(r2), 4),
            "samples_train": len(X_train),
            "samples_val": len(X_val),
            "feature_importance": importances,
        }

    def predict(self, feature_vector, model_path):
        """Load a saved model and predict a single value.

        Returns predicted float or None if model unavailable.
        """
        if not HAS_SKLEARN:
            return None

        if not os.path.isfile(model_path):
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = np.array([feature_vector], dtype=float)
        return float(model.predict(X)[0])
