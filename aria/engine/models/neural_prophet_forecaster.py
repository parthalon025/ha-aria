"""NeuralProphet forecaster — deep learning seasonal forecasting with Prophet fallback.

NeuralProphet extends Facebook Prophet with PyTorch-based neural network components,
enabling autoregression (AR-Net) and lagged regressors while keeping the decomposable
trend + seasonality structure. Falls back to Prophet when neuralprophet is not installed.

Same metric interface as prophet_forecaster.py — operates on daily-frequency snapshots
and produces next-day forecasts with confidence intervals.
"""

import logging
import os
import pickle

logger = logging.getLogger(__name__)

HAS_NEURAL_PROPHET = True
try:
    import numpy as np
    import pandas as pd
    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")
except ImportError:
    HAS_NEURAL_PROPHET = False

# Metrics suitable for NeuralProphet forecasting (same as Prophet)
NEURALPROPHET_METRICS = ["power_watts", "lights_on", "devices_home", "unavailable"]


class NeuralProphetForecaster:
    """NeuralProphet model for seasonal time series forecasting with autoregression."""

    def train(  # noqa: PLR0913 — NeuralProphet training requires metric, data, dir, and hyperparameters
        self,
        metric_name,
        daily_snapshots,
        model_dir,
        epochs=100,
        learning_rate=0.1,
        ar_order=7,
    ):
        """Train a NeuralProphet model on daily snapshot time series.

        Args:
            metric_name: Which metric to forecast (e.g., "power_watts").
            daily_snapshots: List of (date_str, snapshot_dict) tuples sorted by date.
            model_dir: Directory to save the trained model.
            epochs: Number of training epochs (default 100).
            learning_rate: Learning rate for optimizer (default 0.1).
            ar_order: Number of autoregression lags (default 7 = one week).

        Returns:
            Training results dict with components and diagnostics.
        """
        if not HAS_NEURAL_PROPHET:
            return {"error": "neuralprophet not installed"}

        # Build dataframe: ds (date), y (metric value)
        rows = []
        for date_str, snap in daily_snapshots:
            value = self._extract_metric(snap, metric_name)
            if value is not None:
                rows.append({"ds": date_str, "y": value})

        if len(rows) < 14:
            return {"error": f"insufficient data ({len(rows)} days, need 14+)"}

        df = pd.DataFrame(rows)
        df["ds"] = pd.to_datetime(df["ds"])

        # Configure NeuralProphet
        model = NeuralProphet(
            n_forecasts=1,
            n_lags=ar_order,
            yearly_seasonality=len(rows) >= 60,
            weekly_seasonality=True,
            daily_seasonality=False,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=min(32, len(rows)),
        )

        metrics_df = model.fit(df, freq="D")

        # In-sample diagnostics
        forecast = model.predict(df)
        mae, mape = self._compute_diagnostics(forecast)

        # Save model + training data (needed for make_future_dataframe at predict time)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"neuralprophet_{metric_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "df": df}, f)

        components = self._extract_components(forecast)
        final_loss = self._extract_final_loss(metrics_df)

        result = {
            "metric": metric_name,
            "mae": round(mae, 2),
            "mape": round(mape, 1),
            "training_days": len(rows),
            "epochs": epochs,
            "ar_order": ar_order,
            "components": components,
            "backend": "neuralprophet",
        }
        if final_loss is not None:
            result["final_loss"] = final_loss

        return result

    @staticmethod
    def _compute_diagnostics(forecast):
        """Compute MAE and MAPE from in-sample forecast."""
        valid = forecast[["y", "yhat1"]].dropna()
        residuals = valid["y"].values - valid["yhat1"].values
        y_vals = valid["y"].values
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / np.where(y_vals != 0, y_vals, 1)))) * 100
        return mae, mape

    @staticmethod
    def _extract_components(forecast):
        """Extract seasonal components from forecast for insight."""
        components = {}
        if "weekly" in forecast.columns:
            weekly = forecast["weekly"].dropna().tolist()
            if weekly:
                components["weekly_range"] = round(float(max(weekly) - min(weekly)), 2)
        if "trend" in forecast.columns:
            trend = forecast["trend"].dropna()
            if len(trend) >= 2:
                components["trend_start"] = round(float(trend.iloc[0]), 2)
                components["trend_end"] = round(float(trend.iloc[-1]), 2)
        return components

    @staticmethod
    def _extract_final_loss(metrics_df):
        """Extract final training loss from metrics dataframe."""
        if metrics_df is not None and len(metrics_df) > 0:
            loss_cols = [c for c in metrics_df.columns if "Loss" in c or "loss" in c]
            if loss_cols:
                return round(float(metrics_df[loss_cols[0]].iloc[-1]), 4)
        return None

    def predict(self, metric_name, model_dir, horizon_days=1):
        """Forecast the next N days using a trained NeuralProphet model.

        Args:
            metric_name: Which metric to forecast.
            model_dir: Directory containing saved model.
            horizon_days: How many days ahead to forecast (default 1 = tomorrow).

        Returns:
            Dict with predicted value and components. None if model not available.
        """
        if not HAS_NEURAL_PROPHET:
            return None

        model_path = os.path.join(model_dir, f"neuralprophet_{metric_name}.pkl")
        if not os.path.isfile(model_path):
            return None

        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        # Support both new format (dict with model + df) and legacy (model only)
        if isinstance(saved, dict):
            model = saved["model"]
            history_df = saved["df"]
        else:
            model = saved
            history_df = None

        if history_df is None:
            logger.warning("No training data saved with model for %s", metric_name)
            return None

        # Standard NeuralProphet pattern: extend history with future periods, predict once
        future = model.make_future_dataframe(history_df, periods=horizon_days)
        forecast = model.predict(future)

        # Get the last row (next-day prediction)
        last = forecast.iloc[-1]
        yhat_col = "yhat1"

        result = {
            "predicted": round(float(last[yhat_col]), 1),
        }

        # Add trend/weekly if available
        if "trend" in forecast.columns:
            result["trend"] = round(float(last.get("trend", 0)), 1)
        if "weekly" in forecast.columns:
            result["weekly"] = round(float(last.get("weekly", 0)), 1)

        return result

    @staticmethod
    def _extract_metric(snapshot, metric_name):
        """Extract a metric value from a snapshot dict.

        Handles the nested snapshot structure where metrics live in
        sub-dicts like snapshot["power"]["total_watts"].
        """
        METRIC_PATHS = {
            "power_watts": ("power", "total_watts"),
            "lights_on": ("lights", "on"),
            "devices_home": ("occupancy", "device_count_home"),
            "unavailable": ("entities", "unavailable"),
        }

        path = METRIC_PATHS.get(metric_name)
        if not path:
            return None

        value = snapshot
        for key in path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        try:
            return float(value)
        except (ValueError, TypeError):
            return None


def train_neuralprophet_models(daily_snapshots, model_dir, epochs=100, learning_rate=0.1, ar_order=7):
    """Train NeuralProphet models for all supported metrics.

    Args:
        daily_snapshots: List of (date_str, snapshot_dict) tuples.
        model_dir: Directory to save models.
        epochs: Training epochs per metric.
        learning_rate: Optimizer learning rate.
        ar_order: Autoregression lag order.

    Returns:
        Dict of metric -> training result.
    """
    if not HAS_NEURAL_PROPHET:
        return {"error": "neuralprophet not installed"}

    forecaster = NeuralProphetForecaster()
    results = {}

    for metric in NEURALPROPHET_METRICS:
        result = forecaster.train(metric, daily_snapshots, model_dir, epochs, learning_rate, ar_order)
        results[metric] = result
        if "error" not in result:
            logger.info(
                "NeuralProphet %s: MAE=%s, MAPE=%s%%",
                metric,
                result["mae"],
                result["mape"],
            )
        else:
            logger.info("NeuralProphet %s: %s", metric, result["error"])

    return results


def predict_with_neuralprophet(model_dir, horizon_days=1):
    """Generate NeuralProphet forecasts for all available models.

    Returns dict of metric -> forecast value (or empty if no models).
    """
    if not HAS_NEURAL_PROPHET:
        return {}

    forecaster = NeuralProphetForecaster()
    predictions = {}

    for metric in NEURALPROPHET_METRICS:
        result = forecaster.predict(metric, model_dir, horizon_days)
        if result is not None:
            predictions[metric] = result["predicted"]

    return predictions
