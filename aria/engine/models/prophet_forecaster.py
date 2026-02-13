"""Prophet seasonal forecaster for time series decomposition.

Captures daily/weekly seasonality and holiday effects that GradientBoosting
misses. Prophet operates on daily-frequency data (one value per day) and
decomposes into trend + seasonal + residual components.

Blended with GradientBoosting output via the existing blend_predictions
mechanism — Prophet weight increases with data maturity.
"""

import os
import pickle

from aria.engine.models.registry import ModelRegistry, BaseModel

HAS_PROPHET = True
try:
    from prophet import Prophet
    import pandas as pd
    import numpy as np
except ImportError:
    HAS_PROPHET = False

# Metrics suitable for Prophet forecasting (daily frequency, continuous values)
PROPHET_METRICS = ["power_watts", "lights_on", "devices_home", "unavailable"]

# Suppress Prophet's verbose stdout
import logging  # noqa: E402
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


@ModelRegistry.register("prophet")
class ProphetForecaster(BaseModel):
    """Facebook Prophet model for seasonal time series forecasting."""

    def train(self, metric_name, daily_snapshots, model_dir, holidays_list=None):
        """Train a Prophet model on daily snapshot time series.

        Args:
            metric_name: Which metric to forecast (e.g., "power_watts").
            daily_snapshots: List of (date_str, snapshot_dict) tuples sorted by date.
            model_dir: Directory to save the trained model.
            holidays_list: Optional list of holiday dicts with ds and holiday keys.

        Returns:
            Training results dict with components and diagnostics.
        """
        if not HAS_PROPHET:
            return {"error": "prophet not installed"}

        # Build Prophet dataframe: ds (date), y (metric value)
        rows = []
        for date_str, snap in daily_snapshots:
            value = self._extract_metric(snap, metric_name)
            if value is not None:
                rows.append({"ds": date_str, "y": value})

        if len(rows) < 14:
            return {"error": f"insufficient data ({len(rows)} days, need 14+)"}

        df = pd.DataFrame(rows)
        df["ds"] = pd.to_datetime(df["ds"])

        # Configure Prophet
        model = Prophet(
            daily_seasonality=False,   # Daily snapshots — no sub-daily pattern
            weekly_seasonality=True,    # Strong weekly pattern expected
            yearly_seasonality=len(rows) >= 60,  # Only if enough data
            changepoint_prior_scale=0.05,  # Conservative trend changes
            seasonality_mode="additive",
        )

        # Add holidays if available
        if holidays_list:
            model.add_country_holidays(country_name="US")

        model.fit(df)

        # Generate in-sample diagnostics
        forecast = model.predict(df)
        residuals = df["y"].values - forecast["yhat"].values
        mae = float(np.mean(np.abs(residuals)))
        mape = float(np.mean(np.abs(residuals / np.where(df["y"].values != 0, df["y"].values, 1)))) * 100

        # Save model
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"prophet_{metric_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Extract seasonal components for insight
        components = {}
        if "weekly" in forecast.columns:
            weekly = forecast["weekly"].tolist()
            components["weekly_range"] = round(float(max(weekly) - min(weekly)), 2)
        components["trend_start"] = round(float(forecast["trend"].iloc[0]), 2)
        components["trend_end"] = round(float(forecast["trend"].iloc[-1]), 2)

        return {
            "metric": metric_name,
            "mae": round(mae, 2),
            "mape": round(mape, 1),
            "training_days": len(rows),
            "components": components,
        }

    def predict(self, metric_name, model_dir, horizon_days=1):
        """Forecast the next N days using a trained Prophet model.

        Args:
            metric_name: Which metric to forecast.
            model_dir: Directory containing saved model.
            horizon_days: How many days ahead to forecast (default 1 = tomorrow).

        Returns:
            Dict with predicted value, confidence interval, and components.
            None if model not available.
        """
        if not HAS_PROPHET:
            return None

        model_path = os.path.join(model_dir, f"prophet_{metric_name}.pkl")
        if not os.path.isfile(model_path):
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Get the last row (next-day prediction)
        last = forecast.iloc[-1]
        return {
            "predicted": round(float(last["yhat"]), 1),
            "lower": round(float(last["yhat_lower"]), 1),
            "upper": round(float(last["yhat_upper"]), 1),
            "trend": round(float(last["trend"]), 1),
            "weekly": round(float(last.get("weekly", 0)), 1),
        }

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


def train_prophet_models(daily_snapshots, model_dir, holidays_list=None):
    """Train Prophet models for all supported metrics.

    Args:
        daily_snapshots: List of (date_str, snapshot_dict) tuples.
        model_dir: Directory to save models.
        holidays_list: Optional holiday list for Prophet.

    Returns:
        Dict of metric -> training result.
    """
    if not HAS_PROPHET:
        return {"error": "prophet not installed"}

    forecaster = ProphetForecaster()
    results = {}

    for metric in PROPHET_METRICS:
        result = forecaster.train(metric, daily_snapshots, model_dir, holidays_list)
        results[metric] = result
        if "error" not in result:
            print(f"  Prophet {metric}: MAE={result['mae']}, MAPE={result['mape']}%")
        else:
            print(f"  Prophet {metric}: {result['error']}")

    return results


def predict_with_prophet(model_dir, horizon_days=1):
    """Generate Prophet forecasts for all available models.

    Returns dict of metric -> forecast dict (or empty if no models).
    """
    if not HAS_PROPHET:
        return {}

    forecaster = ProphetForecaster()
    predictions = {}

    for metric in PROPHET_METRICS:
        result = forecaster.predict(metric, model_dir, horizon_days)
        if result is not None:
            predictions[metric] = result["predicted"]

    return predictions
