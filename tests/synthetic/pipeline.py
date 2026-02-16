"""PipelineRunner — orchestrates the full ARIA engine pipeline with synthetic data.

Runs save → baselines → features → train → predict → score in a temp directory,
using real ARIA engine code at every step.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from aria.engine.analysis.baselines import compute_baselines
from aria.engine.config import AppConfig, ModelConfig, PathConfig
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG
from aria.engine.features.vector_builder import build_training_data
from aria.engine.models.training import predict_with_ml, train_continuous_model
from aria.engine.predictions.predictor import generate_predictions
from aria.engine.predictions.scoring import score_all_predictions
from aria.engine.storage.data_store import DataStore


class PipelineRunner:
    """Orchestrate a full ARIA pipeline run against synthetic snapshots."""

    def __init__(self, snapshots: list[dict], data_dir):
        self.snapshots = snapshots
        self.paths = PathConfig(
            data_dir=data_dir,
            logbook_path=data_dir / "current.json",
        )
        self.paths.ensure_dirs()
        self.store = DataStore(self.paths)
        self.config = AppConfig(paths=self.paths, model=ModelConfig(min_training_samples=14))

        # State tracking across pipeline stages
        self._baselines: dict | None = None
        self._predictions: dict | None = None
        self._training_results: dict | None = None

    def save_snapshots(self) -> int:
        """Save all synthetic snapshots to the data store."""
        for snap in self.snapshots:
            self.store.save_snapshot(snap)
        return len(self.snapshots)

    def compute_baselines(self) -> dict:
        """Compute per-day-of-week baselines from saved snapshots."""
        baselines = compute_baselines(self.snapshots)
        self.store.save_baselines(baselines)
        self._baselines = baselines
        return baselines

    def build_training_data(self):
        """Build feature matrix and target arrays from snapshots.

        Returns (feature_names, X, targets).
        """
        return build_training_data(self.snapshots, DEFAULT_FEATURE_CONFIG)

    def train_models(self) -> dict:
        """Train GradientBoosting models for each target metric.

        Returns dict of metric_name -> training result.
        """
        feature_names, X, targets = self.build_training_data()
        models_dir = str(self.paths.models_dir)
        results = {}

        for metric_name, y_values in targets.items():
            if len(y_values) < self.config.model.min_training_samples:
                results[metric_name] = {"error": f"insufficient data ({len(y_values)} samples)"}
                continue
            result = train_continuous_model(
                metric_name,
                feature_names,
                X,
                y_values,
                models_dir,
                self.config.model,
            )
            results[metric_name] = result

        # Save feature config so predict_with_ml can load it
        self.store.save_feature_config(DEFAULT_FEATURE_CONFIG.copy())
        self._training_results = results
        return results

    def generate_predictions(self, target_date: str | None = None) -> dict:
        """Generate predictions for a target date.

        If target_date is None, predicts the day after the last snapshot.
        Uses ML predictions if models have been trained.
        """
        if self._baselines is None:
            self._baselines = self.store.load_baselines()

        # Default to day after last snapshot
        if target_date is None:
            last_date = self.snapshots[-1]["date"]
            dt = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
            target_date = dt.strftime("%Y-%m-%d")

        # Try ML predictions if models were trained
        ml_predictions = None
        if self._training_results and any("error" not in v for v in self._training_results.values()):
            # Use the last snapshot as input for ML prediction
            ml_predictions = predict_with_ml(
                self.snapshots[-1],
                config=DEFAULT_FEATURE_CONFIG,
                prev_snapshot=self.snapshots[-2] if len(self.snapshots) > 1 else None,
                models_dir=str(self.paths.models_dir),
                store=self.store,
            )

        predictions = generate_predictions(
            target_date=target_date,
            baselines=self._baselines,
            ml_predictions=ml_predictions or None,
            paths=self.paths,
        )

        self.store.save_predictions(predictions)
        self._predictions = predictions
        return predictions

    def score_predictions(self) -> dict:
        """Score predictions against the last snapshot as a proxy actual.

        Uses the last snapshot as ground truth since we don't have a real
        'next day' snapshot for scoring.
        """
        if self._predictions is None:
            self._predictions = self.store.load_predictions()

        actual_snapshot = self.snapshots[-1]
        return score_all_predictions(self._predictions, actual_snapshot)

    def run_full(self) -> dict:
        """Run the complete pipeline end-to-end.

        Returns a results dict with keys for each stage.
        """
        count = self.save_snapshots()
        baselines = self.compute_baselines()
        training = self.train_models()
        predictions = self.generate_predictions()
        scores = self.score_predictions()

        return {
            "snapshots_saved": count,
            "baselines": baselines,
            "training": training,
            "predictions": predictions,
            "scores": scores,
        }
