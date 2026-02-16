"""Reference model — unmodified baseline for feedback loop stability.

Maintains a model trained with the default feature config (never modified by
the meta-learner). Comparing its accuracy trend against the meta-tuned model
disambiguates two failure modes:

- Both degrade -> behavioral drift (user habits changed, seasonal shift)
- Only tuned model degrades -> meta-learner introduced a bad config change

Reference: Khritankov et al. (2024) "Hidden Feedback Loop" recommends
maintaining an unmodified control model alongside any self-tuning system.
Shumailov et al. (Nature 2024) shows recursive self-training causes model
collapse; the reference model detects this early.

Usage:
    ref = ReferenceModel(config.paths)
    ref.train(snapshots, default_config)  # Uses DEFAULT_FEATURE_CONFIG always
    comparison = ref.compare(tuned_mae)   # Returns drift diagnosis
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from aria.engine.config import PathConfig
from aria.engine.features.feature_config import DEFAULT_FEATURE_CONFIG


class ReferenceModel:
    """Unmodified reference model for meta-learner loop stability.

    Trains with DEFAULT_FEATURE_CONFIG regardless of what the meta-learner
    has tuned. Stores results in a separate directory to avoid contaminating
    the production model artifacts.

    Args:
        paths: PathConfig with data directories.
    """

    def __init__(self, paths: PathConfig):
        self._paths = paths
        self._ref_dir = paths.models_dir / "reference"
        self._history_path = self._ref_dir / "accuracy_history.json"

    def train(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """Train the reference model using default (unmodified) feature config.

        Args:
            snapshots: Training data snapshots (same data as production model).

        Returns:
            Training result dict with per-metric MAE, or error dict.
        """
        if len(snapshots) < 14:
            return {"error": f"insufficient data ({len(snapshots)} snapshots)"}

        try:
            from aria.engine.features.vector_builder import build_training_data
            from aria.engine.models.training import train_continuous_model
        except ImportError:
            return {"error": "sklearn not available"}

        self._ref_dir.mkdir(parents=True, exist_ok=True)
        config = DEFAULT_FEATURE_CONFIG.copy()
        feature_names, X, targets = build_training_data(snapshots, config)

        if len(X) < 14:
            return {"error": "insufficient training vectors"}

        results = {"trained_at": datetime.now().isoformat(), "metrics": {}}

        for metric in config.get("target_metrics", []):
            if metric not in targets:
                continue

            tmpdir = tempfile.mkdtemp()
            try:
                result = train_continuous_model(metric, feature_names, X, targets[metric], tmpdir)
                if "error" not in result:
                    results["metrics"][metric] = {
                        "mae": result.get("mae"),
                        "r2": result.get("r2"),
                    }
                    # Copy model artifact to reference dir
                    src = Path(tmpdir)
                    for f in src.glob("*.joblib"):
                        dest = self._ref_dir / f"ref_{metric}.joblib"
                        shutil.copy2(str(f), str(dest))
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Append to accuracy history
        self._append_history(results)
        return results

    def compare(self, tuned_mae: dict[str, float]) -> dict[str, Any]:
        """Compare tuned model MAE against reference model MAE.

        Args:
            tuned_mae: Dict mapping metric name to the tuned model's MAE.

        Returns:
            Dict with diagnosis per metric:
            - "behavioral_drift": both models degraded
            - "meta_learner_error": only tuned model degraded
            - "meta_learner_helping": tuned model improved, reference didn't
            - "stable": neither degraded significantly
        """
        history = self._load_history()
        if not history:
            return {"diagnosis": "no_reference_data"}

        latest = history[-1]
        ref_metrics = latest.get("metrics", {})

        diagnosis = {}
        for metric, tuned_val in tuned_mae.items():
            ref_data = ref_metrics.get(metric, {})
            ref_val = ref_data.get("mae")
            if ref_val is None:
                diagnosis[metric] = "no_reference_baseline"
                continue

            # Check if reference model is also degrading
            # by comparing current vs historical reference MAE
            ref_history_mae = self._get_metric_trend(metric)
            if len(ref_history_mae) < 2:
                diagnosis[metric] = "insufficient_history"
                continue

            ref_recent = sum(ref_history_mae[-3:]) / len(ref_history_mae[-3:])
            ref_earlier = sum(ref_history_mae[:-3]) / max(len(ref_history_mae[:-3]), 1)

            ref_degraded = ref_recent > ref_earlier * 1.05  # >5% worse
            tuned_degraded = tuned_val > ref_val * 1.05

            if ref_degraded and tuned_degraded:
                diagnosis[metric] = "behavioral_drift"
            elif tuned_degraded and not ref_degraded:
                diagnosis[metric] = "meta_learner_error"
            elif not tuned_degraded and ref_degraded:
                diagnosis[metric] = "meta_learner_helping"
            else:
                diagnosis[metric] = "stable"

        return {"diagnosis": diagnosis, "reference_mae": {m: d.get("mae") for m, d in ref_metrics.items()}}

    def _append_history(self, result: dict[str, Any]):
        """Append a training result to the accuracy history file."""
        history = self._load_history()
        history.append(result)
        # Keep last 90 entries
        history = history[-90:]
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)

    def _load_history(self) -> list[dict[str, Any]]:
        """Load accuracy history from disk."""
        if not self._history_path.is_file():
            return []
        try:
            with open(self._history_path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _get_metric_trend(self, metric: str) -> list[float]:
        """Extract MAE trend for a specific metric from history."""
        history = self._load_history()
        values = []
        for entry in history:
            mae = entry.get("metrics", {}).get(metric, {}).get("mae")
            if mae is not None:
                values.append(mae)
        return values
