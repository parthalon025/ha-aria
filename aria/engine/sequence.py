"""Trajectory sequence classifier using Dynamic Time Warping.

Classifies sliding windows of snapshot feature vectors into trajectory
types using tslearn's KNeighborsTimeSeriesClassifier with DTW metric.

Training labels are generated heuristically from the power metric trend,
then the DTW classifier generalizes across the full multi-variate space.

Tier 3+ only — tslearn is an optional dependency.
"""

import logging

import numpy as np

from aria.shared.constants import TRAJECTORY_CLASSES  # noqa: F401

logger = logging.getLogger(__name__)

# Thresholds for heuristic labeling
_CHANGE_THRESHOLD = 0.20  # 20% change = directional
_ANOMALOUS_CV_THRESHOLD = 0.50  # Coefficient of variation > 50% = anomalous


class SequenceClassifier:
    """DTW-based trajectory classifier for snapshot windows.

    Args:
        window_size: Number of snapshots per window (default 6).
        n_neighbors: KNN neighbors for DTW classifier (default 3).
    """

    def __init__(self, window_size: int = 6, n_neighbors: int = 3):
        self.window_size = window_size
        self.n_neighbors = n_neighbors
        self._model = None
        self._tslearn_available = True
        self._trained_at: str | None = None
        self._n_training_samples: int = 0

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been trained."""
        return self._model is not None

    def _create_model(self):
        """Lazily create the tslearn KNN-DTW model."""
        try:
            from tslearn.neighbors import KNeighborsTimeSeriesClassifier

            return KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, metric="dtw")
        except ImportError:
            logger.warning("tslearn not installed — sequence classifier disabled")
            self._tslearn_available = False
            return None

    def train(self, windows: np.ndarray, labels: list[str]) -> bool:
        """Train on labeled windows.

        Args:
            windows: Array of shape (n_samples, window_size, n_features).
            labels: List of trajectory class labels.

        Returns:
            True if training succeeded, False otherwise.
        """
        if not self._tslearn_available:
            return False

        model = self._create_model()
        if model is None:
            return False

        try:
            model.fit(windows, labels)
            self._model = model
            self._n_training_samples = len(labels)
            from datetime import datetime

            self._trained_at = datetime.now().isoformat()
            logger.info(f"Sequence classifier trained on {len(labels)} windows")
            return True
        except Exception as e:
            logger.error(f"Sequence classifier training failed: {e}")
            return False

    def predict(self, window: np.ndarray) -> str | None:
        """Classify a single window.

        Args:
            window: Array of shape (window_size, n_features).

        Returns:
            Trajectory class string, or None if the model is untrained,
            tslearn is unavailable, or prediction fails.
        """
        if self._model is None or not self._tslearn_available:
            return None

        try:
            result = self._model.predict(window.reshape(1, *window.shape))
            return str(result[0])
        except Exception as e:
            logger.debug(f"Sequence prediction failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Return classifier statistics."""
        return {
            "is_trained": self.is_trained,
            "window_size": self.window_size,
            "n_neighbors": self.n_neighbors,
            "tslearn_available": self._tslearn_available,
            "trained_at": self._trained_at,
            "n_training_samples": self._n_training_samples,
        }

    @staticmethod
    def label_window_heuristic(window: np.ndarray, target_col_idx: int = 0) -> str:
        """Label a window using heuristic rules on the target metric.

        Compares the average of the first two and last two values in
        the target column to determine trend direction. High variance
        with no clear direction = anomalous_transition.

        Args:
            window: Array of shape (window_size, n_features).
            target_col_idx: Column index of the target metric (e.g., power).

        Returns:
            One of TRAJECTORY_CLASSES.
        """
        values = window[:, target_col_idx]
        start_avg = float(np.mean(values[:2]))
        end_avg = float(np.mean(values[-2:]))

        # Avoid division by zero
        baseline = max(abs(start_avg), 1e-6)
        change_pct = (end_avg - start_avg) / baseline

        # Check for anomalous variance first
        mean_abs = max(float(np.mean(np.abs(values))), 1e-6)
        cv = float(np.std(values)) / mean_abs
        if cv > _ANOMALOUS_CV_THRESHOLD and abs(change_pct) < _CHANGE_THRESHOLD:
            return "anomalous_transition"

        if change_pct > _CHANGE_THRESHOLD:
            return "ramping_up"
        elif change_pct < -_CHANGE_THRESHOLD:
            return "winding_down"

        return "stable"
