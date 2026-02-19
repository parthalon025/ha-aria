"""Tests for trajectory sequence classifier."""

import numpy as np

from aria.engine.sequence import TRAJECTORY_CLASSES, SequenceClassifier


class TestSequenceClassifier:
    """Test DTW-based trajectory classification."""

    def test_trajectory_classes(self):
        """Four trajectory classes are defined."""
        assert set(TRAJECTORY_CLASSES) == {"stable", "ramping_up", "winding_down", "anomalous_transition"}

    def test_init_defaults(self):
        """Classifier initializes with sensible defaults."""
        clf = SequenceClassifier()
        assert clf.window_size == 6
        assert clf.n_neighbors == 3
        assert not clf.is_trained

    def test_predict_untrained_returns_none(self):
        """Untrained classifier returns None."""
        clf = SequenceClassifier(window_size=4)
        window = np.zeros((4, 5))
        assert clf.predict(window) is None

    def test_label_heuristic_ramping_up(self):
        """Increasing power trend labels as ramping_up."""
        # 6 snapshots, 5 features — first feature is power
        window = np.zeros((6, 5))
        window[:, 0] = [10, 20, 30, 40, 50, 60]  # Increasing power
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "ramping_up"

    def test_label_heuristic_winding_down(self):
        """Decreasing power trend labels as winding_down."""
        window = np.zeros((6, 5))
        window[:, 0] = [60, 50, 40, 30, 20, 10]  # Decreasing power
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "winding_down"

    def test_label_heuristic_stable(self):
        """Flat power trend labels as stable."""
        window = np.zeros((6, 5))
        window[:, 0] = [50, 51, 49, 50, 51, 50]  # Stable
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "stable"

    def test_label_heuristic_anomalous(self):
        """Wild oscillation labels as anomalous_transition."""
        window = np.zeros((6, 5))
        window[:, 0] = [10, 100, 5, 95, 10, 100]  # Extreme variance
        label = SequenceClassifier.label_window_heuristic(window, target_col_idx=0)
        assert label == "anomalous_transition"

    def test_train_and_predict(self):
        """Train on labeled windows, predict a new one."""
        clf = SequenceClassifier(window_size=4, n_neighbors=1)

        # Build training data: 10 windows of each class
        windows = []
        labels = []
        rng = np.random.RandomState(42)
        for _ in range(10):
            # Ramping up
            w = rng.normal(0, 0.1, (4, 3))
            w[:, 0] = np.linspace(10, 60, 4)
            windows.append(w)
            labels.append("ramping_up")
            # Winding down
            w = rng.normal(0, 0.1, (4, 3))
            w[:, 0] = np.linspace(60, 10, 4)
            windows.append(w)
            labels.append("winding_down")

        X = np.array(windows)
        clf.train(X, labels)
        assert clf.is_trained

        # Test ramping up window
        test_window = np.zeros((4, 3))
        test_window[:, 0] = np.linspace(10, 60, 4)
        pred = clf.predict(test_window)
        assert pred in TRAJECTORY_CLASSES

    def test_get_stats(self):
        """Stats dict includes training info."""
        clf = SequenceClassifier(window_size=4)
        stats = clf.get_stats()
        assert stats["is_trained"] is False
        assert stats["window_size"] == 4

    def test_tslearn_missing_graceful(self):
        """Graceful fallback when tslearn is not installed."""
        clf = SequenceClassifier(window_size=4)
        clf._tslearn_available = False
        window = np.zeros((4, 3))
        assert clf.predict(window) is None
