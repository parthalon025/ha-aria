"""Integration tests for Phase 3 pattern recognition pipeline.

Tests the full flow:
  shadow_resolved event -> pattern recognition window -> trajectory classification
  anomaly detection -> explainer -> top-3 features in prediction output
  feature config -> trajectory_class in feature vector
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from aria.engine.anomaly_explainer import AnomalyExplainer
from aria.engine.pattern_scale import PatternScale
from aria.engine.sequence import SequenceClassifier
from aria.modules.trajectory_classifier import TrajectoryClassifier


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.get_config_value = MagicMock(return_value=None)
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value=None)
    hub.modules = {}
    return hub


class TestPatternRecognitionPipeline:
    """End-to-end pattern recognition tests."""

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_full_pipeline(self, mock_scan, mock_tier, mock_hub):
        """Shadow events -> trajectory classification -> cache update."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)

        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is True

        # Feed 6 events with increasing power (ramping up)
        for i in range(6):
            await module._on_shadow_resolved(
                {
                    "target": "power_watts",
                    "features": {
                        "activity": float(10 + i * 20),
                        "lights": 2.0,
                        "motion": 1.0,
                    },
                    "actual_value": float(10 + i * 20),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Should have classified trajectory
        assert module.current_trajectory == "ramping_up"
        # Cache should have been updated
        mock_hub.set_cache.assert_called()

    async def test_anomaly_explanation_pipeline(self):
        """IsolationForest -> explainer -> top features."""
        np.random.seed(42)
        X_train = np.random.normal(0, 1, (200, 5))
        model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        model.fit(X_train)

        # Extreme anomaly in feature 0
        X_anomaly = np.array([[15.0, 0.0, 0.0, 0.0, 0.0]])
        features = ["power", "lights", "motion", "temp", "humidity"]

        explainer = AnomalyExplainer()
        explanations = explainer.explain(model, X_anomaly, features, top_n=3)

        assert len(explanations) == 3
        assert all(e["contribution"] > 0 for e in explanations)
        # The extreme feature should rank high
        top_features = [e["feature"] for e in explanations]
        assert "power" in top_features

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=2)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_tier_2_gates_out(self, mock_scan, mock_tier, mock_hub):
        """Tier 2 hardware disables pattern recognition."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)

        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is False

        # Events should be ignored
        await module._on_shadow_resolved(
            {
                "target": "power_watts",
                "features": {"power": 100.0},
                "actual_value": 100.0,
                "timestamp": datetime.now().isoformat(),
            }
        )
        assert module.current_trajectory is None

    def test_pattern_scale_classification(self):
        """Patterns classify into correct time scales."""
        assert PatternScale.from_duration_seconds(30) == PatternScale.MICRO
        assert PatternScale.from_duration_seconds(1800) == PatternScale.MESO
        assert PatternScale.from_duration_seconds(86400) == PatternScale.MACRO

    def test_sequence_heuristic_labels(self):
        """Heuristic labeling produces correct trajectory classes."""
        # Ramping up
        window = np.zeros((6, 3))
        window[:, 0] = np.linspace(10, 80, 6)
        assert SequenceClassifier.label_window_heuristic(window) == "ramping_up"

        # Winding down
        window[:, 0] = np.linspace(80, 10, 6)
        assert SequenceClassifier.label_window_heuristic(window) == "winding_down"

        # Stable
        window[:, 0] = [50, 51, 49, 50, 51, 50]
        assert SequenceClassifier.label_window_heuristic(window) == "stable"
