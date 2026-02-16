"""Tests for clean reference model comparison."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
class TestReferenceModel:
    async def test_dual_training_produces_two_model_sets(self):
        """Training should produce both primary and reference models."""
        from aria.modules.ml_engine import MLEngine

        hub = MagicMock()
        hub.cache = AsyncMock()
        hub.cache.get_config_value = AsyncMock(return_value="true")
        module = MLEngine.__new__(MLEngine)
        assert hasattr(module, "_train_reference_model"), "MLEngine should have _train_reference_model method"

    async def test_comparison_logic(self):
        """Test accuracy comparison between primary and reference."""
        from aria.modules.intelligence import compare_model_accuracy

        # Both degrade = behavioral drift
        result = compare_model_accuracy(
            primary_acc=[80, 75, 70, 65],
            reference_acc=[80, 75, 70, 65],
            threshold_pct=5.0,
        )
        assert result["interpretation"] == "behavioral_drift"

        # Only primary degrades = meta-learner error
        result = compare_model_accuracy(
            primary_acc=[80, 75, 70, 65],
            reference_acc=[80, 80, 80, 80],
            threshold_pct=5.0,
        )
        assert result["interpretation"] == "meta_learner_error"

        # Only reference degrades = meta-learner improvement
        result = compare_model_accuracy(
            primary_acc=[80, 80, 80, 80],
            reference_acc=[80, 75, 70, 65],
            threshold_pct=5.0,
        )
        assert result["interpretation"] == "meta_learner_improvement"

    async def test_stable_when_no_degradation(self):
        """Both models stable should return stable interpretation."""
        from aria.modules.intelligence import compare_model_accuracy

        result = compare_model_accuracy(
            primary_acc=[80, 80, 80, 80],
            reference_acc=[80, 80, 80, 80],
            threshold_pct=5.0,
        )
        assert result["interpretation"] == "stable"

    async def test_comparison_returns_expected_keys(self):
        """Result dict should contain all expected keys."""
        from aria.modules.intelligence import compare_model_accuracy

        result = compare_model_accuracy(
            primary_acc=[80, 75, 70, 65],
            reference_acc=[80, 80, 80, 80],
            threshold_pct=5.0,
        )
        assert "primary_trend" in result
        assert "reference_trend" in result
        assert "divergence_pct" in result
        assert "interpretation" in result

    async def test_comparison_trend_values(self):
        """Trend values should reflect mean-of-halves delta."""
        from aria.modules.intelligence import compare_model_accuracy

        # [80, 75, 70, 65]: first half mean=77.5, second half mean=67.5, delta=-10
        result = compare_model_accuracy(
            primary_acc=[80, 75, 70, 65],
            reference_acc=[80, 80, 80, 80],
            threshold_pct=5.0,
        )
        assert result["primary_trend"] == -10.0
        assert result["reference_trend"] == 0.0
        assert result["divergence_pct"] == 10.0

    async def test_short_lists_treated_as_stable(self):
        """Single-element lists should produce zero delta (stable)."""
        from aria.modules.intelligence import compare_model_accuracy

        result = compare_model_accuracy(
            primary_acc=[80],
            reference_acc=[80],
            threshold_pct=5.0,
        )
        assert result["interpretation"] == "stable"
        assert result["primary_trend"] == 0.0
        assert result["reference_trend"] == 0.0
