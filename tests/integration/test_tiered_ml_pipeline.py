"""Integration tests for the tiered ML pipeline (Phase 1).

Exercises the complete pipeline: hardware scan → tier → registry → CV → pruning → fallback.
"""

import numpy as np

from aria.engine.evaluation import expanding_window_cv
from aria.engine.fallback import FallbackTracker
from aria.engine.features.pruning import FeaturePruner
from aria.engine.hardware import HardwareProfile, recommend_tier
from aria.engine.models.registry import TieredModelRegistry


class TestTieredPipelineIntegration:
    def test_full_tier_2_pipeline(self):
        """Simulate complete Tier 2 training pipeline."""
        # 1. Hardware scan → tier recommendation
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        tier = recommend_tier(profile)
        assert tier == 2

        # 2. Registry resolves Tier 2 models
        registry = TieredModelRegistry.with_defaults()
        entries = registry.resolve("power_watts", tier)
        entry_names = {e.name for e in entries}
        assert "gb" in entry_names
        assert "rf" in entry_names
        # lgbm/lgbm_lite present only when lightgbm is installed
        assert len(entries) >= 2

        # 3. CV produces correct fold count for tier
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        n_folds = 3  # Tier 2
        folds = list(expanding_window_cv(X, y, n_folds=n_folds))
        assert len(folds) == 3

        # 4. Feature pruner doesn't activate at Tier 2
        pruner = FeaturePruner(threshold=0.01, required_cycles=999)
        pruner.record_cycle({"feat_a"})
        assert not pruner.should_prune("feat_a")

        # 5. Fallback tracker is ready
        tracker = FallbackTracker(ttl_days=7)
        assert len(tracker.active_fallbacks()) == 0

    def test_full_tier_3_pipeline(self):
        """Simulate Tier 3 pipeline with pruning and more CV folds."""
        profile = HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False)
        tier = recommend_tier(profile)
        assert tier == 3

        registry = TieredModelRegistry.with_defaults()
        entries = registry.resolve("power_watts", tier)
        entry_names = {e.name for e in entries}
        assert "gb" in entry_names
        assert "rf" in entry_names
        assert len(entries) >= 2

        # CV with 5 folds at Tier 3
        X = np.random.randn(60, 5)
        y = np.random.randn(60)
        folds = list(expanding_window_cv(X, y, n_folds=5))
        assert len(folds) == 5

        # Feature pruner active at Tier 3
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle({"low_feat"})
        assert pruner.should_prune("low_feat")

    def test_fallback_degrades_gracefully(self):
        """Model failure at Tier 4 should fall back to Tier 3 behavior."""
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("transformer_power", from_tier=4, to_tier=3, error="CUDA OOM")
        assert tracker.get_effective_tier("transformer_power", original_tier=4) == 3
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 3  # unaffected
