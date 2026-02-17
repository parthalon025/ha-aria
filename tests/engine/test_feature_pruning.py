from aria.engine.features.pruning import FeaturePruner


class TestFeaturePruner:
    def test_identify_low_importance_features(self):
        importances = {
            "hour_sin": 0.30,
            "hour_cos": 0.25,
            "temp_f": 0.20,
            "humidity_pct": 0.15,
            "wind_mph": 0.05,
            "is_weekend_x_temp": 0.004,
            "daylight_x_lights": 0.001,
        }
        pruner = FeaturePruner(threshold=0.01)
        low = pruner.identify_low_importance(importances)
        assert "daylight_x_lights" in low
        assert "is_weekend_x_temp" in low
        assert "hour_sin" not in low

    def test_track_consecutive_low_cycles(self):
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        assert pruner.should_prune("feat_a") is False  # only 1 cycle
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        assert pruner.should_prune("feat_a") is False  # only 2 cycles
        pruner.record_cycle(low_features={"feat_a"})
        assert pruner.should_prune("feat_a") is True  # 3 consecutive
        assert pruner.should_prune("feat_b") is False  # reset (not in 3rd cycle)

    def test_drift_resets_pruning(self):
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle(low_features={"feat_a"})
        assert pruner.should_prune("feat_a") is True
        pruner.on_drift_detected()
        assert pruner.should_prune("feat_a") is False

    def test_get_active_features(self):
        all_features = ["hour_sin", "hour_cos", "wind_mph"]
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        for _ in range(3):
            pruner.record_cycle(low_features={"wind_mph"})
        active = pruner.get_active_features(all_features)
        assert "hour_sin" in active
        assert "wind_mph" not in active

    def test_to_dict_serialization(self):
        pruner = FeaturePruner(threshold=0.01, required_cycles=3)
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        pruner.record_cycle(low_features={"feat_a", "feat_b"})
        result = pruner.to_dict()
        assert "consecutive_low" in result
        assert result["consecutive_low"]["feat_a"] == 2
        assert result["pruned"] == []
