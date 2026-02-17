"""Tests for River-based online learning model wrapper."""

from aria.engine.online import OnlineModelWrapper


class TestOnlineModelWrapper:
    def test_learn_one_and_predict(self):
        model = OnlineModelWrapper(target="power_watts", min_samples=1)
        features = {"hour_sin": 0.5, "hour_cos": 0.87, "temp_f": 65.0}
        model.learn_one(features, y=500.0)
        pred = model.predict_one(features)
        assert isinstance(pred, float)

    def test_predict_before_learning_returns_none(self):
        model = OnlineModelWrapper(target="power_watts")
        features = {"hour_sin": 0.5, "hour_cos": 0.87}
        pred = model.predict_one(features)
        assert pred is None

    def test_predict_after_min_samples(self):
        """Need at least 5 samples before producing predictions."""
        model = OnlineModelWrapper(target="power_watts", min_samples=5)
        features = {"hour_sin": 0.5, "temp_f": 65.0}
        for i in range(4):
            model.learn_one(features, y=500.0 + i * 10)
        assert model.predict_one(features) is None
        model.learn_one(features, y=540.0)
        assert model.predict_one(features) is not None

    def test_get_stats(self):
        model = OnlineModelWrapper(target="power_watts")
        features = {"hour_sin": 0.5}
        model.learn_one(features, y=500.0)
        stats = model.get_stats()
        assert stats["target"] == "power_watts"
        assert stats["samples_seen"] == 1
        assert "model_type" in stats

    def test_reset_clears_state(self):
        model = OnlineModelWrapper(target="power_watts")
        model.learn_one({"hour_sin": 0.5}, y=500.0)
        assert model.samples_seen == 1
        model.reset()
        assert model.samples_seen == 0

    def test_feature_filtering(self):
        """Online model should handle missing features gracefully."""
        model = OnlineModelWrapper(target="power_watts")
        model.learn_one({"hour_sin": 0.5, "temp_f": 65.0}, y=500.0)
        # Predict with subset of features — should not crash
        model.predict_one({"hour_sin": 0.5})
        # River handles missing features natively, so this should not crash
