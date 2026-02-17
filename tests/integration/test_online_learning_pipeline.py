"""Integration tests for the full online learning pipeline."""

from aria.engine.hardware import HardwareProfile, recommend_tier
from aria.engine.online import OnlineModelWrapper
from aria.engine.weight_tuner import EnsembleWeightTuner


class TestOnlineLearningPipeline:
    def test_full_online_learning_cycle(self):
        """Simulate: learn from outcomes -> predict -> track accuracy -> tune weights."""
        # 1. Verify tier gate
        profile = HardwareProfile(ram_gb=32.0, cpu_cores=8, gpu_available=False)
        assert recommend_tier(profile) >= 3

        # 2. Create online model
        model = OnlineModelWrapper(target="power_watts", min_samples=3)

        # 3. Feed observations (simulating shadow resolutions)
        observations = [
            ({"hour_sin": 0.5, "temp_f": 65.0}, 500.0),
            ({"hour_sin": 0.7, "temp_f": 70.0}, 550.0),
            ({"hour_sin": 0.9, "temp_f": 60.0}, 480.0),
            ({"hour_sin": 0.3, "temp_f": 72.0}, 520.0),
        ]
        for features, actual in observations:
            model.learn_one(features, actual)

        # 4. Verify predictions available
        pred = model.predict_one({"hour_sin": 0.6, "temp_f": 67.0})
        assert pred is not None
        assert 400 < pred < 700  # Reasonable range

        # 5. Weight tuner tracks accuracy
        tuner = EnsembleWeightTuner(window_days=7)
        tuner.record("batch_gb", prediction=510.0, actual=500.0)  # MAE 10
        tuner.record("batch_lgbm", prediction=505.0, actual=500.0)  # MAE 5
        tuner.record("online_arf", prediction=502.0, actual=500.0)  # MAE 2
        weights = tuner.compute_weights()

        # Online model should get highest weight (lowest MAE)
        assert weights["online_arf"] > weights["batch_gb"]
        assert weights["online_arf"] > weights["batch_lgbm"]

    def test_cold_start_solved(self):
        """Online model produces predictions much sooner than batch."""
        model = OnlineModelWrapper(target="power_watts", min_samples=3)
        # After just 3 observations, online model is ready
        for i in range(3):
            model.learn_one({"hour_sin": 0.5 + i * 0.1}, y=500.0 + i * 10)
        assert model.predict_one({"hour_sin": 0.6}) is not None
        # Batch model needs 30+ days of daily snapshots

    def test_tier_2_no_online(self):
        """Tier 2 hardware should not activate online learning."""
        profile = HardwareProfile(ram_gb=4.0, cpu_cores=2, gpu_available=False)
        assert recommend_tier(profile) == 2
        # Module self-gates -- would create models but not subscribe
