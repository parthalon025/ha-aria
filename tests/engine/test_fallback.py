from datetime import datetime, timedelta

from aria.engine.fallback import FallbackTracker


class TestFallbackTracker:
    def test_record_fallback(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="MemoryError", memory_mb=25600)
        assert tracker.is_fallen_back("lgbm_power")

    def test_fallback_expires(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        # Simulate expiry
        tracker._events["lgbm_power"].timestamp = datetime.now() - timedelta(days=8)
        assert not tracker.is_fallen_back("lgbm_power")

    def test_get_effective_tier(self):
        tracker = FallbackTracker(ttl_days=7)
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 3
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        assert tracker.get_effective_tier("lgbm_power", original_tier=3) == 2

    def test_active_fallbacks_list(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        tracker.record("transformer_power", from_tier=4, to_tier=3, error="CUDA OOM")
        active = tracker.active_fallbacks()
        assert len(active) == 2

    def test_clear_fallback(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM")
        tracker.clear("lgbm_power")
        assert not tracker.is_fallen_back("lgbm_power")

    def test_to_dict_serialization(self):
        tracker = FallbackTracker(ttl_days=7)
        tracker.record("lgbm_power", from_tier=3, to_tier=2, error="OOM", memory_mb=25600)
        result = tracker.to_dict()
        assert len(result) == 1
        assert result[0]["model"] == "lgbm_power"
        assert result[0]["from_tier"] == 3
        assert result[0]["to_tier"] == 2
        assert "timestamp" in result[0]
