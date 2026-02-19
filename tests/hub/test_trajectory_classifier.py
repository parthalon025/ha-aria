"""Tests for pattern recognition hub module."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aria.modules.trajectory_classifier import TrajectoryClassifier


def _make_mock_hub(config_overrides: dict | None = None):
    """Build a hub mock with a cache stub that serves config values."""
    hub = MagicMock()
    hub.subscribe = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.publish = AsyncMock()
    hub.modules = {}

    # Config values served from the cache layer (pattern used by all modules)
    _config = {
        "pattern.min_tier": 3,
        "pattern.sequence_window_size": 6,
    }
    if config_overrides:
        _config.update(config_overrides)

    async def _get_config_value(key, fallback=None):
        return _config.get(key, fallback)

    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(side_effect=_get_config_value)
    return hub


@pytest.fixture
def mock_hub():
    return _make_mock_hub()


class TestPatternRecognitionInit:
    """Test module initialization and tier gating."""

    def test_module_id(self, mock_hub):
        module = TrajectoryClassifier(mock_hub)
        assert module.module_id == "trajectory_classifier"

    def test_no_subscribe_in_constructor(self, mock_hub):
        TrajectoryClassifier(mock_hub)
        # Subscribe should NOT happen in __init__ — moved to initialize()
        mock_hub.subscribe.assert_not_called()

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=2)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_tier_gate_blocks_below_tier_3(self, mock_scan, mock_tier, mock_hub):
        """Module disables itself at Tier 2."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2)
        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is False

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_tier_gate_allows_tier_3(self, mock_scan, mock_tier, mock_hub):
        """Module activates at Tier 3."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is True


class TestTrajectoryClassification:
    """Test trajectory window management and classification."""

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_on_shadow_resolved_updates_cache(self, mock_scan, mock_tier, mock_hub):
        """Shadow resolved events feed the trajectory window."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4)
        module = TrajectoryClassifier(mock_hub)
        await module.initialize()

        # Feed enough events to build a window
        for i in range(6):
            await module._on_shadow_resolved(
                {
                    "target": "power_watts",
                    "features": {"activity": float(i * 10), "lights": 1.0},
                    "actual_value": float(i * 10),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Should have trajectory classification result
        assert module.current_trajectory == "ramping_up"

        # Verify cache payload structure
        cache_args = mock_hub.set_cache.call_args[0]
        assert cache_args[0] == "pattern_trajectory"
        assert cache_args[1]["trajectory"] == "ramping_up"
        assert cache_args[1]["method"] == "heuristic"
        assert "timestamp" in cache_args[1]

    async def test_store_and_retrieve_anomaly_explanations(self, mock_hub):
        """store_anomaly_explanations -> get_current_state round-trip."""
        with (
            patch("aria.modules.trajectory_classifier.scan_hardware") as mock_hw,
            patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3),
        ):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = TrajectoryClassifier(mock_hub)

        explanations = [{"feature": "power_watts", "contribution": 0.45}]
        module.store_anomaly_explanations(explanations)
        state = module.get_current_state()
        assert state["anomaly_explanations"] == explanations

    async def test_get_current_state(self, mock_hub):
        """get_current_state returns trajectory and scale info."""
        module = TrajectoryClassifier(mock_hub)
        state = module.get_current_state()
        assert "trajectory" in state
        assert "pattern_scales" in state
        assert "anomaly_explanations" in state

    async def test_get_stats(self, mock_hub):
        """get_stats includes sequence classifier info."""
        module = TrajectoryClassifier(mock_hub)
        stats = module.get_stats()
        assert "active" in stats
        assert "sequence_classifier" in stats
        assert "window_count" in stats


class TestPatternRecognitionLifecycle:
    """Test subscribe/unsubscribe lifecycle — subscribe in initialize(), not __init__."""

    @pytest.fixture
    def mock_hub(self):
        hub = _make_mock_hub()
        hub.unsubscribe = MagicMock()
        return hub

    def test_no_subscribe_in_init(self, mock_hub):
        """Subscribe should NOT happen in __init__ — only after initialize()."""
        with (
            patch("aria.modules.trajectory_classifier.scan_hardware") as mock_hw,
            patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3),
        ):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            TrajectoryClassifier(mock_hub)

        mock_hub.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_after_initialize_when_active(self, mock_hub):
        """Subscribe should happen in initialize() when tier >= 3."""
        with (
            patch("aria.modules.trajectory_classifier.scan_hardware") as mock_hw,
            patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3),
        ):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = TrajectoryClassifier(mock_hub)
            await module.initialize()

        mock_hub.subscribe.assert_called_once_with("shadow_resolved", module._on_shadow_resolved)

    @pytest.mark.asyncio
    async def test_no_subscribe_when_tier_too_low(self, mock_hub):
        """Should NOT subscribe when tier < MIN_TIER."""
        with (
            patch("aria.modules.trajectory_classifier.scan_hardware") as mock_hw,
            patch("aria.modules.trajectory_classifier.recommend_tier", return_value=2),
        ):
            mock_hw.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
            module = TrajectoryClassifier(mock_hub)
            await module.initialize()

        mock_hub.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_unsubscribes(self, mock_hub):
        """shutdown() must unsubscribe from shadow_resolved."""
        with (
            patch("aria.modules.trajectory_classifier.scan_hardware") as mock_hw,
            patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3),
        ):
            mock_hw.return_value = MagicMock(ram_gb=32, cpu_cores=8, gpu_available=False)
            module = TrajectoryClassifier(mock_hub)
            await module.initialize()
            await module.shutdown()

        mock_hub.unsubscribe.assert_called_once_with("shadow_resolved", module._on_shadow_resolved)


class TestAttentionExplainerIntegration:
    """Test attention explainer wiring in pattern recognition."""

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=4)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_tier_4_initializes_attention_explainer(self, mock_scan, mock_tier, mock_hub):
        """At Tier 4, attention explainer should be created."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=8, gpu_available=True, gpu_name="Test GPU")
        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is True
        assert module.attention_explainer is not None

    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_tier_3_no_attention_explainer(self, mock_scan, mock_tier, mock_hub):
        """At Tier 3, attention explainer should be None."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4, gpu_available=False)
        module = TrajectoryClassifier(mock_hub)
        await module.initialize()
        assert module.active is True
        assert module.attention_explainer is None

    def test_get_stats_includes_attention(self, mock_hub):
        module = TrajectoryClassifier(mock_hub)
        stats = module.get_stats()
        assert "attention_explainer" in stats


class TestModuleRegistration:
    """Test that pattern_recognition registers correctly in hub."""

    async def test_module_registers_without_error(self, mock_hub):
        """Module can be instantiated and registered."""
        module = TrajectoryClassifier(mock_hub)
        mock_hub.register_module = MagicMock()
        mock_hub.register_module(module)
        mock_hub.register_module.assert_called_once_with(module)


class TestConfigRead:
    """Verify initialize() reads pattern.min_tier and pattern.sequence_window_size from config."""

    @pytest.mark.asyncio
    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_reads_window_size_from_config(self, mock_scan, mock_tier):
        """sequence_classifier.window_size must reflect pattern.sequence_window_size config."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4, gpu_available=False)
        hub = _make_mock_hub({"pattern.sequence_window_size": 10})
        module = TrajectoryClassifier(hub)
        await module.initialize()
        assert module.sequence_classifier.window_size == 10
        assert module._max_window == 20

    @pytest.mark.asyncio
    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=2)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_reads_min_tier_from_config_blocks_low_tier(self, mock_scan, mock_tier):
        """Module respects pattern.min_tier from config — tier 2 blocked when min_tier=3."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
        hub = _make_mock_hub({"pattern.min_tier": 3})
        module = TrajectoryClassifier(hub)
        await module.initialize()
        assert module.active is False

    @pytest.mark.asyncio
    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=2)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_lower_min_tier_config_allows_activation(self, mock_scan, mock_tier):
        """Setting pattern.min_tier=2 in config allows tier 2 hardware to activate."""
        mock_scan.return_value = MagicMock(ram_gb=4, cpu_cores=2, gpu_available=False)
        hub = _make_mock_hub({"pattern.min_tier": 2})
        module = TrajectoryClassifier(hub)
        await module.initialize()
        assert module.active is True

    @pytest.mark.asyncio
    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_config_values_queried_on_initialize(self, mock_scan, mock_tier):
        """initialize() must call get_config_value for both config keys."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4, gpu_available=False)
        hub = _make_mock_hub()
        module = TrajectoryClassifier(hub)
        await module.initialize()
        queried_keys = [call.args[0] for call in hub.cache.get_config_value.call_args_list]
        assert "pattern.min_tier" in queried_keys
        assert "pattern.sequence_window_size" in queried_keys

    @pytest.mark.asyncio
    @patch("aria.modules.trajectory_classifier.recommend_tier", return_value=3)
    @patch("aria.modules.trajectory_classifier.scan_hardware")
    async def test_fallback_defaults_when_config_unavailable(self, mock_scan, mock_tier):
        """Module falls back to _DEFAULT_WINDOW_SIZE=6 when config returns None."""
        mock_scan.return_value = MagicMock(ram_gb=16, cpu_cores=4, gpu_available=False)
        hub = _make_mock_hub()
        # Override to always return None (simulates cold DB with no defaults seeded)
        hub.cache.get_config_value = AsyncMock(return_value=None)
        module = TrajectoryClassifier(hub)
        await module.initialize()
        assert module.sequence_classifier.window_size == 6  # _DEFAULT_WINDOW_SIZE
        assert module._max_window == 12
