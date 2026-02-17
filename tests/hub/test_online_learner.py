"""Tests for online learner hub module."""

from unittest.mock import AsyncMock, Mock

import pytest

from aria.modules.online_learner import OnlineLearnerModule


@pytest.fixture
def mock_hub():
    hub = Mock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.logger = Mock()
    hub.subscribe = Mock()
    hub.unsubscribe = Mock()
    hub.get_module = Mock(return_value=None)
    return hub


@pytest.fixture
def online_learner(mock_hub):
    return OnlineLearnerModule(mock_hub)


class TestOnlineLearnerModule:
    @pytest.mark.asyncio
    async def test_initialize_creates_models_per_target(self, online_learner):
        await online_learner.initialize()
        targets = ["power_watts", "lights_on", "devices_home", "unavailable", "useful_events"]
        for target in targets:
            assert target in online_learner.models

    @pytest.mark.asyncio
    async def test_initialize_subscribes_to_events(self, online_learner, mock_hub):
        await online_learner.initialize()
        mock_hub.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_on_shadow_resolved_feeds_model(self, online_learner):
        await online_learner.initialize()
        event_data = {
            "target": "power_watts",
            "features": {"hour_sin": 0.5, "hour_cos": 0.87, "temp_f": 65.0},
            "actual_value": 520.0,
            "outcome": "correct",
        }
        await online_learner._on_shadow_resolved(event_data)
        assert online_learner.models["power_watts"].samples_seen == 1

    @pytest.mark.asyncio
    async def test_get_prediction_returns_none_when_cold(self, online_learner):
        await online_learner.initialize()
        features = {"hour_sin": 0.5}
        pred = online_learner.get_prediction("power_watts", features)
        assert pred is None

    @pytest.mark.asyncio
    async def test_get_prediction_returns_value_after_learning(self, online_learner):
        await online_learner.initialize()
        features = {"hour_sin": 0.5, "temp_f": 65.0}
        for i in range(6):
            await online_learner._on_shadow_resolved(
                {
                    "target": "power_watts",
                    "features": features,
                    "actual_value": 500.0 + i * 10,
                    "outcome": "correct",
                }
            )
        pred = online_learner.get_prediction("power_watts", features)
        assert pred is not None
        assert isinstance(pred, float)

    @pytest.mark.asyncio
    async def test_get_all_stats(self, online_learner):
        await online_learner.initialize()
        stats = online_learner.get_all_stats()
        assert "power_watts" in stats
        assert stats["power_watts"]["samples_seen"] == 0

    @pytest.mark.asyncio
    async def test_on_drift_resets_affected_model(self, online_learner):
        await online_learner.initialize()
        for _i in range(3):
            await online_learner._on_shadow_resolved(
                {
                    "target": "power_watts",
                    "features": {"hour_sin": 0.5},
                    "actual_value": 500.0,
                    "outcome": "correct",
                }
            )
        assert online_learner.models["power_watts"].samples_seen == 3
        await online_learner.on_event("drift_detected", {"target": "power_watts"})
        assert online_learner.models["power_watts"].samples_seen == 0
