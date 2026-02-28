"""Tests that registered config keys in activity_monitor and ml_engine
actually affect module behavior when changed via hub config.

Closes #319.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from aria.hub.core import IntelligenceHub
from aria.modules.activity_monitor import ActivityMonitor
from aria.modules.ml_engine import MLEngine


@pytest.fixture
def mock_hub():
    hub = Mock(spec=IntelligenceHub)
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.logger = Mock()
    hub.hardware_profile = None
    hub.cache = Mock()
    hub.cache.get_config_value = AsyncMock(return_value=None)
    hub.cache.get_included_entity_ids = AsyncMock(return_value=set())
    hub.cache.get_all_curation = AsyncMock(return_value=[])
    hub.schedule_task = AsyncMock()
    return hub


@pytest.mark.asyncio
async def test_activity_monitor_reads_daily_cap_from_config_closes_319(mock_hub):
    """ActivityMonitor.initialize() must read activity.daily_snapshot_cap from config."""
    custom_cap = 42
    mock_hub.cache.get_config_value = AsyncMock(
        side_effect=lambda key, fallback=None: custom_cap if key == "activity.daily_snapshot_cap" else fallback
    )

    monitor = ActivityMonitor(mock_hub, "http://test-host:8123", "test-token")
    await monitor.initialize()

    # The module must have applied the config value to control snapshot behavior
    assert monitor._daily_snapshot_cap == custom_cap, (
        f"ActivityMonitor.initialize() did not read activity.daily_snapshot_cap from config — "
        f"got {monitor._daily_snapshot_cap!r}, expected {custom_cap} (#319)"
    )


@pytest.mark.asyncio
async def test_activity_monitor_reads_cooldown_from_config_closes_319(mock_hub):
    """ActivityMonitor.initialize() must read activity.snapshot_cooldown_s from config."""
    custom_cooldown = 900
    mock_hub.cache.get_config_value = AsyncMock(
        side_effect=lambda key, fallback=None: custom_cooldown if key == "activity.snapshot_cooldown_s" else fallback
    )

    monitor = ActivityMonitor(mock_hub, "http://test-host:8123", "test-token")
    await monitor.initialize()

    assert monitor._snapshot_cooldown_s == custom_cooldown, (
        f"ActivityMonitor.initialize() did not read activity.snapshot_cooldown_s from config — "
        f"got {monitor._snapshot_cooldown_s!r}, expected {custom_cooldown} (#319)"
    )


@pytest.mark.asyncio
async def test_ml_engine_reads_decay_from_config_closes_319(mock_hub, tmp_path):
    """MLEngine.initialize() must read features.decay_half_life_days from config."""
    custom_decay = 14.0
    mock_hub.cache.get_config_value = AsyncMock(
        side_effect=lambda key, fallback=None: custom_decay if key == "features.decay_half_life_days" else fallback
    )
    mock_hub.get_cache_fresh = AsyncMock(return_value={"data": {"power_monitoring": {"available": True}}})

    engine = MLEngine(mock_hub, str(tmp_path / "models"), str(tmp_path / "training"))
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "training").mkdir(exist_ok=True)
    await engine.initialize()

    assert engine._decay_half_life_days == custom_decay, (
        f"MLEngine.initialize() did not read features.decay_half_life_days from config — "
        f"got {engine._decay_half_life_days!r}, expected {custom_decay} (#319)"
    )
