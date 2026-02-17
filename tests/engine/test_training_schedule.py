"""Verify first-boot immediate training — gap confirmed closed."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

try:
    from aria.modules.ml_engine import MLEngine

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

pytestmark = pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")


@pytest.fixture
def mock_hub():
    hub = Mock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.schedule_task = AsyncMock()
    hub.logger = Mock()
    return hub


@pytest.fixture
def ml_engine(mock_hub, tmp_path):
    models_dir = tmp_path / "models"
    training_data_dir = tmp_path / "training_data"
    models_dir.mkdir()
    training_data_dir.mkdir()
    return MLEngine(mock_hub, str(models_dir), str(training_data_dir))


class TestTrainingScheduleStartup:
    @pytest.mark.asyncio
    async def test_first_boot_trains_immediately(self, ml_engine, mock_hub):
        """When no training metadata exists, training should run immediately."""
        mock_hub.get_cache.return_value = None
        await ml_engine.schedule_periodic_training(interval_days=7)
        mock_hub.schedule_task.assert_called_once()
        call_kwargs = mock_hub.schedule_task.call_args[1]
        assert call_kwargs["run_immediately"] is True

    @pytest.mark.asyncio
    async def test_recent_training_does_not_run_immediately(self, ml_engine, mock_hub):
        """When training ran recently, do not run immediately."""
        recent = datetime.now() - timedelta(days=1)
        mock_hub.get_cache.return_value = {"last_trained": recent.isoformat()}
        await ml_engine.schedule_periodic_training(interval_days=7)
        mock_hub.schedule_task.assert_called_once()
        call_kwargs = mock_hub.schedule_task.call_args[1]
        assert call_kwargs["run_immediately"] is False

    @pytest.mark.asyncio
    async def test_stale_training_runs_immediately(self, ml_engine, mock_hub):
        """When training is older than interval, run immediately."""
        stale = datetime.now() - timedelta(days=10)
        mock_hub.get_cache.return_value = {"last_trained": stale.isoformat()}
        await ml_engine.schedule_periodic_training(interval_days=7)
        mock_hub.schedule_task.assert_called_once()
        call_kwargs = mock_hub.schedule_task.call_args[1]
        assert call_kwargs["run_immediately"] is True
