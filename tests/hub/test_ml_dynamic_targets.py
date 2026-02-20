"""Tests for per-area and per-domain dynamic prediction targets."""

from unittest.mock import AsyncMock, Mock

import pytest

from aria.hub.core import IntelligenceHub
from aria.modules.ml_engine import MLEngine


@pytest.fixture
def mock_hub():
    hub = Mock(spec=IntelligenceHub)
    hub.get_cache = AsyncMock(return_value=None)
    hub.get_cache_fresh = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.logger = Mock()
    hub.hardware_profile = None
    hub.event_store = Mock()
    hub.entity_graph = Mock()
    return hub


@pytest.fixture
def ml_engine(mock_hub, tmp_path):
    models_dir = tmp_path / "models"
    training_data_dir = tmp_path / "training_data"
    models_dir.mkdir()
    training_data_dir.mkdir()
    return MLEngine(mock_hub, str(models_dir), str(training_data_dir))


class TestGetDynamicTargets:
    @pytest.mark.asyncio
    async def test_returns_area_targets_for_active_areas(self, ml_engine):
        """Areas with >100 events in 7 days get prediction targets."""
        # Mock EventStore to return area activity counts
        ml_engine.hub.event_store.query_events = AsyncMock(
            return_value=[{"area_id": "kitchen", "domain": "light", "entity_id": f"light.k{i}"} for i in range(150)]
            + [{"area_id": "bedroom", "domain": "light", "entity_id": f"light.b{i}"} for i in range(50)]
        )

        targets = await ml_engine._get_dynamic_targets()
        # kitchen has 150 events (>100) — included
        assert "area_kitchen_activity" in targets
        # bedroom has 50 events (<100) — excluded
        assert "area_bedroom_activity" not in targets

    @pytest.mark.asyncio
    async def test_returns_domain_targets(self, ml_engine):
        """Key domains with sufficient events get prediction targets."""
        ml_engine.hub.event_store.query_events = AsyncMock(
            return_value=[{"area_id": None, "domain": "light", "entity_id": f"light.x{i}"} for i in range(200)]
            + [{"area_id": None, "domain": "binary_sensor", "entity_id": f"binary_sensor.x{i}"} for i in range(120)]
        )

        targets = await ml_engine._get_dynamic_targets()
        assert "domain_light_event_count" in targets
        assert "domain_binary_sensor_event_count" in targets

    @pytest.mark.asyncio
    async def test_no_targets_when_no_event_store(self, ml_engine):
        """Without EventStore, returns empty dict."""
        ml_engine.hub.event_store = None
        targets = await ml_engine._get_dynamic_targets()
        assert targets == {}

    @pytest.mark.asyncio
    async def test_no_targets_when_no_events(self, ml_engine):
        """With EventStore but no events, returns empty dict."""
        ml_engine.hub.event_store.query_events = AsyncMock(return_value=[])
        targets = await ml_engine._get_dynamic_targets()
        assert targets == {}

    @pytest.mark.asyncio
    async def test_target_values_from_segment(self, ml_engine):
        """Dynamic target extractors return correct values from segment data."""
        ml_engine.hub.event_store.query_events = AsyncMock(
            return_value=[{"area_id": "kitchen", "domain": "light", "entity_id": f"light.k{i}"} for i in range(150)]
        )

        targets = await ml_engine._get_dynamic_targets()
        # Each target should have an extractor function
        assert "area_kitchen_activity" in targets
        extractor = targets["area_kitchen_activity"]

        # Test the extractor with a segment
        segment = {
            "per_area_activity": {"kitchen": 10, "bedroom": 3},
            "per_domain_counts": {"light": 8, "binary_sensor": 5},
        }
        assert extractor(segment) == 10

    @pytest.mark.asyncio
    async def test_domain_target_extractor(self, ml_engine):
        """Domain target extractor pulls from per_domain_counts."""
        ml_engine.hub.event_store.query_events = AsyncMock(
            return_value=[{"area_id": None, "domain": "light", "entity_id": f"light.x{i}"} for i in range(200)]
        )

        targets = await ml_engine._get_dynamic_targets()
        extractor = targets["domain_light_event_count"]

        segment = {
            "per_area_activity": {},
            "per_domain_counts": {"light": 15, "switch": 3},
        }
        assert extractor(segment) == 15
