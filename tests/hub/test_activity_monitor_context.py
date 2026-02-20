"""Tests for activity monitor context_parent_id extraction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.modules.activity_monitor import ActivityMonitor


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.event_store = MagicMock()
    hub.event_store.insert_event = AsyncMock()
    hub.entity_graph = MagicMock()
    hub.entity_graph.get_area.return_value = "bedroom"
    hub.entity_graph.get_device.return_value = {"device_id": "dev_123"}
    return hub


@pytest.fixture
def monitor(mock_hub):
    m = ActivityMonitor.__new__(ActivityMonitor)
    m.hub = mock_hub
    m.logger = MagicMock()
    return m


def _make_event():
    return {
        "entity_id": "light.bedroom",
        "domain": "light",
        "from": "off",
        "to": "on",
        "timestamp": "2026-02-20T07:00:00",
    }


class TestContextParentIdExtraction:
    @pytest.mark.asyncio
    async def test_manual_event_persists_none_context(self, monitor, mock_hub):
        """Manual actions should persist with context_parent_id=None."""
        ws_context = {"id": "abc123", "parent_id": None, "user_id": "user1"}
        monitor._persist_to_event_store(_make_event(), {}, ws_context=ws_context)
        # Allow the created task to run
        await asyncio.sleep(0.01)

        call_args = mock_hub.event_store.insert_event.call_args
        assert call_args is not None
        assert call_args.kwargs.get("context_parent_id") is None

    @pytest.mark.asyncio
    async def test_automated_event_persists_parent_id(self, monitor, mock_hub):
        """Automation-triggered events should persist context_parent_id."""
        ws_context = {"id": "abc123", "parent_id": "automation.morning_lights", "user_id": None}
        monitor._persist_to_event_store(_make_event(), {}, ws_context=ws_context)
        await asyncio.sleep(0.01)

        call_args = mock_hub.event_store.insert_event.call_args
        assert call_args is not None
        assert call_args.kwargs.get("context_parent_id") == "automation.morning_lights"

    @pytest.mark.asyncio
    async def test_no_context_persists_none(self, monitor, mock_hub):
        """When ws_context is None, context_parent_id defaults to None."""
        monitor._persist_to_event_store(_make_event(), {}, ws_context=None)
        await asyncio.sleep(0.01)

        call_args = mock_hub.event_store.insert_event.call_args
        assert call_args is not None
        assert call_args.kwargs.get("context_parent_id") is None
