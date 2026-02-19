"""Tests for config_updated event propagation to modules (Issue #24).

Verifies that publishing a ``config_updated`` event via the hub reaches all
registered modules through their ``on_config_updated()`` method, and that
modules without an override do not crash.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from aria.hub.core import IntelligenceHub, Module

# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def hub():
    """Minimal initialized hub backed by a temporary SQLite file."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "test.db"
        h = IntelligenceHub(str(cache_path))
        await h.initialize()
        yield h
        await h.shutdown()


# ============================================================================
# Module.on_config_updated — base class contract
# ============================================================================


class TestModuleBaseClass:
    """Module.on_config_updated must exist and be a no-op by default."""

    @pytest.mark.asyncio
    async def test_default_on_config_updated_is_noop(self):
        """Calling on_config_updated on a plain Module does not raise."""
        mock_hub = MagicMock()
        module = Module(module_id="test_module", hub=mock_hub)

        # Should complete without raising — base implementation is a no-op
        await module.on_config_updated({"key": "shadow.min_confidence", "value": "0.5"})

    @pytest.mark.asyncio
    async def test_default_on_config_updated_returns_none(self):
        """Base on_config_updated returns None (pure side-effect interface)."""
        mock_hub = MagicMock()
        module = Module(module_id="test_module", hub=mock_hub)

        result = await module.on_config_updated({"key": "shadow.min_confidence", "value": "0.5"})
        assert result is None

    @pytest.mark.asyncio
    async def test_on_config_updated_accepts_empty_dict(self):
        """on_config_updated tolerates an empty payload without raising."""
        mock_hub = MagicMock()
        module = Module(module_id="test_module", hub=mock_hub)

        await module.on_config_updated({})


# ============================================================================
# IntelligenceHub.on_config_updated — dispatch to modules
# ============================================================================


class TestHubOnConfigUpdated:
    """Hub.on_config_updated propagates to every registered module."""

    @pytest.mark.asyncio
    async def test_hub_on_config_updated_calls_all_modules(self, hub):
        """All registered modules receive on_config_updated with the config dict."""

        # Create two modules with tracked on_config_updated methods
        class TrackingModule(Module):
            def __init__(self, module_id, hub_ref):
                super().__init__(module_id, hub_ref)
                self.received: list[dict] = []

            async def on_config_updated(self, config):
                self.received.append(config)

        mod_a = TrackingModule("mod_a", hub)
        mod_b = TrackingModule("mod_b", hub)
        hub.register_module(mod_a)
        hub.register_module(mod_b)

        payload = {"key": "shadow.min_confidence", "value": "0.7"}
        await hub.on_config_updated(payload)

        assert mod_a.received == [payload]
        assert mod_b.received == [payload]

    @pytest.mark.asyncio
    async def test_hub_on_config_updated_no_modules_does_not_raise(self, hub):
        """Hub with no registered modules handles config update gracefully."""
        await hub.on_config_updated({"key": "shadow.min_confidence", "value": "0.5"})

    @pytest.mark.asyncio
    async def test_hub_on_config_updated_module_exception_does_not_stop_others(self, hub):
        """If one module's on_config_updated raises, other modules still receive the update."""

        class FailingModule(Module):
            async def on_config_updated(self, config):
                raise RuntimeError("boom")

        class GoodModule(Module):
            def __init__(self, module_id, hub_ref):
                super().__init__(module_id, hub_ref)
                self.received: list[dict] = []

            async def on_config_updated(self, config):
                self.received.append(config)

        failing = FailingModule("failing", hub)
        good = GoodModule("good", hub)
        hub.register_module(failing)
        hub.register_module(good)

        payload = {"key": "shadow.min_confidence", "value": "0.5"}
        # Must not raise despite FailingModule error
        await hub.on_config_updated(payload)

        # GoodModule must still receive the update
        assert good.received == [payload]

    @pytest.mark.asyncio
    async def test_hub_on_config_updated_base_module_does_not_raise(self, hub):
        """Modules using the base no-op default do not crash the hub."""
        base_module = Module("plain", hub)
        hub.register_module(base_module)

        await hub.on_config_updated({"key": "activity.daily_snapshot_cap", "value": "15"})


# ============================================================================
# Event bus integration — config_updated event → modules
# ============================================================================


class TestConfigUpdatedEventBus:
    """Publishing config_updated on the hub event bus reaches modules."""

    @pytest.mark.asyncio
    async def test_publish_config_updated_reaches_module(self, hub):
        """hub.publish('config_updated', ...) triggers on_config_updated on modules."""

        class ListeningModule(Module):
            def __init__(self, module_id, hub_ref):
                super().__init__(module_id, hub_ref)
                self.received: list[dict] = []

            async def on_config_updated(self, config):
                self.received.append(config)

        listener = ListeningModule("listener", hub)
        hub.register_module(listener)

        payload = {"key": "shadow.min_confidence", "value": "0.4"}
        await hub.publish("config_updated", payload)

        assert listener.received == [payload]

    @pytest.mark.asyncio
    async def test_publish_other_event_does_not_call_on_config_updated(self, hub):
        """Publishing a different event type does not invoke on_config_updated."""

        class GuardModule(Module):
            def __init__(self, module_id, hub_ref):
                super().__init__(module_id, hub_ref)
                self.config_called = False

            async def on_config_updated(self, config):
                self.config_called = True

        guard = GuardModule("guard", hub)
        hub.register_module(guard)

        await hub.publish("cache_updated", {"category": "intelligence", "version": 1})

        assert guard.config_called is False

    @pytest.mark.asyncio
    async def test_config_updated_subscription_registered_during_initialize(self, hub):
        """After initialize(), 'config_updated' must have at least one subscriber."""
        assert "config_updated" in hub.subscribers
        assert len(hub.subscribers["config_updated"]) >= 1

    @pytest.mark.asyncio
    async def test_multiple_modules_all_receive_config_updated_via_publish(self, hub):
        """All modules receive on_config_updated when config_updated is published."""

        class CountingModule(Module):
            def __init__(self, module_id, hub_ref):
                super().__init__(module_id, hub_ref)
                self.count = 0

            async def on_config_updated(self, config):
                self.count += 1

        mods = [CountingModule(f"mod_{i}", hub) for i in range(3)]
        for mod in mods:
            hub.register_module(mod)

        await hub.publish("config_updated", {"key": "activity.flush_interval_s", "value": "600"})

        for mod in mods:
            assert mod.count == 1, f"{mod.module_id} did not receive config_updated"


# ============================================================================
# API integration — PUT /api/config publishes event
# ============================================================================


class TestApiConfigPropagation:
    """PUT /api/config/{key} publishes config_updated to modules via hub.publish."""

    def test_put_config_calls_hub_publish(self, api_hub, api_client):
        """PUT /api/config invokes hub.publish('config_updated', ...) after saving."""
        result = {"key": "shadow.min_confidence", "value": "0.5", "source": "user"}
        api_hub.cache.set_config = AsyncMock(return_value=result)
        api_hub.publish = AsyncMock()

        response = api_client.put(
            "/api/config/shadow.min_confidence",
            json={"value": "0.5", "changed_by": "user"},
        )
        assert response.status_code == 200

        api_hub.publish.assert_awaited_once_with(
            "config_updated",
            {"key": "shadow.min_confidence", "value": "0.5"},
        )

    def test_put_config_does_not_publish_on_error(self, api_hub, api_client):
        """hub.publish is NOT called when set_config raises ValueError."""
        api_hub.cache.set_config = AsyncMock(side_effect=ValueError("bad value"))
        api_hub.publish = AsyncMock()

        response = api_client.put(
            "/api/config/shadow.min_confidence",
            json={"value": "999"},
        )
        assert response.status_code == 400
        api_hub.publish.assert_not_awaited()
