"""Regression tests for #246 (orchestrator session guard) and #250 (activity_monitor shutdown).

#246: OrchestratorModule._create_automation and update_pattern_detection_sensor
      must not raise AttributeError when _session is None (race with shutdown).
#250: ActivityMonitor must have a shutdown() that flushes the event buffer.
"""

import asyncio

import pytest

from aria.modules.activity_monitor import ActivityMonitor
from aria.modules.orchestrator import OrchestratorModule

# ---------------------------------------------------------------------------
# Minimal mocks
# ---------------------------------------------------------------------------


class _MockHub:
    def __init__(self):
        self._cache: dict = {}
        self.modules: dict = {}

    async def get_cache(self, key):
        return self._cache.get(key)

    async def set_cache(self, key, data, metadata=None):
        self._cache[key] = {"data": data}

    async def publish(self, *a, **kw):
        pass

    async def schedule_task(self, *a, **kw):
        pass

    def is_running(self):
        return False

    @property
    def cache(self):
        class _C:
            async def get_config_value(self, k, d=None):
                return d

            async def get_included_entity_ids(self):
                return set()

            async def get_all_curation(self):
                return []

        return _C()


# ---------------------------------------------------------------------------
# #246 — orchestrator session guard
# ---------------------------------------------------------------------------


class TestOrchestratorSessionGuardCloses246:
    @pytest.mark.asyncio
    async def test_create_automation_returns_failure_when_session_none(self):
        """_create_automation must return failure dict, not raise, when _session is None."""
        hub = _MockHub()
        orch = OrchestratorModule(hub=hub, ha_url="http://ha", ha_token="tok")
        # _session is None (never initialized)
        assert orch._session is None

        result = await orch._create_automation("auto_1", {"alias": "test"})

        assert result["success"] is False
        assert "session" in result.get("error", "").lower() or "not initialized" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_update_sensor_returns_none_when_session_none(self):
        """update_pattern_detection_sensor must silently return when _session is None."""
        hub = _MockHub()
        orch = OrchestratorModule(hub=hub, ha_url="http://ha", ha_token="tok")
        assert orch._session is None

        # Must not raise
        result = await orch.update_pattern_detection_sensor("wake_up", "p1", 0.9)
        assert result is None


# ---------------------------------------------------------------------------
# #250 — activity_monitor shutdown flushes buffer
# ---------------------------------------------------------------------------


class TestActivityMonitorShutdownCloses250:
    @pytest.mark.asyncio
    async def test_shutdown_exists(self):
        """ActivityMonitor must define an async shutdown() method."""
        hub = _MockHub()
        am = ActivityMonitor(hub=hub, ha_url="http://ha", ha_token="tok")
        assert hasattr(am, "shutdown"), "ActivityMonitor must have shutdown()"
        assert asyncio.iscoroutinefunction(am.shutdown), "shutdown() must be async"

    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffer(self):
        """shutdown() must flush non-empty _activity_buffer before exiting."""
        hub = _MockHub()
        am = ActivityMonitor(hub=hub, ha_url="http://ha", ha_token="tok")

        # Seed a fake event in the buffer
        am._activity_buffer.append({"entity_id": "light.test", "to": "on"})

        # Replace _flush_activity_buffer with a spy
        flushed = []

        async def _spy_flush():
            flushed.append(True)

        am._flush_activity_buffer = _spy_flush

        await am.shutdown()

        assert flushed, "shutdown() must have called _flush_activity_buffer"

    @pytest.mark.asyncio
    async def test_shutdown_empty_buffer_does_not_flush(self):
        """shutdown() must skip flush when buffer is already empty."""
        hub = _MockHub()
        am = ActivityMonitor(hub=hub, ha_url="http://ha", ha_token="tok")
        assert am._activity_buffer == []

        flushed = []

        async def _spy_flush():
            flushed.append(True)

        am._flush_activity_buffer = _spy_flush
        await am.shutdown()

        assert not flushed, "shutdown() must not flush when buffer is empty"
