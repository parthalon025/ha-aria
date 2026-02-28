"""Tests for Batch 2c-ii fixes: #244, #246, #249, #250, #252, #255, #258, #260, #261, #262."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aria.hub.core import IntelligenceHub

# ---------------------------------------------------------------------------
# Shared hub fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def hub(tmp_path):
    """Minimal initialized hub backed by a temp SQLite file."""
    h = IntelligenceHub(str(tmp_path / "hub.db"))
    await h.initialize()
    yield h
    await h.shutdown()


# ---------------------------------------------------------------------------
# #244 — MLEngine.shutdown() cancels tasks and closes sessions
# ---------------------------------------------------------------------------


class TestMLEngineShutdown:
    """Verify MLEngine.shutdown() follows Convention D template."""

    @pytest.mark.asyncio
    async def test_shutdown_method_exists(self, tmp_path):
        from aria.modules.ml_engine import MLEngine

        hub_mock = MagicMock()
        hub_mock.hardware_profile = None
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.get_cache_fresh = AsyncMock(return_value=None)
        mod = MLEngine(hub_mock, models_dir=str(tmp_path / "models"), training_data_dir=str(tmp_path / "data"))
        assert hasattr(mod, "shutdown")
        assert asyncio.iscoroutinefunction(mod.shutdown)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_in_flight_task(self, tmp_path):
        from aria.modules.ml_engine import MLEngine

        hub_mock = MagicMock()
        hub_mock.hardware_profile = None
        mod = MLEngine(hub_mock, models_dir=str(tmp_path / "models"), training_data_dir=str(tmp_path / "data"))

        # Simulate an in-flight task
        async def slow():
            await asyncio.sleep(60)

        mod._task = asyncio.create_task(slow())
        await mod.shutdown()
        assert mod._task is None

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self, tmp_path):
        from aria.modules.ml_engine import MLEngine

        hub_mock = MagicMock()
        hub_mock.hardware_profile = None
        mod = MLEngine(hub_mock, models_dir=str(tmp_path / "models"), training_data_dir=str(tmp_path / "data"))

        mock_session = AsyncMock()
        mod._session = mock_session
        await mod.shutdown()
        mock_session.close.assert_called_once()
        assert mod._session is None

    @pytest.mark.asyncio
    async def test_shutdown_no_task_no_error(self, tmp_path):
        """Shutdown with no task/session should complete without error."""
        from aria.modules.ml_engine import MLEngine

        hub_mock = MagicMock()
        hub_mock.hardware_profile = None
        mod = MLEngine(hub_mock, models_dir=str(tmp_path / "models"), training_data_dir=str(tmp_path / "data"))
        await mod.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# #246 — OrchestratorModule: guard None session in _create_automation
# ---------------------------------------------------------------------------


class TestOrchestratorSessionGuard:
    """Verify OrchestratorModule guards None session before HTTP calls."""

    @pytest.mark.asyncio
    async def test_create_automation_returns_error_when_session_none(self):
        from aria.modules.orchestrator import OrchestratorModule

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.set_cache = AsyncMock()
        hub_mock.publish = AsyncMock()
        hub_mock.schedule_task = AsyncMock()
        mod = OrchestratorModule(hub_mock, "http://ha-host", "token")
        mod._session = None

        result = await mod._create_automation("test-id", {})
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_update_pattern_sensor_returns_early_when_session_none(self):
        from aria.modules.orchestrator import OrchestratorModule

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.schedule_task = AsyncMock()
        mod = OrchestratorModule(hub_mock, "http://ha-host", "token")
        mod._session = None

        # Should return early without raising AttributeError
        await mod.update_pattern_detection_sensor("test", "pat-1", 0.9)


# ---------------------------------------------------------------------------
# #249 — Presence: home=False gate logs at INFO with signal count
# ---------------------------------------------------------------------------


class TestPresenceHomeFalseLogging:
    """Verify UniFi home=False gate logs at INFO level with signal count."""

    @pytest.mark.asyncio
    async def test_home_false_gate_logs_info(self, caplog):
        import logging

        from aria.modules.presence import PresenceModule

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value={"home": False})
        hub_mock.get_module = MagicMock(return_value=None)
        mod = PresenceModule(hub_mock, ha_url="http://ha-host", ha_token="token")

        # Seed some signals
        mod._room_signals["living_room"] = [("motion", 0.8, "test", None)]
        mod._room_signals["bedroom"] = [("light", 0.6, "test", None), ("motion", 0.7, "test", None)]

        # Debounce (#249): call twice to confirm consecutive home=False before clearing
        with caplog.at_level(logging.INFO, logger="aria.modules.presence"):
            await mod._apply_unifi_cross_validation()
            await mod._apply_unifi_cross_validation()

        # All signals cleared
        assert all(len(sigs) == 0 for sigs in mod._room_signals.values())
        # Logged at INFO (not DEBUG)
        info_msgs = [r for r in caplog.records if r.levelno >= logging.INFO]
        assert any("home=False" in r.message or "clearing" in r.message.lower() for r in info_msgs), (
            f"Expected INFO log about home=False clearing, got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_home_false_gate_not_debug_only(self, caplog):
        """home=False log must not be at DEBUG level — operator must see it."""
        import logging

        from aria.modules.presence import PresenceModule

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value={"home": False})
        hub_mock.get_module = MagicMock(return_value=None)
        mod = PresenceModule(hub_mock, ha_url="http://ha-host", ha_token="token")
        mod._room_signals["office"] = [("network_client_present", 0.75, "test", None)]

        with caplog.at_level(logging.DEBUG, logger="aria.modules.presence"):
            await mod._apply_unifi_cross_validation()

        # Find any clearing-related log record
        clearing_records = [r for r in caplog.records if "clearing" in r.message.lower() or "home=False" in r.message]
        assert clearing_records, "No clearing log found at all"
        # Must be at INFO or above (not DEBUG)
        assert all(r.levelno >= logging.INFO for r in clearing_records), (
            f"Clearing log must be at INFO+, got: {[(r.levelname, r.message) for r in clearing_records]}"
        )


# ---------------------------------------------------------------------------
# #250 — ActivityMonitor.shutdown() flushes buffer
# ---------------------------------------------------------------------------


class TestActivityMonitorShutdown:
    """Verify ActivityMonitor.shutdown() flushes buffer before exit."""

    @pytest.mark.asyncio
    async def test_shutdown_method_exists(self):
        from aria.modules.activity_monitor import ActivityMonitor

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.set_cache = AsyncMock()
        mod = ActivityMonitor(hub_mock, ha_url="http://ha-host", ha_token="token")
        assert hasattr(mod, "shutdown")
        assert asyncio.iscoroutinefunction(mod.shutdown)

    @pytest.mark.asyncio
    async def test_shutdown_flushes_buffer(self):
        from aria.modules.activity_monitor import ActivityMonitor

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.set_cache = AsyncMock()
        mod = ActivityMonitor(hub_mock, ha_url="http://ha-host", ha_token="token")

        # Seed the buffer
        mod._activity_buffer.append({"entity_id": "light.test", "domain": "light", "timestamp": "2026-01-01T00:00:00"})

        flush_calls = []
        original_flush = mod._flush_activity_buffer

        async def tracked_flush():
            flush_calls.append(True)
            await original_flush()

        mod._flush_activity_buffer = tracked_flush
        await mod.shutdown()
        assert len(flush_calls) == 1, "shutdown() must call _flush_activity_buffer() when buffer is non-empty"

    @pytest.mark.asyncio
    async def test_shutdown_skips_flush_when_buffer_empty(self):
        """Empty buffer: no flush needed."""
        from aria.modules.activity_monitor import ActivityMonitor

        hub_mock = MagicMock()
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.set_cache = AsyncMock()
        mod = ActivityMonitor(hub_mock, ha_url="http://ha-host", ha_token="token")
        # Buffer is empty by default
        flush_mock = AsyncMock()
        mod._flush_activity_buffer = flush_mock
        await mod.shutdown()
        flush_mock.assert_not_called()


# ---------------------------------------------------------------------------
# #252 — Presence: _process_face_async reuses self._http_session
# ---------------------------------------------------------------------------


class TestPresenceSessionReuse:
    """Verify _process_face_async uses module-level session, not per-call."""

    @pytest.mark.asyncio
    async def test_process_face_async_uses_module_session(self):
        from aria.modules.presence import PresenceModule

        hub_mock = MagicMock()
        hub_mock.faces_store = MagicMock()
        mod = PresenceModule(hub_mock, ha_url="http://ha-host", ha_token="token")
        # No session set — should return early with warning, not crash
        mod._http_session = None
        # Should not raise AttributeError
        await mod._process_face_async("evt-1", "http://localhost/snap.jpg", "cam", "office")

    @pytest.mark.asyncio
    async def test_process_face_async_skips_when_session_closed(self):
        from aria.modules.presence import PresenceModule

        hub_mock = MagicMock()
        hub_mock.faces_store = MagicMock()
        mod = PresenceModule(hub_mock, ha_url="http://ha-host", ha_token="token")

        closed_session = MagicMock()
        closed_session.closed = True
        mod._http_session = closed_session
        # Should return early — no network calls on closed session
        await mod._process_face_async("evt-1", "http://localhost/snap.jpg", "cam", "office")
        # The closed session's .get() must NOT have been called
        closed_session.get.assert_not_called()


# ---------------------------------------------------------------------------
# #255 — UniFi: datetime timestamps are isoformat strings in signals
# ---------------------------------------------------------------------------


class TestUniFiIsoformatTimestamps:
    """Verify _process_clients emits ISO string timestamps, not datetime objects."""

    @pytest.mark.asyncio
    async def test_process_clients_ts_is_string(self):
        from aria.modules.unifi import UniFiModule

        hub_mock = MagicMock()
        cache_mock = MagicMock()
        cache_mock.get_config_value = AsyncMock(return_value=None)
        hub_mock.cache = cache_mock
        mod = UniFiModule(hub_mock, host="192.168.1.1", api_key="test-key")
        mod._ap_rooms = {"11:22:33:44:55:66": "office"}
        mod._device_people = {}
        mod._rssi_threshold = -75
        mod._active_kbps = 100

        clients = [
            {
                "mac": "aa:bb:cc:dd:ee:ff",
                "ap_mac": "11:22:33:44:55:66",
                "hostname": "iphone",
                "rssi": -55,
                "tx_bytes_r": 5000,
                "rx_bytes_r": 2000,
            }
        ]
        signals = mod._process_clients(clients)
        assert len(signals) > 0
        for sig in signals:
            assert isinstance(sig["ts"], str), f"Expected ISO string ts, got {type(sig['ts'])}"
            # Verify it's parseable ISO format
            from datetime import datetime

            datetime.fromisoformat(sig["ts"])  # raises ValueError if malformed

    @pytest.mark.asyncio
    async def test_process_clients_device_active_ts_is_string(self):
        """device_active signal also uses isoformat ts."""
        from aria.modules.unifi import UniFiModule

        hub_mock = MagicMock()
        cache_mock = MagicMock()
        cache_mock.get_config_value = AsyncMock(return_value=None)
        hub_mock.cache = cache_mock
        mod = UniFiModule(hub_mock, host="192.168.1.1", api_key="test-key")
        mod._ap_rooms = {"11:22:33:44:55:66": "office"}
        mod._device_people = {}
        mod._rssi_threshold = -75
        mod._active_kbps = 100  # bytes/s threshold for kbps check

        clients = [
            {
                "mac": "aa:bb:cc:dd:ee:ff",
                "ap_mac": "11:22:33:44:55:66",
                "hostname": "iphone",
                "rssi": -55,
                "tx_bytes_r": 20000,  # 20000+8000 bytes/s = 224 kbps > 100 threshold
                "rx_bytes_r": 8000,
            }
        ]
        signals = mod._process_clients(clients)
        device_active = [s for s in signals if s["signal_type"] == "device_active"]
        assert device_active, "Expected at least one device_active signal"
        assert isinstance(device_active[0]["ts"], str)


# ---------------------------------------------------------------------------
# #258 — UniFi: _load_config uses hub.cache.get_config_value
# ---------------------------------------------------------------------------


class TestUniFiCacheConfigAccess:
    """Verify _load_config uses hub.cache.get_config_value (not hub.get_config_value)."""

    @pytest.mark.asyncio
    async def test_load_config_calls_cache_get_config_value(self):
        from aria.modules.unifi import UniFiModule

        hub_mock = MagicMock()
        cache_mock = MagicMock()
        config_data = {
            "unifi.enabled": "false",
            "unifi.site": "default",
            "unifi.poll_interval_s": "30",
            "unifi.ap_rooms": "{}",
            "unifi.device_people": "{}",
            "unifi.rssi_room_threshold": "-75",
            "unifi.device_active_kbps": "100",
            "unifi.host": "",
        }
        cache_mock.get_config_value = AsyncMock(side_effect=lambda key, default=None: config_data.get(key, default))
        hub_mock.cache = cache_mock

        mod = UniFiModule(hub_mock, host="", api_key="test-key")
        await mod._load_config()

        # Must have called hub.cache.get_config_value (not hub.get_config_value)
        assert cache_mock.get_config_value.called

    @pytest.mark.asyncio
    async def test_load_config_graceful_when_no_cache(self):
        """When hub has no cache attribute, _load_config should warn and return."""
        from aria.modules.unifi import UniFiModule

        hub_mock = MagicMock(spec=[])  # no .cache attribute
        mod = UniFiModule(hub_mock, host="test-host", api_key="key")
        # Should not raise
        await mod._load_config()

    @pytest.mark.asyncio
    async def test_load_config_does_not_call_hub_get_config_value(self):
        """hub.get_config_value must NOT be called — only hub.cache.get_config_value."""
        from aria.modules.unifi import UniFiModule

        hub_mock = MagicMock()
        old_style_mock = MagicMock()
        hub_mock.get_config_value = old_style_mock  # legacy method that no longer exists

        # Provide typed values so int() conversions succeed
        config_data = {
            "unifi.enabled": "false",
            "unifi.host": "",
            "unifi.site": "default",
            "unifi.poll_interval_s": "30",
            "unifi.rssi_room_threshold": "-75",
            "unifi.device_active_kbps": "100",
            "unifi.ap_rooms": "{}",
            "unifi.device_people": "{}",
        }
        cache_mock = MagicMock()
        cache_mock.get_config_value = AsyncMock(side_effect=lambda key, default=None: config_data.get(key, default))
        hub_mock.cache = cache_mock

        mod = UniFiModule(hub_mock, host="host", api_key="key")
        await mod._load_config()

        # The old hub-level method must NOT be called
        old_style_mock.assert_not_called()


# ---------------------------------------------------------------------------
# #260 — MLEngine: ROLLING_FEATURE_NAMES shared constant
# ---------------------------------------------------------------------------


class TestRollingFeatureNamesConstant:
    """Verify ROLLING_FEATURE_NAMES is defined and used consistently."""

    def test_constant_exists(self):
        from aria.modules.ml_engine import ROLLING_FEATURE_NAMES

        assert isinstance(ROLLING_FEATURE_NAMES, list)
        assert len(ROLLING_FEATURE_NAMES) > 0

    def test_constant_covers_all_windows(self):
        from aria.modules.ml_engine import ROLLING_FEATURE_NAMES, ROLLING_WINDOWS_HOURS

        for h in ROLLING_WINDOWS_HOURS:
            assert f"rolling_{h}h_event_count" in ROLLING_FEATURE_NAMES
            assert f"rolling_{h}h_domain_entropy" in ROLLING_FEATURE_NAMES
            assert f"rolling_{h}h_dominant_domain_pct" in ROLLING_FEATURE_NAMES
            assert f"rolling_{h}h_trend" in ROLLING_FEATURE_NAMES

    def test_constant_length_matches_windows_times_metrics(self):
        from aria.modules.ml_engine import ROLLING_FEATURE_NAMES, ROLLING_WINDOWS_HOURS

        expected = len(ROLLING_WINDOWS_HOURS) * 4  # 4 metrics per window
        assert len(ROLLING_FEATURE_NAMES) == expected

    @pytest.mark.asyncio
    async def test_feature_names_include_rolling_constants(self, tmp_path):
        """_get_feature_names() must include all names in ROLLING_FEATURE_NAMES."""
        from aria.modules.ml_engine import ROLLING_FEATURE_NAMES, MLEngine

        hub_mock = MagicMock()
        hub_mock.hardware_profile = None
        hub_mock.get_cache = AsyncMock(return_value=None)
        hub_mock.get_cache_fresh = AsyncMock(return_value=None)
        mod = MLEngine(
            hub_mock,
            models_dir=str(tmp_path / "models"),
            training_data_dir=str(tmp_path / "data"),
        )

        names = await mod._get_feature_names()
        for name in ROLLING_FEATURE_NAMES:
            assert name in names, f"Expected {name} in feature names from _get_feature_names()"


# ---------------------------------------------------------------------------
# #261 — Hub: on_event dispatch wrapped in asyncio.wait_for timeout
# ---------------------------------------------------------------------------


class TestOnEventTimeout:
    """Verify on_event dispatch uses asyncio.wait_for with 5s timeout."""

    @pytest.mark.asyncio
    async def test_slow_module_logs_warning_not_blocked(self, hub, caplog):
        """A module that hangs >5s should be warned, not block the bus."""
        import logging

        class SlowModule:
            module_id = "slow_test"

            async def on_event(self, event_type, data):
                await asyncio.sleep(60)  # Simulate hung handler

        slow_mod = SlowModule()
        hub.modules["slow_test"] = slow_mod

        with caplog.at_level(logging.WARNING):
            # Should complete quickly (timeout fires at 5s), not hang
            await asyncio.wait_for(hub.publish("test_event", {}), timeout=10.0)

        # Warning logged for the slow handler
        timeout_warnings = [r for r in caplog.records if "timeout" in r.message.lower() and "slow_test" in r.message]
        assert len(timeout_warnings) > 0, (
            f"Expected timeout warning for slow_test, got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_fast_module_completes_normally(self, hub):
        """A fast module should complete without timeout warning."""
        events_received = []

        class FastModule:
            module_id = "fast_test"

            async def on_event(self, event_type, data):
                events_received.append(event_type)

        hub.modules["fast_test"] = FastModule()
        await hub.publish("hello_event", {"x": 1})
        assert "hello_event" in events_received

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise_to_bus(self, hub):
        """Timeout in one module must not propagate as an exception to the caller."""

        class HungModule:
            module_id = "hung_test"

            async def on_event(self, event_type, data):
                await asyncio.sleep(60)

        hub.modules["hung_test"] = HungModule()
        # publish should return normally, not raise TimeoutError
        await hub.publish("test_event", {})


# ---------------------------------------------------------------------------
# #262 — Snapshot: time_features written to intraday snapshot
# ---------------------------------------------------------------------------


class TestIntradayTimeFeatures:
    """Verify build_intraday_snapshot populates time_features."""

    def _make_store(self, tmp_path):
        """Build a DataStore backed by tmp_path."""
        from aria.engine.config import PathConfig
        from aria.engine.storage.data_store import DataStore

        paths = PathConfig(
            data_dir=tmp_path / "intelligence",
            logbook_path=tmp_path / "current.json",
        )
        paths.ensure_dirs()
        return DataStore(paths)

    def test_intraday_snapshot_has_time_features(self, tmp_path):
        from aria.engine.collectors.snapshot import build_intraday_snapshot
        from aria.engine.config import AppConfig

        store = self._make_store(tmp_path)
        config = AppConfig.from_env()

        presence_collector = MagicMock()
        presence_collector.inject_presence = MagicMock()

        with (
            patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[]),
            patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None),
            patch("aria.engine.collectors.snapshot.parse_weather", return_value={}),
            patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={}),
            patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None),
            patch("aria.engine.collectors.snapshot.CollectorRegistry.all", return_value={}),
            patch(
                "aria.engine.collectors.snapshot.CollectorRegistry.get",
                return_value=lambda: presence_collector,
            ),
        ):
            snapshot = build_intraday_snapshot(hour=9, date_str="2026-02-25", config=config, store=store)

        assert "time_features" in snapshot, "build_intraday_snapshot must set snapshot['time_features']"
        tf = snapshot["time_features"]
        assert isinstance(tf, dict)
        assert "hour" in tf
        assert "dow" in tf
        assert "is_weekend" in tf

    def test_time_features_values_are_correct_type(self, tmp_path):
        """time_features values should be numeric or bool — not raw datetime."""
        import json

        from aria.engine.collectors.snapshot import build_intraday_snapshot
        from aria.engine.config import AppConfig

        store = self._make_store(tmp_path)
        config = AppConfig.from_env()

        presence_collector = MagicMock()
        presence_collector.inject_presence = MagicMock()

        with (
            patch("aria.engine.collectors.snapshot.fetch_ha_states", return_value=[]),
            patch("aria.engine.collectors.snapshot.fetch_weather", return_value=None),
            patch("aria.engine.collectors.snapshot.parse_weather", return_value={}),
            patch("aria.engine.collectors.snapshot.summarize_logbook", return_value={}),
            patch("aria.engine.collectors.snapshot._fetch_presence_cache", return_value=None),
            patch("aria.engine.collectors.snapshot.CollectorRegistry.all", return_value={}),
            patch(
                "aria.engine.collectors.snapshot.CollectorRegistry.get",
                return_value=lambda: presence_collector,
            ),
        ):
            snapshot = build_intraday_snapshot(hour=14, date_str="2026-02-25", config=config, store=store)

        tf = snapshot.get("time_features", {})
        # All values must be JSON-serializable (no datetime objects)
        json.dumps(tf)  # raises TypeError if any value is a datetime
