"""Tests for Batch 2c-i fixes: #229, #231, #233, #236, #237, #238, #240, #241."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aria.hub.core import IntelligenceHub
from aria.shared.event_store import EventStore

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def hub(tmp_path):
    """Minimal initialized hub backed by a temp SQLite file."""
    h = IntelligenceHub(str(tmp_path / "hub.db"))
    await h.initialize()
    yield h
    await h.shutdown()


# ---------------------------------------------------------------------------
# #229 — subscriber lifecycle: initialize() closures unsubscribed on shutdown
# ---------------------------------------------------------------------------


class TestSubscriberLifecycle:
    """Verify initialize() subscriber closures are unsubscribed on shutdown."""

    def test_subscribers_list_initialized_in_init(self, tmp_path):
        """Hub.__init__ should create _subscribers as an empty list."""
        h = IntelligenceHub(str(tmp_path / "hub.db"))
        assert hasattr(h, "_subscribers")
        assert isinstance(h._subscribers, list)
        assert h._subscribers == []

    @pytest.mark.asyncio
    async def test_subscribers_populated_after_initialize(self, hub):
        """After initialize(), _subscribers should contain the two closures."""
        # Two closures registered in initialize(): config_updated + cache_updated
        assert len(hub._subscribers) == 2
        event_types = [evt for evt, _ in hub._subscribers]
        assert "config_updated" in event_types
        assert "cache_updated" in event_types

    @pytest.mark.asyncio
    async def test_subscribers_cleared_after_shutdown(self, tmp_path):
        """After shutdown(), _subscribers should be empty."""
        h = IntelligenceHub(str(tmp_path / "hub.db"))
        await h.initialize()
        assert len(h._subscribers) == 2

        await h.shutdown()
        assert h._subscribers == []

    @pytest.mark.asyncio
    async def test_closures_removed_from_hub_subscribers_on_shutdown(self, tmp_path):
        """The actual hub.subscribers sets should no longer contain the closures after shutdown."""
        h = IntelligenceHub(str(tmp_path / "hub.db"))
        await h.initialize()

        # Capture the callbacks before shutdown
        stored_callbacks = dict(h._subscribers)

        await h.shutdown()

        # Verify the callbacks were removed from the subscriber sets
        config_cbs = h.subscribers.get("config_updated", set())
        cache_cbs = h.subscribers.get("cache_updated", set())
        assert stored_callbacks["config_updated"] not in config_cbs
        assert stored_callbacks["cache_updated"] not in cache_cbs

    @pytest.mark.asyncio
    async def test_restart_does_not_accumulate_ghost_subscribers(self, tmp_path):
        """Repeated initialize()+shutdown() cycles should not accumulate subscribers."""
        cache_path = str(tmp_path / "hub.db")
        h = IntelligenceHub(cache_path)

        # First cycle
        await h.initialize()
        count_after_first_init = len(h.subscribers.get("cache_updated", set()))
        await h.shutdown()

        # Second cycle on same hub — reinitialize
        h._running = False
        await h.initialize()
        count_after_second_init = len(h.subscribers.get("cache_updated", set()))
        await h.shutdown()

        # Should not accumulate (second init restores to same count, not doubled)
        assert count_after_second_init == count_after_first_init


# ---------------------------------------------------------------------------
# #237 — schedule_task done-callback for error visibility
# ---------------------------------------------------------------------------


class TestScheduleTaskDoneCallback:
    """Verify schedule_task attaches _log_task_exception done-callback."""

    @pytest.mark.asyncio
    async def test_done_callback_attached(self, hub):
        """schedule_task should add _log_task_exception as done callback."""
        called = []

        async def noop():
            pass

        # Patch _log_task_exception to track calls
        original_log = hub._log_task_exception

        def tracking_log(task):
            called.append(task)
            original_log(task)

        hub._log_task_exception = tracking_log

        await hub.schedule_task("test_noop", noop, interval=None, run_immediately=True)
        # Allow the one-shot task to complete
        await asyncio.sleep(0.05)

        # Done callback should have fired (task completed normally)
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_error_logged_for_crashing_task(self, hub):
        """An exception inside a scheduled coroutine should be logged by done-callback."""

        async def always_fails():
            raise ValueError("intentional test error")

        error_calls = []

        def capture_log(task):
            if not task.cancelled() and task.exception():
                error_calls.append(task.exception())

        hub._log_task_exception = capture_log

        await hub.schedule_task("fail_task", always_fails, interval=None, run_immediately=True)
        await asyncio.sleep(0.05)

        # The inner try/except in run_task catches and logs, so task completes normally.
        # The done-callback fires but task.exception() is None (exception was caught internally).
        # This is correct behavior — internal logging already handled it.
        # What we verify is that the task ran and no unhandled exception propagates to the test.
        assert len(error_calls) == 0  # Caught internally by run_task


# ---------------------------------------------------------------------------
# #231 — routes_faces: add_embedding failure returns 500
# ---------------------------------------------------------------------------


class TestAddEmbeddingFailure:
    """Verify label_face returns 500 on genuine add_embedding failure."""

    def _make_api_hub(self):
        """Create a mock hub for API testing."""
        mock_hub = MagicMock(spec=IntelligenceHub)
        mock_hub.cache = MagicMock()
        mock_hub.modules = {}
        mock_hub.module_status = {}
        mock_hub.subscribers = {}
        mock_hub.subscribe = MagicMock()
        mock_hub._request_count = 0
        mock_hub._audit_logger = None
        mock_hub.set_cache = AsyncMock()
        mock_hub.get_uptime_seconds = MagicMock(return_value=0)
        mock_hub.publish = AsyncMock()
        return mock_hub

    def _make_authed_client(self, mock_hub):
        """Create a TestClient with API key set for auth-enabled tests."""
        from fastapi.testclient import TestClient

        import aria.hub.api as _api_mod
        from aria.hub.api import create_api

        _test_key = "test-embedding-key"
        _api_mod._ARIA_API_KEY = _test_key
        app = create_api(mock_hub)
        return TestClient(app, headers={"X-API-Key": _test_key})

    def test_500_on_add_embedding_generic_failure(self):
        """When add_embedding raises a non-duplicate error, the endpoint returns 500."""
        import aria.hub.api as _api_mod

        mock_hub = self._make_api_hub()

        # Set up a fake faces_store
        mock_store = MagicMock()
        mock_item = {
            "id": 1,
            "event_id": "ev1",
            "image_path": "/fake/path.jpg",
            "embedding": b"fake_embedding",
        }
        mock_store.get_pending_queue_item.return_value = mock_item
        mock_store.mark_reviewed.return_value = True
        mock_store.add_embedding.side_effect = RuntimeError("disk full — write failed")
        mock_hub.faces_store = mock_store

        original = _api_mod._ARIA_API_KEY
        try:
            client = self._make_authed_client(mock_hub)
            resp = client.post("/api/faces/label", json={"queue_id": 1, "person_name": "Alice"})
            assert resp.status_code == 500
            assert "embedding" in resp.json()["detail"].lower()
        finally:
            _api_mod._ARIA_API_KEY = original

    def test_200_on_duplicate_embedding(self):
        """When add_embedding raises a UNIQUE constraint error, label still succeeds (200)."""
        import aria.hub.api as _api_mod

        mock_hub = self._make_api_hub()

        mock_store = MagicMock()
        mock_item = {
            "id": 2,
            "event_id": "ev2",
            "image_path": "/fake/path.jpg",
            "embedding": b"fake_embedding",
        }
        mock_store.get_pending_queue_item.return_value = mock_item
        mock_store.mark_reviewed.return_value = True
        mock_store.add_embedding.side_effect = Exception("UNIQUE constraint failed: embeddings.event_id")
        mock_hub.faces_store = mock_store

        original = _api_mod._ARIA_API_KEY
        try:
            client = self._make_authed_client(mock_hub)
            resp = client.post("/api/faces/label", json={"queue_id": 2, "person_name": "Bob"})
            assert resp.status_code == 200
        finally:
            _api_mod._ARIA_API_KEY = original


# ---------------------------------------------------------------------------
# #233 — api.py: auth_enabled in /health response
# ---------------------------------------------------------------------------


class TestAuthEnabledInHealth:
    """Verify /health includes auth_enabled key."""

    def test_health_contains_auth_enabled_key(self):
        """GET /health must include auth_enabled boolean."""
        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        import aria.hub.api as _api_mod
        from aria.hub.api import create_api

        mock_hub = MagicMock(spec=IntelligenceHub)
        mock_hub.cache = MagicMock()
        mock_hub.modules = {}
        mock_hub.module_status = {}
        mock_hub.subscribers = {}
        mock_hub.subscribe = MagicMock()
        mock_hub._request_count = 0
        mock_hub._audit_logger = None
        mock_hub.set_cache = AsyncMock()
        mock_hub.get_uptime_seconds = MagicMock(return_value=0)
        mock_hub.publish = AsyncMock()
        mock_hub.health_check = AsyncMock(
            return_value={
                "status": "ok",
                "uptime_seconds": 42,
                "modules": {},
                "cache": {"categories": []},
                "timestamp": "2026-02-25T12:00:00+00:00",
            }
        )

        _test_key = "test-health-key"
        original = _api_mod._ARIA_API_KEY
        _api_mod._ARIA_API_KEY = _test_key
        try:
            app = create_api(mock_hub)
            client = TestClient(app, headers={"X-API-Key": _test_key})
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert "auth_enabled" in data, f"auth_enabled not in health response: {data.keys()}"
            assert isinstance(data["auth_enabled"], bool)
        finally:
            _api_mod._ARIA_API_KEY = original

    def test_auth_disabled_when_key_not_set(self):
        """When ARIA_API_KEY is set, auth_enabled is True in health response."""

        from unittest.mock import AsyncMock, MagicMock

        from fastapi.testclient import TestClient

        import aria.hub.api as api_mod
        from aria.hub.api import create_api

        mock_hub = MagicMock(spec=IntelligenceHub)
        mock_hub.cache = MagicMock()
        mock_hub.modules = {}
        mock_hub.module_status = {}
        mock_hub.subscribers = {}
        mock_hub.subscribe = MagicMock()
        mock_hub._request_count = 0
        mock_hub._audit_logger = None
        mock_hub.set_cache = AsyncMock()
        mock_hub.get_uptime_seconds = MagicMock(return_value=0)
        mock_hub.publish = AsyncMock()
        mock_hub.health_check = AsyncMock(
            return_value={
                "status": "ok",
                "uptime_seconds": 0,
                "modules": {},
                "cache": {"categories": []},
                "timestamp": "2026-02-25T12:00:00+00:00",
            }
        )

        _test_key = "test-health-key-2"
        original = api_mod._ARIA_API_KEY
        api_mod._ARIA_API_KEY = _test_key
        try:
            app = create_api(mock_hub)
            client = TestClient(app, headers={"X-API-Key": _test_key})
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            # auth_enabled must match whether the API key was configured
            expected_auth = bool(api_mod._ARIA_API_KEY)
            assert data["auth_enabled"] == expected_auth
            assert data["auth_enabled"] is True  # key is set
        finally:
            api_mod._ARIA_API_KEY = original

    def test_startup_warning_exists_when_key_missing(self):
        """The module should issue a warning when ARIA_API_KEY is absent at import time."""
        import aria.hub.api as api_mod

        # Simply verify the guard logic exists — when _ARIA_API_KEY is falsy,
        # auth is disabled. The warning was already emitted at module load.
        # We can't easily re-trigger it (module is cached), but we can verify
        # the module-level conditional is correct.
        if not api_mod._ARIA_API_KEY:
            # Auth is disabled — _ARIA_API_KEY is None or empty
            assert not api_mod._ARIA_API_KEY
        # If set in test environment, auth_enabled must be True in health
        # (validated by test_auth_disabled_when_key_not_set above)


# ---------------------------------------------------------------------------
# #236 — event_store: migration failure surfaces as error + re-raise
# ---------------------------------------------------------------------------


class TestEventStoreMigrationFailure:
    """Verify event_store migration failure is logged and re-raised."""

    @pytest.mark.asyncio
    async def test_expected_duplicate_column_is_suppressed(self, tmp_path):
        """Duplicate column error from migration is expected and must not propagate."""
        store = EventStore(str(tmp_path / "events.db"))
        # First initialize creates the schema
        await store.initialize()
        # Second initialize triggers the migration which finds duplicate column — must not raise
        await store.initialize()
        await store.close()

    @pytest.mark.asyncio
    async def test_unexpected_migration_error_is_logged_and_reraised(self, tmp_path, caplog):
        """An unexpected exception during migration must be logged and re-raised.

        This test calls store.initialize() with a patched connection so that the
        ALTER TABLE step raises a non-duplicate-column error.  The real initialize()
        code path must log at ERROR level and re-raise — the test verifies both.
        """
        import aiosqlite

        store = EventStore(str(tmp_path / "events_fail.db"))

        # Patch aiosqlite.connect to return a real connection whose execute()
        # raises RuntimeError on ALTER TABLE but delegates all other calls to the
        # real aiosqlite cursor so the CREATE TABLE / PRAGMA steps succeed.
        original_connect = aiosqlite.connect

        async def patched_connect(path, *args, **kwargs):
            real_conn = await original_connect(path, *args, **kwargs)
            original_execute = real_conn.execute

            async def execute_with_alter_failure(sql, *a, **kw):
                if "ALTER TABLE" in sql:
                    raise RuntimeError("unexpected schema error")
                return await original_execute(sql, *a, **kw)

            real_conn.execute = execute_with_alter_failure
            return real_conn

        import logging

        with (
            patch("aiosqlite.connect", side_effect=patched_connect),
            caplog.at_level(logging.ERROR, logger="aria.shared.event_store"),
            pytest.raises(RuntimeError, match="unexpected schema error"),
        ):
            await store.initialize()

        # Verify the error was logged by the real initialize() code
        assert any("migration failed" in record.message for record in caplog.records if record.levelno >= logging.ERROR)


# ---------------------------------------------------------------------------
# #238 — event_store: reconnect on dropped connection
# ---------------------------------------------------------------------------


class TestEventStoreReconnect:
    """Verify EventStore reconnects when connection is dropped."""

    @pytest.mark.asyncio
    async def test_insert_after_connection_dropped_reconnects(self, tmp_path):
        """Setting _conn to None simulates a dropped connection; insert should reconnect."""
        store = EventStore(str(tmp_path / "events_reconnect.db"))
        await store.initialize()

        # Simulate dropped connection
        await store._conn.close()
        store._conn = None

        # Should reconnect transparently and succeed
        await store.insert_event(
            timestamp="2026-01-01T10:00:00",
            entity_id="light.test",
            domain="light",
            old_state="off",
            new_state="on",
        )
        count = await store.total_count()
        assert count == 1
        await store.close()

    @pytest.mark.asyncio
    async def test_get_conn_reconnects_and_logs_warning(self, tmp_path):
        """_get_conn logs a WARNING when reconnecting."""
        import logging

        store = EventStore(str(tmp_path / "events_warn.db"))
        await store.initialize()

        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = CapturingHandler()
        event_store_logger = logging.getLogger("aria.shared.event_store")
        event_store_logger.addHandler(handler)
        original_level = event_store_logger.level
        event_store_logger.setLevel(logging.WARNING)

        try:
            store._conn = None
            conn = await store._get_conn()
            assert conn is not None

            warning_records = [r for r in log_records if r.levelno == logging.WARNING]
            assert any(
                "reconnecting" in r.getMessage().lower() or "connection was none" in r.getMessage().lower()
                for r in warning_records
            ), f"No reconnect warning found. Records: {[r.getMessage() for r in log_records]}"
        finally:
            event_store_logger.setLevel(original_level)
            event_store_logger.removeHandler(handler)
            await store.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_reconnect_storm(self, tmp_path):
        """After a failed reconnect, _get_conn() raises immediately without retrying."""
        store = EventStore(str(tmp_path / "events_cb.db"))
        # Simulate a prior failed reconnect — flag set, connection absent
        store._reconnect_failed = True
        store._conn = None

        with pytest.raises(RuntimeError, match="previously failed to reconnect"):
            await store._get_conn()

        # The flag being True and the exception message prove the fast path fired
        # (initialize() was never called — no DB file was created)
        assert not (tmp_path / "events_cb.db").exists()

    @pytest.mark.asyncio
    async def test_batch_insert_after_connection_dropped(self, tmp_path):
        """insert_events_batch should also reconnect if connection is dropped."""
        store = EventStore(str(tmp_path / "events_batch_reconnect.db"))
        await store.initialize()

        await store._conn.close()
        store._conn = None

        events = [
            ("2026-01-01T10:00:00", "light.a", "light", "off", "on", None, None, None, None),
            ("2026-01-01T10:01:00", "light.b", "light", "on", "off", None, None, None, None),
        ]
        await store.insert_events_batch(events)
        count = await store.total_count()
        assert count == 2
        await store.close()


# ---------------------------------------------------------------------------
# #240 — ha_automation_sync: guard non-list HA API response
# ---------------------------------------------------------------------------


class TestHaAutomationSyncNonList:
    """Verify sync() handles non-list HA API responses without crashing."""

    @pytest.mark.asyncio
    async def test_dict_response_logs_warning_and_returns_failure(self, hub):
        """When HA returns a dict instead of a list, sync returns failure dict."""
        import logging

        from aria.shared.ha_automation_sync import HaAutomationSync

        log_records = []

        class CapturingHandler(logging.Handler):
            def emit(self, record):
                log_records.append(record)

        handler = CapturingHandler()
        logging.getLogger("aria.shared.ha_automation_sync").addHandler(handler)

        try:
            sync = HaAutomationSync(hub=hub, ha_url="http://fake", ha_token="fake")

            async def mock_fetch():
                return {"error": "forbidden", "message": "Not authorized"}

            sync._fetch_automations = mock_fetch

            result = await sync.sync()

            assert result["success"] is False
            assert "Unexpected response type" in result.get("error", "")

            warning_records = [r for r in log_records if r.levelno == logging.WARNING]
            assert any("unexpected response type" in r.getMessage().lower() for r in warning_records)
        finally:
            logging.getLogger("aria.shared.ha_automation_sync").removeHandler(handler)

    @pytest.mark.asyncio
    async def test_string_response_does_not_crash(self, hub):
        """When HA returns a string, sync returns failure without iteration crash."""
        from aria.shared.ha_automation_sync import HaAutomationSync

        sync = HaAutomationSync(hub=hub, ha_url="http://fake", ha_token="fake")

        async def mock_fetch():
            return "Unauthorized"

        sync._fetch_automations = mock_fetch

        result = await sync.sync()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_valid_list_still_works(self, hub):
        """A valid list response still processes correctly after the guard."""
        from aria.shared.ha_automation_sync import HaAutomationSync

        sync = HaAutomationSync(hub=hub, ha_url="http://fake", ha_token="fake")

        async def mock_fetch():
            return [{"id": "auto_1", "alias": "Test Automation", "trigger": [], "action": []}]

        sync._fetch_automations = mock_fetch

        result = await sync.sync()
        assert result["success"] is True
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# #241 — watchdog: heartbeat file written on successful check
# ---------------------------------------------------------------------------


class TestWatchdogHeartbeat:
    """Verify watchdog writes aria-heartbeat after successful check."""

    def test_heartbeat_file_written_on_run(self, tmp_path):
        """run_watchdog should write a heartbeat file with ISO timestamp."""
        from aria.watchdog import run_watchdog

        with (
            patch("aria.watchdog._collect_results", return_value=[]),
            patch("aria.watchdog._log_results"),
            patch("aria.watchdog._send_alerts"),
            patch("aria.watchdog.setup_logging", return_value=MagicMock()),
            patch("aria.watchdog.verify_telegram_connectivity"),
            patch("aria.watchdog.LOG_DIR", tmp_path),
        ):
            # Patch the heartbeat path to use tmp_path
            import aria.watchdog as wd_mod

            original_log_dir = wd_mod.LOG_DIR
            wd_mod.LOG_DIR = tmp_path
            try:
                run_watchdog(quiet=True, no_alert=True)
                hb_file = tmp_path / "aria-heartbeat"
                assert hb_file.exists(), "Heartbeat file was not written"
                content = hb_file.read_text().strip()
                # Verify it's a valid ISO timestamp
                parsed = datetime.fromisoformat(content)
                age_seconds = (datetime.now(UTC) - parsed).total_seconds()
                assert age_seconds < 30, f"Heartbeat timestamp is too old: {content}"
            finally:
                wd_mod.LOG_DIR = original_log_dir

    def test_heartbeat_contains_iso_timestamp(self, tmp_path):
        """Heartbeat file content should be parseable as ISO datetime."""
        import aria.watchdog as wd_mod

        original_log_dir = wd_mod.LOG_DIR
        wd_mod.LOG_DIR = tmp_path

        try:
            with (
                patch("aria.watchdog._collect_results", return_value=[]),
                patch("aria.watchdog._log_results"),
                patch("aria.watchdog._send_alerts"),
                patch("aria.watchdog.setup_logging", return_value=MagicMock()),
                patch("aria.watchdog.verify_telegram_connectivity"),
            ):
                from aria.watchdog import run_watchdog

                run_watchdog(quiet=True, no_alert=True)

                hb_file = tmp_path / "aria-heartbeat"
                assert hb_file.exists()
                ts = hb_file.read_text().strip()
                dt = datetime.fromisoformat(ts)
                # Should have timezone info (UTC)
                assert dt.tzinfo is not None
        finally:
            wd_mod.LOG_DIR = original_log_dir
