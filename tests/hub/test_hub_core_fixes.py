"""Tests for hub-core issue fixes.

Covers:
  #304 — scope check: aria/modules/activity_monitor.py belongs to Hub-Modules
  #325 — unguarded WebSocket send_json after accept crashes on dirty connect
  #314 — API write endpoints unauthenticated when ARIA_API_KEY unset
  #324 — IntelligencePayload TypedDict contradicts REQUIRED_INTELLIGENCE_KEYS
  #315 — 15x direct hub.cache.get() bypasses metadata (replace with hub.get_cache())
  #316 — phantom cache keys in /api/ml/* — anomaly_alerts never populated
  #317 — Config PUT missing event bus publish
  #239 — audit_export uses deprecated asyncio.get_event_loop()
  #294 — Missing auth on /health endpoint
"""

import asyncio
import inspect
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import aria.hub.api as _api_module
from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub

_TEST_KEY = "test-aria-key"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_hub():
    """Minimal mock hub for API tests."""
    mock_hub = MagicMock(spec=IntelligenceHub)
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.module_status = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub._audit_logger = None
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_uptime_seconds = MagicMock(return_value=42)
    mock_hub.publish = AsyncMock()
    return mock_hub


@pytest.fixture
def hub():
    return _make_hub()


@pytest.fixture
def client(hub):
    original = _api_module._ARIA_API_KEY
    _api_module._ARIA_API_KEY = _TEST_KEY
    try:
        app = create_api(hub)
        yield TestClient(app, headers={"X-API-Key": _TEST_KEY})
    finally:
        _api_module._ARIA_API_KEY = original


# ---------------------------------------------------------------------------
# #304 — scope check: activity_monitor.py belongs to Hub-Modules
# ---------------------------------------------------------------------------


class TestIssue304ScopeCheck:
    """#304 json.loads in activity_monitor.py is Hub-Modules scope — not fixed here."""

    def test_activity_monitor_is_in_modules_directory_closes_304_scope_check(self):
        """Confirms aria/modules/activity_monitor.py is Hub-Modules scope, not Hub-Core."""
        import importlib.util
        from pathlib import Path

        spec = importlib.util.find_spec("aria.modules.activity_monitor")
        assert spec is not None, "aria.modules.activity_monitor not found"
        path = Path(spec.origin)
        assert "aria/modules" in str(path), f"activity_monitor.py not in aria/modules — path: {path}"
        # Confirm it has the unguarded json.loads (scope belongs to Hub-Modules team)
        src = path.read_text()
        assert "json.loads" in src, "json.loads not found in activity_monitor.py"


# ---------------------------------------------------------------------------
# #325 — WebSocket send_json after accept must not raise unhandled RuntimeError
# ---------------------------------------------------------------------------


class TestWebSocketSendGuard325:
    """audit_websocket initial send_json must be wrapped in try/except RuntimeError."""

    def test_audit_websocket_initial_send_json_guarded_closes_325(self):
        """audit_websocket initial send_json must be in try/except RuntimeError (line ~1846)."""
        src = inspect.getsource(_api_module)
        lines = src.splitlines()
        in_audit_ws = False
        found_initial_send = False
        found_runtime_guard = False
        for i, line in enumerate(lines):
            if "async def audit_websocket" in line:
                in_audit_ws = True
            if in_audit_ws and 'send_json({"type": "connected"' in line:
                found_initial_send = True
                # Check surrounding lines (±5) for RuntimeError guard
                context = "\n".join(lines[max(0, i - 4) : i + 6])
                if "RuntimeError" in context:
                    found_runtime_guard = True
                break

        assert found_initial_send, "audit_websocket initial send_json not found in source"
        assert found_runtime_guard, "audit_websocket initial send_json not wrapped in try/except RuntimeError — #325"

    def test_websocket_endpoint_handles_connect_closes_325(self, hub, client):
        """Normal WebSocket connect/disconnect must work without exception."""
        with client.websocket_connect("/ws?token=" + _TEST_KEY) as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"


# ---------------------------------------------------------------------------
# #314 — verify_api_key must reject when ARIA_API_KEY is unset
# ---------------------------------------------------------------------------


class TestApiAuthWhenKeyUnset314:
    """When ARIA_API_KEY is not set, write endpoints must NOT pass all traffic."""

    def test_put_config_returns_403_when_key_unset_closes_314(self, hub):
        """PUT /api/config/{key} returns 403 when ARIA_API_KEY is empty string."""
        original = _api_module._ARIA_API_KEY
        try:
            _api_module._ARIA_API_KEY = ""
            app = create_api(hub)
            no_auth_client = TestClient(app, raise_server_exceptions=False)
            hub.cache.set_config = AsyncMock(return_value={"key": "general.name", "value": "x"})
            response = no_auth_client.put("/api/config/general.name", json={"value": "x"})
            assert response.status_code == 403, (
                f"Expected 403, got {response.status_code} — write endpoint open when ARIA_API_KEY unset (#314)"
            )
        finally:
            _api_module._ARIA_API_KEY = original

    def test_verify_api_key_raises_when_key_is_empty_string_closes_314(self):
        """verify_api_key must raise 403 when _ARIA_API_KEY is empty string."""
        from fastapi import HTTPException

        original = _api_module._ARIA_API_KEY
        try:
            _api_module._ARIA_API_KEY = ""

            async def _run():
                with pytest.raises(HTTPException) as exc_info:
                    await _api_module.verify_api_key(key=None)
                assert exc_info.value.status_code == 403

            asyncio.run(_run())
        finally:
            _api_module._ARIA_API_KEY = original

    def test_valid_key_passes_auth_closes_314(self, hub, client):
        """A valid API key must pass verify_api_key."""
        hub.cache.get_all_config = AsyncMock(return_value=[])
        response = client.get("/api/config")
        assert response.status_code == 200, f"Expected 200 with valid key, got {response.status_code}"


# ---------------------------------------------------------------------------
# #324 — IntelligencePayload TypedDict must enforce required fields
# ---------------------------------------------------------------------------


class TestIntelligencePayloadTypedDict324:
    """IntelligencePayload required fields must be total=True (not total=False)."""

    def test_required_keys_are_not_all_optional_closes_324(self):
        """REQUIRED_INTELLIGENCE_KEYS fields must not be total=False optional."""
        from aria.schemas import IntelligencePayload

        required_keys = IntelligencePayload.__required_keys__
        assert len(required_keys) > 0, (
            "IntelligencePayload has no required keys (total=False) — "
            "all REQUIRED_INTELLIGENCE_KEYS fields are optional, contradicting the set (#324)"
        )
        # Structural keys used by hub cache assembly must be required
        assert "data_maturity" in required_keys, "data_maturity must be required in IntelligencePayload"
        assert "predictions" in required_keys, "predictions must be required in IntelligencePayload"
        assert "ml_models" in required_keys, "ml_models must be required in IntelligencePayload"

    def test_required_keys_match_required_intelligence_keys_set_closes_324(self):
        """__required_keys__ must cover all REQUIRED_INTELLIGENCE_KEYS entries."""
        from aria.schemas import REQUIRED_INTELLIGENCE_KEYS, IntelligencePayload

        required_in_typeddict = IntelligencePayload.__required_keys__
        missing_from_typeddict = REQUIRED_INTELLIGENCE_KEYS - set(required_in_typeddict)
        assert not missing_from_typeddict, (
            f"These keys are in REQUIRED_INTELLIGENCE_KEYS but optional in TypedDict: "
            f"{sorted(missing_from_typeddict)} (#324)"
        )


# ---------------------------------------------------------------------------
# #315 — hub.get_cache() called instead of hub.cache.get()
# ---------------------------------------------------------------------------


class TestHubGetCacheMethod315:
    """ML API endpoints must use hub.get_cache() not hub.cache.get() directly."""

    def test_get_ml_drift_uses_hub_get_cache_closes_315(self, hub, client):
        """GET /api/ml/drift must call hub.get_cache, not hub.cache.get directly."""
        hub.get_cache = AsyncMock(return_value=None)
        hub.get_cache = AsyncMock(return_value=None)

        response = client.get("/api/ml/drift")
        assert response.status_code == 200
        hub.get_cache.assert_awaited()  # hub.get_cache() must be called (#315)

    def test_get_ml_models_uses_hub_get_cache_closes_315(self, hub, client):
        """GET /api/ml/models must call hub.get_cache, not hub.cache.get directly."""
        hub.get_cache = AsyncMock(return_value=None)
        hub.get_cache = AsyncMock(return_value=None)

        response = client.get("/api/ml/models")
        assert response.status_code == 200
        hub.get_cache.assert_awaited()  # hub.get_cache() must be called (#315)

    def test_get_ml_anomalies_uses_hub_get_cache_closes_315(self, hub, client):
        """GET /api/ml/anomalies must call hub.get_cache, not hub.cache.get directly."""
        hub.get_cache = AsyncMock(return_value=None)
        hub.get_cache = AsyncMock(return_value=None)

        response = client.get("/api/ml/anomalies")
        assert response.status_code == 200
        hub.get_cache.assert_awaited()  # hub.get_cache() must be called (#315)


# ---------------------------------------------------------------------------
# #316 — phantom cache keys: anomaly_alerts, incremental_training, forecaster_backend
# ---------------------------------------------------------------------------


class TestPhantomCacheKeys316:
    """ML endpoints must not silently return null for keys never populated."""

    def test_ml_models_returns_populated_ml_models_closes_316(self, hub, client):
        """GET /api/ml/models must return ml_models when present in cache."""
        hub.get_cache = AsyncMock(
            return_value={
                "data": {
                    "ml_models": {"scores": {"lighting": 0.9}},
                    "reference_model": {"last_trained": "2026-01-01"},
                    # NOTE: "incremental_training" and "forecaster_backend" deliberately absent
                }
            }
        )

        response = client.get("/api/ml/models")
        assert response.status_code == 200
        data = response.json()
        assert data.get("ml_models") is not None, (
            "ml_models is null — endpoint reads phantom key 'incremental_training' (#316)"
        )
        assert data["ml_models"]["scores"]["lighting"] == pytest.approx(0.9)

    def test_ml_anomalies_reads_sequence_anomalies_key_closes_316(self, hub, client):
        """GET /api/ml/anomalies must map 'anomaly_alerts' phantom key to 'sequence_anomalies'."""
        hub.get_cache = AsyncMock(
            return_value={
                "data": {
                    "sequence_anomalies": {"anomalies": [{"entity": "light.living_room", "score": 0.9}]},
                    "autoencoder_status": {"enabled": True},
                    "isolation_forest_status": {"trained": True},
                }
            }
        )

        response = client.get("/api/ml/anomalies")
        assert response.status_code == 200
        data = response.json()
        # After fix: anomalies populated from sequence_anomalies
        assert isinstance(data.get("anomalies"), list), (
            "anomalies is not a list — still reading phantom key 'anomaly_alerts' (#316)"
        )
        assert len(data["anomalies"]) > 0, "anomalies list is empty despite cache having sequence_anomalies (#316)"


# ---------------------------------------------------------------------------
# #317 — Config PUT in routes_module_config.py missing event bus publish
# ---------------------------------------------------------------------------


class TestModuleConfigPublish317:
    """PUT /api/config/modules/{module}/sources must publish config_updated event."""

    def test_put_module_sources_publishes_config_updated_closes_317(self, hub, client):
        """After updating module sources, hub.publish('config_updated', ...) must be called."""
        hub.cache.set_config = AsyncMock()
        hub.cache.get_config = AsyncMock(return_value={"value": "camera_person,motion"})

        response = client.put(
            "/api/config/modules/presence/sources",
            json={"sources": ["camera_person", "motion"]},
        )

        assert response.status_code == 200
        hub.publish.assert_awaited()  # publish must be called after config update (#317)
        published_event_types = [call.args[0] for call in hub.publish.await_args_list]
        assert "config_updated" in published_event_types, (
            f"hub.publish was called but not with 'config_updated' — calls: {published_event_types} (#317)"
        )


# ---------------------------------------------------------------------------
# #239 — audit.py must use asyncio.get_running_loop() not get_event_loop()
# ---------------------------------------------------------------------------


class TestAuditEventLoopApi239:
    """AuditLogger must not use asyncio.get_event_loop() in async context."""

    def test_export_archive_uses_get_running_loop_closes_239(self):
        """export_archive must use asyncio.get_running_loop() not get_event_loop()."""
        from aria.hub import audit as audit_module

        src = inspect.getsource(audit_module.AuditLogger.export_archive)
        assert "get_event_loop" not in src, (
            "export_archive still uses asyncio.get_event_loop() — "
            "deprecated, raises RuntimeError in Python 3.13+ (#239)"
        )

    def test_write_dead_letter_uses_get_running_loop_closes_239(self):
        """_write_dead_letter must use asyncio.get_running_loop() not get_event_loop()."""
        from aria.hub import audit as audit_module

        src = inspect.getsource(audit_module.AuditLogger._write_dead_letter)
        assert "get_event_loop" not in src, "_write_dead_letter still uses asyncio.get_event_loop() — deprecated (#239)"


# ---------------------------------------------------------------------------
# #294 — /health must require auth when ARIA_API_KEY is set
# ---------------------------------------------------------------------------


class TestHealthEndpointAuth294:
    """GET /health must require auth when ARIA_API_KEY is set."""

    def test_health_endpoint_requires_auth_closes_294(self, hub):
        """GET /health without valid API key must return 403."""
        original = _api_module._ARIA_API_KEY
        try:
            _api_module._ARIA_API_KEY = "some-prod-key"
            app = create_api(hub)
            no_auth_client = TestClient(app, raise_server_exceptions=False)

            hub.health_check = AsyncMock(
                return_value={
                    "status": "ok",
                    "uptime_seconds": 100,
                    "modules": {"discovery": "running"},
                    "cache": {"categories": ["intelligence"]},
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )

            response = no_auth_client.get("/health")
            assert response.status_code == 403, (
                f"/health returned {response.status_code} without auth — "
                "exposes module/cache state to unauthenticated callers (#294)"
            )
        finally:
            _api_module._ARIA_API_KEY = original

    def test_health_endpoint_accessible_with_valid_key_closes_294(self, hub, client):
        """GET /health with valid API key must return 200."""
        hub.health_check = AsyncMock(
            return_value={
                "status": "ok",
                "uptime_seconds": 100,
                "modules": {},
                "cache": {"categories": []},
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# #313 — _prune_snapshot_log must write atomically (tmp + os.replace)
# ---------------------------------------------------------------------------


class TestAtomicSnapshotLogPrune313:
    """_prune_snapshot_log must use atomic write to prevent partial reads (#313)."""

    @pytest.mark.asyncio
    async def test_prune_uses_tmp_then_replace_closes_313(self, tmp_path):
        """Pruned snapshot_log is written via .tmp and os.replace — .tmp cleaned up (#313)."""
        import json
        from datetime import datetime, timedelta
        from unittest.mock import patch

        # Set up fake snapshot log mirroring real path structure
        log_dir = tmp_path / "ha-logs" / "intelligence"
        log_dir.mkdir(parents=True)
        log_file = log_dir / "snapshot_log.jsonl"

        old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        recent_ts = datetime.now(UTC).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": old_ts, "data": "old1"})
            + "\n"
            + json.dumps({"timestamp": old_ts, "data": "old2"})
            + "\n"
            + json.dumps({"timestamp": recent_ts, "data": "recent"})
            + "\n"
        )

        cache_path = str(tmp_path / "hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        try:
            with patch("pathlib.Path.home", return_value=tmp_path):
                pruned = await hub._prune_snapshot_log(retention_days=30)
        finally:
            await hub.shutdown()

        assert pruned == 2, f"Expected 2 pruned entries, got {pruned}"
        # .tmp must not linger after successful atomic replace (#313)
        assert not (log_dir / "snapshot_log.jsonl.tmp").exists(), (
            ".tmp file left on disk — atomic os.replace did not clean up (#313)"
        )
        remaining = [ln for ln in log_file.read_text().strip().split("\n") if ln]
        assert len(remaining) == 1, f"Expected 1 retained entry, got {len(remaining)}"
        assert json.loads(remaining[0])["data"] == "recent"
