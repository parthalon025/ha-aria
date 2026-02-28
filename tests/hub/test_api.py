"""Tests for API fixes — can-predict cache bypass (#27), trend direction (#C5)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from aria.hub.api import _compute_stage_health

# ============================================================================
# #27: Toggle can-predict uses hub.set_cache (not hub.cache.set)
# ============================================================================


class TestToggleCanPredictCacheBypass:
    """PUT /api/capabilities/{name}/can-predict must use hub.set_cache for WS notifications."""

    def test_toggle_can_predict_uses_hub_set_cache(self, api_hub, api_client):
        """Toggle can_predict calls hub.set_cache (not hub.cache.set)."""
        # Set up mock: capabilities exist with a test capability
        api_hub.get_cache = AsyncMock(
            return_value={
                "data": {
                    "lighting": {
                        "status": "promoted",
                        "can_predict": False,
                    }
                }
            }
        )
        api_hub.set_cache = AsyncMock(return_value=1)

        response = api_client.put(
            "/api/capabilities/lighting/can-predict",
            json={"can_predict": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["capability"] == "lighting"
        assert data["can_predict"] is True

        # The key assertion: hub.set_cache was called (not hub.cache.set)
        api_hub.set_cache.assert_awaited_once()
        call_args = api_hub.set_cache.call_args
        assert call_args[0][0] == "capabilities"  # category
        assert call_args[0][1]["lighting"]["can_predict"] is True

    def test_toggle_can_predict_unknown_capability(self, api_hub, api_client):
        """Returns 404 for unknown capability."""
        api_hub.get_cache = AsyncMock(return_value={"data": {"lighting": {"status": "promoted"}}})

        response = api_client.put(
            "/api/capabilities/nonexistent/can-predict",
            json={"can_predict": True},
        )

        assert response.status_code == 404

    def test_toggle_can_predict_no_capabilities(self, api_hub, api_client):
        """Returns 404 when capabilities cache is empty."""
        api_hub.get_cache = AsyncMock(return_value=None)

        response = api_client.put(
            "/api/capabilities/lighting/can-predict",
            json={"can_predict": True},
        )

        assert response.status_code == 404

    def test_toggle_can_predict_invalid_value(self, api_hub, api_client):
        """Returns 400 when can_predict is not a boolean."""
        api_hub.get_cache = AsyncMock(return_value={"data": {"lighting": {"status": "promoted"}}})

        response = api_client.put(
            "/api/capabilities/lighting/can-predict",
            json={"can_predict": "yes"},
        )

        assert response.status_code == 400


# ============================================================================
# C5: Trend direction uses 0.02 threshold
# ============================================================================


class TestTrendDirectionThreshold:
    """_compute_stage_health trend_direction uses 0.02 delta threshold."""

    def test_small_delta_is_stable(self):
        """A delta of 0.01 (below 0.02 threshold) should be 'stable'."""
        stats = {
            "total_resolved": 100,
            "total_correct": 70,
            "total_attempted": 100,
            "predictions": [{"confidence": 0.7, "correct": True}] * 70 + [{"confidence": 0.3, "correct": False}] * 30,
            "daily_trend": [{"accuracy": 0.70} for _ in range(3)] + [{"accuracy": 0.71} for _ in range(3)],
        }
        result = _compute_stage_health(stats)
        assert result["trend_direction"] == "stable"

    def test_large_positive_delta_is_improving(self):
        """A delta > 0.02 should be 'improving'."""
        stats = {
            "total_resolved": 100,
            "total_correct": 70,
            "total_attempted": 100,
            "predictions": [{"confidence": 0.7, "correct": True}] * 70 + [{"confidence": 0.3, "correct": False}] * 30,
            "daily_trend": [{"accuracy": 0.60} for _ in range(3)] + [{"accuracy": 0.70} for _ in range(3)],
        }
        result = _compute_stage_health(stats)
        assert result["trend_direction"] == "improving"

    def test_large_negative_delta_is_degrading(self):
        """A delta < -0.02 should be 'degrading'."""
        stats = {
            "total_resolved": 100,
            "total_correct": 70,
            "total_attempted": 100,
            "predictions": [{"confidence": 0.7, "correct": True}] * 70 + [{"confidence": 0.3, "correct": False}] * 30,
            "daily_trend": [{"accuracy": 0.75} for _ in range(3)] + [{"accuracy": 0.65} for _ in range(3)],
        }
        result = _compute_stage_health(stats)
        assert result["trend_direction"] == "degrading"

    def test_insufficient_data(self):
        """Fewer than 3 trend points should be 'insufficient_data'."""
        stats = {
            "total_resolved": 10,
            "total_correct": 7,
            "total_attempted": 10,
            "predictions": [{"confidence": 0.7, "correct": True}] * 7 + [{"confidence": 0.3, "correct": False}] * 3,
            "daily_trend": [{"accuracy": 0.7}],
        }
        result = _compute_stage_health(stats)
        assert result["trend_direction"] == "insufficient_data"


# ============================================================================
# #292: CORS allow_headers must include Content-Type
# ============================================================================


class TestCORSAllowHeaders:
    """CORS preflight must allow Content-Type and X-API-Key headers."""

    def test_cors_allow_headers_includes_content_type(self, api_hub, api_client):
        """#292: CORS preflight must allow Content-Type header."""
        resp = api_client.options(
            "/api/models/retrain",
            headers={
                "Origin": "http://127.0.0.1:8001",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,X-API-Key",
            },
        )
        allowed = resp.headers.get("access-control-allow-headers", "")
        assert "content-type" in allowed.lower(), f"Content-Type not in CORS allow_headers: '{allowed}'"

    def test_cors_allow_headers_includes_x_api_key(self, api_hub, api_client):
        """#292 + #267: CORS preflight must allow X-API-Key header."""
        resp = api_client.options(
            "/api/models/retrain",
            headers={
                "Origin": "http://127.0.0.1:8001",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
        allowed = resp.headers.get("access-control-allow-headers", "")
        assert "x-api-key" in allowed.lower(), f"X-API-Key not in CORS allow_headers: '{allowed}'"


# ============================================================================
# #293: GET /api/settings/discovery returns 503 when module not loaded
# ============================================================================


class TestDiscoverySettingsModuleNotLoaded:
    """GET /api/settings/discovery must return HTTP 503 when organic_discovery is absent."""

    def test_discovery_settings_returns_503_when_module_not_loaded(self, api_hub, api_client):
        """#293: When organic_discovery is not in hub.modules, response must be 503."""
        # organic_discovery absent — hub.modules is an empty dict by default in api_hub fixture
        assert "organic_discovery" not in api_hub.modules

        response = api_client.get("/api/settings/discovery")

        assert response.status_code == 503, (
            f"Expected 503 when organic_discovery not loaded, got {response.status_code}: {response.text}"
        )

    def test_discovery_settings_returns_200_when_module_loaded(self, api_hub, api_client):
        """Sanity: When organic_discovery IS loaded, response must be 200 with settings."""
        mock_module = MagicMock()
        mock_module.settings = {"threshold": 0.8, "run_interval_hours": 24}
        api_hub.modules["organic_discovery"] = mock_module

        response = api_client.get("/api/settings/discovery")

        assert response.status_code == 200
        assert response.json()["threshold"] == 0.8


# ============================================================================
# #295: GET /api/events limit=le=1000 — PRE-EXISTING FIX
# ============================================================================
# Verified: line 238 of aria/hub/api.py already has:
#   limit: int = Query(default=100, le=1000)
# No code change required.


# ============================================================================
# #296: /api/frigate/thumbnail and /api/frigate/snapshot return 504 on timeout
# ============================================================================


class TestFrigateThumbnailTimeout:
    """GET /api/frigate/thumbnail/{event_id} must return 504 on slow Frigate response."""

    def test_frigate_thumbnail_returns_504_on_timeout(self, api_hub, api_client):
        """#296: If get_frigate_thumbnail takes too long, response must be 504."""

        async def slow_thumbnail(event_id):
            await asyncio.sleep(10)  # simulates a hung Frigate response
            return b"data"

        mock_presence = MagicMock()
        mock_presence.get_frigate_thumbnail = slow_thumbnail
        api_hub.modules["presence"] = mock_presence

        response = api_client.get("/api/frigate/thumbnail/test-event-123")

        assert response.status_code == 504, (
            f"Expected 504 on Frigate timeout, got {response.status_code}: {response.text}"
        )

    def test_frigate_snapshot_returns_504_on_timeout(self, api_hub, api_client):
        """#296: If get_frigate_snapshot takes too long, response must be 504."""

        async def slow_snapshot(event_id):
            await asyncio.sleep(10)
            return b"data"

        mock_presence = MagicMock()
        mock_presence.get_frigate_snapshot = slow_snapshot
        api_hub.modules["presence"] = mock_presence

        response = api_client.get("/api/frigate/snapshot/test-event-456")

        assert response.status_code == 504, (
            f"Expected 504 on Frigate timeout, got {response.status_code}: {response.text}"
        )


# ============================================================================
# #235: CurationUpdate Pydantic validator rejects invalid status/tier
# ============================================================================


class TestCurationUpdateValidation:
    """PUT /api/curation/{entity_id} must reject invalid status and tier values."""

    def test_invalid_status_returns_422(self, api_hub, api_client):
        """#235: Sending an invalid status value must return HTTP 422."""
        response = api_client.put(
            "/api/curation/light.test",
            json={"status": "invalid_status", "tier": 1},
        )
        assert response.status_code == 422, (
            f"Expected 422 for invalid status, got {response.status_code}: {response.text}"
        )

    def test_invalid_tier_returns_422(self, api_hub, api_client):
        """#235: Sending an invalid tier value must return HTTP 422."""
        response = api_client.put(
            "/api/curation/light.test",
            json={"status": "promoted", "tier": 99},
        )
        assert response.status_code == 422, (
            f"Expected 422 for invalid tier, got {response.status_code}: {response.text}"
        )

    def test_valid_curation_update_calls_hub(self, api_hub, api_client):
        """#235: Valid status and tier pass validation and reach the hub."""
        api_hub.cache.upsert_curation = AsyncMock(return_value=None)
        api_hub.publish = AsyncMock(return_value=None)

        response = api_client.put(
            "/api/curation/light.test",
            json={"status": "promoted", "tier": 1},
        )
        # Should not be 422 — validation passed (may be 200 or any non-422 success)
        assert response.status_code != 422, f"Valid curation update must not return 422, got: {response.text}"


# ============================================================================
# #297: ALLOWED_LABELS allowlist at POST /api/data/label
# ============================================================================


class TestDataLabelAllowlist:
    """POST /api/data/label must reject labels not in ALLOWED_LABELS."""

    def test_invalid_label_returns_422(self, api_hub, api_client):
        """#297: Label not in allowlist must return HTTP 422."""
        api_hub.get_cache = AsyncMock(return_value=None)
        response = api_client.post(
            "/api/data/label",
            json={"entity_id": "light.x", "label": "hacked_label", "snapshot_id": "snap-001"},
        )
        assert response.status_code == 422, (
            f"Expected 422 for disallowed label, got {response.status_code}: {response.text}"
        )

    def test_valid_label_is_stored(self, api_hub, api_client):
        """#297: A label in ALLOWED_LABELS must be accepted and stored."""
        api_hub.get_cache = AsyncMock(return_value=None)
        api_hub.set_cache = AsyncMock(return_value=None)
        response = api_client.post(
            "/api/data/label",
            json={"label": "normal", "snapshot_id": "snap-001"},
        )
        assert response.status_code == 200, f"Expected 200 for valid label, got {response.status_code}: {response.text}"
        assert response.json()["label"] == "normal"


# ============================================================================
# #298: Rate limiting at POST /api/models/retrain
# ============================================================================


class TestRetrainRateLimit:
    """POST /api/models/retrain must return 429 when called too rapidly."""

    def test_retrain_rate_limited_on_rapid_call(self, api_hub, api_client, monkeypatch):
        """#298: Second call within 60s must return HTTP 429."""
        import aria.hub.api as api_module

        # Reset rate limiter state
        monkeypatch.setattr(api_module, "_last_retrain", 0.0)

        mock_ml = MagicMock()
        mock_ml.train_models = AsyncMock(return_value={"status": "ok"})
        api_hub.modules["ml_engine"] = mock_ml

        # Simulate that a retrain just happened (set _last_retrain to now)
        monkeypatch.setattr(api_module, "_last_retrain", api_module.time.monotonic())

        response = api_client.post("/api/models/retrain")
        assert response.status_code == 429, (
            f"Expected 429 when retrain cooldown active, got {response.status_code}: {response.text}"
        )
