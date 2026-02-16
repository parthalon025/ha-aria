"""Tests for new API feature endpoints: /api/version, /api/cache/keys, /api/metrics, /health."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    """Create a mock hub for API tests."""
    mock_hub = MagicMock(spec=IntelligenceHub)
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.module_status = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub.get_uptime_seconds = MagicMock(return_value=3600.0)
    return mock_hub


@pytest.fixture
def client(hub):
    """Create test client with mock hub."""
    app = create_api(hub)
    return TestClient(app)


# ============================================================================
# GET /api/version
# ============================================================================


class TestGetVersion:
    def test_returns_version_info(self, hub, client):
        """Returns version, package name, and Python version."""
        response = client.get("/api/version")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert data["package"] == "ha-aria"
        assert "python" in data
        # Python version should be X.Y.Z format
        parts = data["python"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_version_matches_package(self, hub, client):
        """Version matches what's in aria.__version__."""
        from aria import __version__

        response = client.get("/api/version")
        data = response.json()
        assert data["version"] == __version__


# ============================================================================
# GET /api/cache/keys
# ============================================================================


class TestGetCacheKeys:
    def test_empty_cache(self, hub, client):
        """Returns empty list when no cache categories exist."""
        hub.cache.list_categories = AsyncMock(return_value=[])

        response = client.get("/api/cache/keys")
        assert response.status_code == 200

        data = response.json()
        assert data["keys"] == []
        assert data["count"] == 0

    def test_with_categories(self, hub, client):
        """Returns categories with metadata."""
        hub.cache.list_categories = AsyncMock(return_value=["entities", "areas"])
        hub.cache.get = AsyncMock(
            side_effect=[
                {"last_updated": "2026-02-13T10:00:00", "version": 3, "data": {}},
                {"last_updated": "2026-02-13T09:00:00", "version": 1, "data": {}},
            ]
        )

        response = client.get("/api/cache/keys")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert data["keys"][0]["category"] == "entities"
        assert data["keys"][0]["version"] == 3
        assert data["keys"][1]["category"] == "areas"

    def test_category_with_no_entry(self, hub, client):
        """Handles categories where get returns None gracefully."""
        hub.cache.list_categories = AsyncMock(return_value=["empty_cat"])
        hub.cache.get = AsyncMock(return_value=None)

        response = client.get("/api/cache/keys")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert data["keys"][0]["last_updated"] is None
        assert data["keys"][0]["version"] is None


# ============================================================================
# GET /api/metrics
# ============================================================================


class TestGetMetrics:
    def test_returns_metrics(self, hub, client):
        """Returns all expected metric fields."""
        hub.cache.list_categories = AsyncMock(return_value=["a", "b", "c"])

        response = client.get("/api/metrics")
        assert response.status_code == 200

        data = response.json()
        assert data["cache_categories"] == 3
        assert data["uptime_seconds"] == 3600
        assert "requests_total" in data
        assert data["websocket_clients"] == 0

    def test_request_counter_increments(self, hub, client):
        """Request counter increments on each request."""
        hub.cache.list_categories = AsyncMock(return_value=[])

        # Make a few requests first
        client.get("/")
        client.get("/health")
        hub.health_check = AsyncMock(
            return_value={
                "status": "ok",
                "uptime_seconds": 0,
                "modules": {},
                "cache": {"categories": []},
                "timestamp": "t",
            }
        )

        response = client.get("/api/metrics")
        data = response.json()
        # request_count was 0, middleware incremented on each of the 3 requests above
        assert data["requests_total"] >= 3


# ============================================================================
# GET /health (enhanced)
# ============================================================================


class TestEnhancedHealth:
    def test_health_returns_module_status(self, hub, client):
        """Health endpoint returns module status and uptime."""
        hub.health_check = AsyncMock(
            return_value={
                "status": "ok",
                "uptime_seconds": 120,
                "modules": {
                    "discovery": "running",
                    "ml_engine": "running",
                    "shadow_engine": "failed",
                },
                "cache": {"categories": ["entities"]},
                "timestamp": "2026-02-13T10:00:00",
            }
        )

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["uptime_seconds"] == 120
        assert data["modules"]["discovery"] == "running"
        assert data["modules"]["shadow_engine"] == "failed"

    def test_health_error_handling(self, hub, client):
        """Health endpoint returns 500 when hub health check fails."""
        hub.health_check = AsyncMock(side_effect=RuntimeError("db error"))

        response = client.get("/health")
        assert response.status_code == 500
        assert response.json()["status"] == "error"


# ============================================================================
# Request timing middleware
# ============================================================================


class TestRequestTimingMiddleware:
    def test_increments_request_count(self, hub, client):
        """Middleware increments request count on each request."""
        initial = hub._request_count
        client.get("/")
        assert hub._request_count == initial + 1

    def test_multiple_requests_counted(self, hub, client):
        """Multiple requests increment counter correctly."""
        initial = hub._request_count
        for _ in range(5):
            client.get("/")
        assert hub._request_count == initial + 5
