"""Tests for /api/transfer and /api/anomalies/explain endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import aria.hub.api as _api_module
from aria.hub.api import create_api

_TEST_API_KEY = "test-aria-key"


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.cache = MagicMock()
    hub.modules = {}
    hub.subscribers = {}
    hub.subscribe = MagicMock()
    hub._request_count = 0
    hub._audit_logger = None
    hub.get_uptime_seconds = MagicMock(return_value=0)
    hub.get_module = MagicMock(return_value=None)
    hub.get_cache = AsyncMock(return_value=None)
    return hub


@pytest.fixture
def client(mock_hub):
    original = _api_module._ARIA_API_KEY
    _api_module._ARIA_API_KEY = _TEST_API_KEY
    try:
        app = create_api(mock_hub)
        yield TestClient(app, headers={"X-API-Key": _TEST_API_KEY})
    finally:
        _api_module._ARIA_API_KEY = original


class TestTransferEndpoint:
    """Test GET /api/transfer."""

    def test_full_response(self, mock_hub, client):
        transfer_mod = MagicMock()
        transfer_mod.get_current_state.return_value = {
            "candidates": [
                {
                    "source_capability": "kitchen_lighting",
                    "target_context": "bedroom",
                    "state": "testing",
                    "hit_rate": 0.75,
                }
            ],
            "summary": {"total": 1, "by_state": {"testing": 1}},
        }
        transfer_mod.get_stats.return_value = {
            "active": True,
            "candidates_total": 1,
        }
        mock_hub.get_module.return_value = transfer_mod

        resp = client.get("/api/transfer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert len(data["candidates"]) == 1

    def test_module_not_available(self, mock_hub, client):
        mock_hub.get_module.return_value = None

        resp = client.get("/api/transfer")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["candidates"] == []


class TestAnomalyExplainEndpoint:
    """Test GET /api/anomalies/explain."""

    def test_with_explanations(self, mock_hub, client):
        pattern_mod = MagicMock()
        pattern_mod.get_current_state.return_value = {
            "anomaly_explanations": [
                {"feature": "power", "contribution": 0.45},
                {"feature": "lights", "contribution": 0.30},
            ],
            "trajectory": "ramping_up",
        }
        pattern_mod.attention_explainer = None
        mock_hub.get_module.return_value = pattern_mod

        resp = client.get("/api/anomalies/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["path_tracing"]) == 2
        assert data["attention"] is None

    def test_module_not_available(self, mock_hub, client):
        mock_hub.get_module.return_value = None

        resp = client.get("/api/anomalies/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path_tracing"] == []
        assert data["attention"] is None
