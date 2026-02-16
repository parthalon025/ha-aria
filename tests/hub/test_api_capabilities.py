"""Tests for /api/capabilities/registry endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aria.hub.api import create_api


@pytest.fixture
def client():
    """Create a test client with a mocked hub."""
    mock_hub = MagicMock()
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub.get_uptime_seconds = MagicMock(return_value=0)
    app = create_api(mock_hub)
    return TestClient(app)


class TestCapabilityRegistryAPI:
    def test_list_capabilities(self, client):
        resp = client.get("/api/capabilities/registry")
        assert resp.status_code == 200
        data = resp.json()
        assert "capabilities" in data
        assert "total" in data
        assert data["total"] >= 22
        assert "by_layer" in data
        assert "by_status" in data

    def test_list_filter_by_layer(self, client):
        resp = client.get("/api/capabilities/registry?layer=hub")
        assert resp.status_code == 200
        data = resp.json()
        for cap in data["capabilities"]:
            assert cap["layer"] == "hub"

    def test_list_filter_by_status(self, client):
        resp = client.get("/api/capabilities/registry?status=stable")
        assert resp.status_code == 200
        data = resp.json()
        for cap in data["capabilities"]:
            assert cap["status"] == "stable"

    def test_get_single_capability(self, client):
        resp = client.get("/api/capabilities/registry/discovery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "discovery"
        assert data["layer"] == "hub"

    def test_get_nonexistent_capability(self, client):
        resp = client.get("/api/capabilities/registry/nonexistent")
        assert resp.status_code == 404

    def test_capabilities_graph(self, client):
        resp = client.get("/api/capabilities/registry/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 22

    def test_capabilities_health(self, client):
        resp = client.get("/api/capabilities/registry/health")
        assert resp.status_code == 200
        data = resp.json()
        # Should have entries for all capabilities
        assert "discovery" in data
        assert "snapshot" in data
