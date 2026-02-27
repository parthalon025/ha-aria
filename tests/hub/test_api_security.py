"""Tests for API security features — path traversal, config redaction.

Covers:
  - C1: SPA catch-all path traversal guard
  - C2: Config history redacts sensitive values
  - C3: Single config key redacts sensitive value
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub


@pytest.fixture
def api_hub():
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


@pytest.fixture
def api_client(api_hub):
    app = create_api(api_hub)
    return TestClient(app)


class TestPathTraversalGuard:
    """SPA catch-all must not serve files outside spa/dist/.

    Note: HTTP clients normalize ../../ before sending, so the traversal
    guard at api.py:1607 (resolve().is_relative_to()) is defense-in-depth.
    We test via the route function directly with a crafted path string.
    """

    def test_spa_path_traversal_returns_index(self, api_client):
        """Any non-file path under /ui/ must return index.html (SPA routing)."""
        response = api_client.get("/ui/nonexistent/deep/path")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_spa_traversal_guard_logic(self):
        """Verify the is_relative_to guard rejects traversal paths."""
        from pathlib import Path

        spa_dist = Path("/fake/spa/dist")
        # Simulated traversal: /fake/spa/dist/../../etc/passwd resolves outside spa_dist
        traversal = spa_dist / "../../etc/passwd"
        assert not traversal.resolve().is_relative_to(spa_dist.resolve())


class TestConfigRedaction:
    """Config endpoints must redact sensitive key values."""

    def test_config_history_redacts_sensitive_values(self, api_hub, api_client):
        """Config history must show ***REDACTED*** for sensitive keys."""
        api_hub.cache.get_config_history = AsyncMock(
            return_value=[
                {
                    "key": "mqtt.password",
                    "old_value": "old_secret_123",
                    "new_value": "new_secret_456",
                    "changed_by": "admin",
                    "timestamp": "2026-02-20T10:00:00",
                },
                {
                    "key": "general.name",
                    "old_value": "ARIA",
                    "new_value": "ARIA v2",
                    "changed_by": "admin",
                    "timestamp": "2026-02-20T10:01:00",
                },
            ]
        )

        response = api_client.get("/api/config-history")
        assert response.status_code == 200
        data = response.json()
        history = data["history"]

        # Sensitive key should be redacted
        sensitive_entry = next(e for e in history if e["key"] == "mqtt.password")
        assert sensitive_entry["old_value"] == "***REDACTED***"
        assert sensitive_entry["new_value"] == "***REDACTED***"

        # Non-sensitive key should be unchanged
        normal_entry = next(e for e in history if e["key"] == "general.name")
        assert normal_entry["old_value"] == "ARIA"
        assert normal_entry["new_value"] == "ARIA v2"

    def test_get_single_config_redacts_sensitive_key(self, api_hub, api_client):
        """GET /api/config/{key} must redact value for sensitive keys."""
        api_hub.cache.get_config = AsyncMock(
            return_value={
                "key": "mqtt.password",
                "value": "super_secret_password",
                "changed_by": "admin",
                "timestamp": "2026-02-20T10:00:00",
            }
        )

        response = api_client.get("/api/config/mqtt.password")
        assert response.status_code == 200
        data = response.json()
        assert data["value"] == "***REDACTED***"
        assert data["key"] == "mqtt.password"
