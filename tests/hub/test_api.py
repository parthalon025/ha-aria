"""Tests for API fixes — can-predict cache bypass (#27)."""

from unittest.mock import AsyncMock

# ============================================================================
# #27: Toggle can-predict uses hub.set_cache (not hub.cache.set)
# ============================================================================


class TestToggleCanPredictCacheBypass:
    """PUT /api/capabilities/{name}/can-predict must use hub.set_cache for WS notifications."""

    def test_toggle_can_predict_uses_hub_set_cache(self, api_hub, api_client):
        """Toggle can_predict calls hub.set_cache (not hub.cache.set)."""
        # Set up mock: capabilities exist with a test capability
        api_hub.cache.get = AsyncMock(
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
        api_hub.cache.get = AsyncMock(return_value={"data": {"lighting": {"status": "promoted"}}})

        response = api_client.put(
            "/api/capabilities/nonexistent/can-predict",
            json={"can_predict": True},
        )

        assert response.status_code == 404

    def test_toggle_can_predict_no_capabilities(self, api_hub, api_client):
        """Returns 404 when capabilities cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.put(
            "/api/capabilities/lighting/can-predict",
            json={"can_predict": True},
        )

        assert response.status_code == 404

    def test_toggle_can_predict_invalid_value(self, api_hub, api_client):
        """Returns 400 when can_predict is not a boolean."""
        api_hub.cache.get = AsyncMock(return_value={"data": {"lighting": {"status": "promoted"}}})

        response = api_client.put(
            "/api/capabilities/lighting/can-predict",
            json={"can_predict": "yes"},
        )

        assert response.status_code == 400
