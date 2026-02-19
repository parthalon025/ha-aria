"""Tests for GET /api/patterns endpoint."""

from unittest.mock import MagicMock


class TestPatternsEndpoint:
    """Test GET /api/patterns."""

    def test_full_response(self, api_hub, api_client):
        """Returns pattern data when module is active."""
        pattern_mod = MagicMock()
        pattern_mod.get_current_state.return_value = {
            "trajectory": "ramping_up",
            "pattern_scales": {"micro": "desc", "meso": "desc", "macro": "desc"},
            "anomaly_explanations": [{"feature": "power", "contribution": 0.45}],
            "shadow_events_processed": 42,
        }
        pattern_mod.get_stats.return_value = {
            "active": True,
            "sequence_classifier": {"is_trained": True},
            "window_count": {"power_watts": 6},
        }

        def mock_get_module(name):
            if name == "trajectory_classifier":
                return pattern_mod
            return None

        api_hub.get_module = MagicMock(side_effect=mock_get_module)

        resp = api_client.get("/api/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"] == "ramping_up"
        assert "anomaly_explanations" in data
        assert "stats" in data

    def test_module_not_available(self, api_hub, api_client):
        """Returns empty state when module not registered."""
        api_hub.get_module = MagicMock(return_value=None)

        resp = api_client.get("/api/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trajectory"] is None
        assert data["active"] is False

    def test_module_error(self, api_hub, api_client):
        """Returns 500 on unexpected error."""
        api_hub.get_module = MagicMock(side_effect=Exception("boom"))

        resp = api_client.get("/api/patterns")
        assert resp.status_code == 500
