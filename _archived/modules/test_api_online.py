"""Tests for GET /api/ml/online endpoint."""

from unittest.mock import MagicMock

# ============================================================================
# GET /api/ml/online
# ============================================================================


class TestGetOnlineLearningStats:
    def test_returns_full_data(self, api_hub, api_client):
        """Returns models, weight_tuner, and online_blend_weight."""
        online_learner = MagicMock()
        online_learner.get_all_stats.return_value = {
            "power_watts": {"samples": 150, "mae": 2.3},
        }

        ml_engine = MagicMock()
        ml_engine.weight_tuner.to_dict.return_value = {
            "ema_error_online": 3.1,
            "ema_error_batch": 4.5,
            "current_weight": 0.35,
        }
        ml_engine.online_blend_weight = 0.35

        def mock_get_module(name):
            return {"online_learner": online_learner, "ml_engine": ml_engine}.get(name)

        api_hub.get_module = MagicMock(side_effect=mock_get_module)

        response = api_client.get("/api/ml/online")
        assert response.status_code == 200

        data = response.json()
        assert data["models"]["power_watts"]["samples"] == 150
        assert data["weight_tuner"]["current_weight"] == 0.35
        assert data["online_blend_weight"] == 0.35

    def test_returns_defaults_when_modules_missing(self, api_hub, api_client):
        """Returns empty defaults when both modules are None."""
        api_hub.get_module = MagicMock(return_value=None)

        response = api_client.get("/api/ml/online")
        assert response.status_code == 200

        data = response.json()
        assert data["models"] == {}
        assert data["weight_tuner"] == {}
        assert data["online_blend_weight"] == 0.0

    def test_returns_defaults_when_ml_engine_lacks_attributes(self, api_hub, api_client):
        """Returns defaults when ml_engine exists but has no weight_tuner or blend_weight."""
        online_learner = MagicMock()
        online_learner.get_all_stats.return_value = {}

        ml_engine = MagicMock(spec=[])  # empty spec = no attributes

        def mock_get_module(name):
            return {"online_learner": online_learner, "ml_engine": ml_engine}.get(name)

        api_hub.get_module = MagicMock(side_effect=mock_get_module)

        response = api_client.get("/api/ml/online")
        assert response.status_code == 200

        data = response.json()
        assert data["models"] == {}
        assert data["weight_tuner"] == {}
        assert data["online_blend_weight"] == 0.0

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on unexpected error."""
        api_hub.get_module = MagicMock(side_effect=RuntimeError("module error"))

        response = api_client.get("/api/ml/online")
        assert response.status_code == 500
