"""Tests for ML and shadow propagation API endpoints."""

from unittest.mock import AsyncMock, MagicMock

# ============================================================================
# GET /api/ml/drift
# ============================================================================


class TestGetMLDrift:
    def test_returns_defaults_when_no_intelligence(self, api_hub, api_client):
        """Returns safe defaults when intelligence cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.get("/api/ml/drift")
        assert response.status_code == 200

        data = response.json()
        assert data["needs_retrain"] is False
        assert data["days_analyzed"] == 0
        assert data["metrics"] == {}

    def test_returns_drift_data(self, api_hub, api_client):
        """Extracts drift status from intelligence cache."""
        api_hub.cache.get = AsyncMock(
            return_value={
                "drift_status": {
                    "needs_retrain": True,
                    "reason": "drift detected in power_watts",
                    "drifted_metrics": ["power_watts"],
                    "rolling_mae": {"power_watts": 12.5},
                    "current_mae": {"power_watts": 28.0},
                    "threshold": {"power_watts": 15.0},
                    "page_hinkley": {"power_watts": {"detected": True}},
                    "adwin": {"power_watts": {"detected": False}},
                    "days_analyzed": 7,
                }
            }
        )

        response = api_client.get("/api/ml/drift")
        assert response.status_code == 200

        data = response.json()
        assert data["needs_retrain"] is True
        assert data["reason"] == "drift detected in power_watts"
        assert data["drifted_metrics"] == ["power_watts"]
        assert data["rolling_mae"]["power_watts"] == 12.5
        assert data["current_mae"]["power_watts"] == 28.0
        assert data["days_analyzed"] == 7

    def test_returns_defaults_when_no_drift_status(self, api_hub, api_client):
        """Returns defaults when intelligence exists but has no drift_status."""
        api_hub.cache.get = AsyncMock(return_value={"some_other_key": "value"})

        response = api_client.get("/api/ml/drift")
        assert response.status_code == 200

        data = response.json()
        assert data["needs_retrain"] is False
        assert data["drifted_metrics"] == []
        assert data["days_analyzed"] == 0

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/ml/drift")
        assert response.status_code == 500


# ============================================================================
# GET /api/ml/features
# ============================================================================


class TestGetMLFeatures:
    def test_returns_defaults_when_no_intelligence(self, api_hub, api_client):
        """Returns safe defaults when intelligence cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.get("/api/ml/features")
        assert response.status_code == 200

        data = response.json()
        assert data["selected"] == []
        assert data["total"] == 0
        assert data["method"] == "none"

    def test_returns_feature_selection(self, api_hub, api_client):
        """Extracts feature selection from intelligence cache."""
        api_hub.cache.get = AsyncMock(
            return_value={
                "feature_selection": {
                    "selected_features": ["hour_sin", "power_watts_lag1", "day_of_week"],
                    "total_features": 48,
                    "method": "mrmr",
                    "max_features": 30,
                    "last_computed": "2026-02-13T03:00:00",
                }
            }
        )

        response = api_client.get("/api/ml/features")
        assert response.status_code == 200

        data = response.json()
        assert data["selected"] == ["hour_sin", "power_watts_lag1", "day_of_week"]
        assert data["total"] == 48
        assert data["method"] == "mrmr"
        assert data["max_features"] == 30
        assert data["last_computed"] == "2026-02-13T03:00:00"

    def test_returns_defaults_when_no_feature_selection(self, api_hub, api_client):
        """Returns defaults when intelligence exists but has no feature_selection."""
        api_hub.cache.get = AsyncMock(return_value={"some_other_key": "value"})

        response = api_client.get("/api/ml/features")
        assert response.status_code == 200

        data = response.json()
        assert data["selected"] == []
        assert data["total"] == 0
        assert data["method"] == "mrmr"

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/ml/features")
        assert response.status_code == 500


# ============================================================================
# GET /api/ml/models
# ============================================================================


class TestGetMLModels:
    def test_returns_defaults_when_no_intelligence(self, api_hub, api_client):
        """Returns None fields when intelligence cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.get("/api/ml/models")
        assert response.status_code == 200

        data = response.json()
        assert data["reference"] is None
        assert data["incremental"] is None
        assert data["forecaster"] is None
        assert data["ml_models"] is None

    def test_returns_model_data(self, api_hub, api_client):
        """Extracts model health from intelligence cache."""
        api_hub.cache.get = AsyncMock(
            return_value={
                "reference_model": {"r2": 0.85, "mae": 3.2},
                "incremental_training": {"last_batch": "2026-02-13", "samples": 500},
                "forecaster_backend": "prophet",
                "ml_models": {"gradient_boosting": {"r2": 0.82}, "random_forest": {"r2": 0.78}},
            }
        )

        response = api_client.get("/api/ml/models")
        assert response.status_code == 200

        data = response.json()
        assert data["reference"]["r2"] == 0.85
        assert data["incremental"]["samples"] == 500
        assert data["forecaster"] == "prophet"
        assert "gradient_boosting" in data["ml_models"]

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/ml/models")
        assert response.status_code == 500


# ============================================================================
# GET /api/ml/anomalies
# ============================================================================


class TestGetMLAnomalies:
    def test_returns_defaults_when_no_intelligence(self, api_hub, api_client):
        """Returns safe defaults when intelligence cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.get("/api/ml/anomalies")
        assert response.status_code == 200

        data = response.json()
        assert data["anomalies"] == []
        assert data["autoencoder"]["enabled"] is False
        assert data["isolation_forest"] == {}

    def test_returns_anomaly_data(self, api_hub, api_client):
        """Extracts anomaly data from intelligence cache."""
        api_hub.cache.get = AsyncMock(
            return_value={
                "anomaly_alerts": [
                    {"metric": "power_watts", "severity": "high", "timestamp": "2026-02-13T10:00:00"},
                ],
                "autoencoder_status": {"enabled": True, "reconstruction_error": 0.05},
                "isolation_forest_status": {"contamination": 0.01, "n_estimators": 100},
            }
        )

        response = api_client.get("/api/ml/anomalies")
        assert response.status_code == 200

        data = response.json()
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["metric"] == "power_watts"
        assert data["autoencoder"]["enabled"] is True
        assert data["isolation_forest"]["contamination"] == 0.01

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/ml/anomalies")
        assert response.status_code == 500


# ============================================================================
# GET /api/ml/shap
# ============================================================================


class TestGetMLSHAP:
    def test_returns_empty_when_no_intelligence(self, api_hub, api_client):
        """Returns unavailable when intelligence cache is empty."""
        api_hub.cache.get = AsyncMock(return_value=None)

        response = api_client.get("/api/ml/shap")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is False
        assert data["attributions"] == []

    def test_returns_attributions(self, api_hub, api_client):
        """Extracts SHAP attributions from intelligence cache."""
        api_hub.cache.get = AsyncMock(
            return_value={
                "shap_attributions": {
                    "attributions": [
                        {"feature": "hour_sin", "contribution": 0.15, "direction": "positive"},
                        {"feature": "power_watts_lag1", "contribution": -0.08, "direction": "negative"},
                    ],
                    "model_type": "GradientBoosting",
                    "computed_at": "2026-02-13T03:30:00",
                }
            }
        )

        response = api_client.get("/api/ml/shap")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is True
        assert len(data["attributions"]) == 2
        assert data["attributions"][0]["feature"] == "hour_sin"
        assert data["model_type"] == "GradientBoosting"
        assert data["computed_at"] == "2026-02-13T03:30:00"

    def test_returns_unavailable_when_no_shap_data(self, api_hub, api_client):
        """Returns unavailable when intelligence exists but has no SHAP data."""
        api_hub.cache.get = AsyncMock(return_value={"some_other_key": "value"})

        response = api_client.get("/api/ml/shap")
        assert response.status_code == 200

        data = response.json()
        assert data["available"] is False
        assert data["attributions"] == []

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/ml/shap")
        assert response.status_code == 500


# ============================================================================
# GET /api/shadow/propagation
# ============================================================================


class TestGetShadowPropagation:
    def test_returns_disabled_when_no_shadow_module(self, api_hub, api_client):
        """Returns disabled when shadow_engine module is not loaded."""
        api_hub.modules = {}

        response = api_client.get("/api/shadow/propagation")
        assert response.status_code == 200

        data = response.json()
        assert data["enabled"] is False
        assert data["stats"] == {}

    def test_returns_disabled_when_no_propagator(self, api_hub, api_client):
        """Returns disabled when shadow module has no propagator attribute."""
        shadow_mod = MagicMock()
        del shadow_mod.propagator  # ensure hasattr returns False
        api_hub.modules = {"shadow_engine": shadow_mod}

        response = api_client.get("/api/shadow/propagation")
        assert response.status_code == 200

        data = response.json()
        assert data["enabled"] is False

    def test_returns_disabled_when_propagator_is_none(self, api_hub, api_client):
        """Returns disabled when propagator is None."""
        shadow_mod = MagicMock()
        shadow_mod.propagator = None
        api_hub.modules = {"shadow_engine": shadow_mod}

        response = api_client.get("/api/shadow/propagation")
        assert response.status_code == 200

        data = response.json()
        assert data["enabled"] is False

    def test_returns_propagation_stats(self, api_hub, api_client):
        """Returns stats from propagator when available."""
        propagator = MagicMock()
        propagator._replay_buffer = [1, 2, 3]
        propagator.buffer_size = 100
        propagator._cell_observations = {"a": 1, "b": 2}
        propagator.bandwidth = 0.5

        shadow_mod = MagicMock()
        shadow_mod.propagator = propagator
        api_hub.modules = {"shadow_engine": shadow_mod}

        response = api_client.get("/api/shadow/propagation")
        assert response.status_code == 200

        data = response.json()
        assert data["enabled"] is True
        assert data["stats"]["replay_buffer_size"] == 3
        assert data["stats"]["replay_buffer_capacity"] == 100
        assert data["stats"]["cell_observations"] == 2
        assert data["stats"]["bandwidth"] == 0.5

    def test_error_returns_500(self, api_hub, api_client):
        """Returns 500 on unexpected error."""
        shadow_mod = MagicMock()
        shadow_mod.propagator = MagicMock(side_effect=RuntimeError("boom"))
        # Make hasattr return True but accessing propagator raises
        type(shadow_mod).propagator = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        api_hub.modules = {"shadow_engine": shadow_mod}

        response = api_client.get("/api/shadow/propagation")
        assert response.status_code == 500


# ============================================================================
# Cache key verification (all ML endpoints use "intelligence")
# ============================================================================


class TestMLEndpointsCacheKey:
    """Verify all ML endpoints query the correct cache category."""

    def test_drift_queries_intelligence(self, api_hub, api_client):
        api_hub.cache.get = AsyncMock(return_value=None)
        api_client.get("/api/ml/drift")
        api_hub.cache.get.assert_called_with("intelligence")

    def test_features_queries_intelligence(self, api_hub, api_client):
        api_hub.cache.get = AsyncMock(return_value=None)
        api_client.get("/api/ml/features")
        api_hub.cache.get.assert_called_with("intelligence")

    def test_models_queries_intelligence(self, api_hub, api_client):
        api_hub.cache.get = AsyncMock(return_value=None)
        api_client.get("/api/ml/models")
        api_hub.cache.get.assert_called_with("intelligence")

    def test_anomalies_queries_intelligence(self, api_hub, api_client):
        api_hub.cache.get = AsyncMock(return_value=None)
        api_client.get("/api/ml/anomalies")
        api_hub.cache.get.assert_called_with("intelligence")

    def test_shap_queries_intelligence(self, api_hub, api_client):
        api_hub.cache.get = AsyncMock(return_value=None)
        api_client.get("/api/ml/shap")
        api_hub.cache.get.assert_called_with("intelligence")
