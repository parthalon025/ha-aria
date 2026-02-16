"""Tests for shadow engine API endpoints."""

from unittest.mock import AsyncMock

# ============================================================================
# GET /api/shadow/predictions
# ============================================================================


class TestGetPredictions:
    def test_get_predictions_empty(self, api_hub, api_client):
        """Returns empty list when no predictions exist."""
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=[])

        response = api_client.get("/api/shadow/predictions")
        assert response.status_code == 200

        data = response.json()
        assert data["predictions"] == []
        assert data["count"] == 0

    def test_get_predictions_with_data(self, api_hub, api_client):
        """Returns predictions with correct format."""
        predictions = [
            {
                "id": "pred-001",
                "timestamp": "2026-02-12T10:00:00",
                "context": {"room": "living_room"},
                "predictions": [{"action": "light.turn_on"}],
                "outcome": "correct",
                "actual": {"event": "light.turn_on"},
                "confidence": 0.85,
                "is_exploration": False,
                "propagated_count": 0,
                "window_seconds": 300,
                "resolved_at": "2026-02-12T10:05:00",
            },
            {
                "id": "pred-002",
                "timestamp": "2026-02-12T09:00:00",
                "context": {"room": "kitchen"},
                "predictions": [{"action": "light.turn_off"}],
                "outcome": "disagreement",
                "actual": None,
                "confidence": 0.60,
                "is_exploration": False,
                "propagated_count": 0,
                "window_seconds": 300,
                "resolved_at": "2026-02-12T09:05:00",
            },
        ]
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=predictions)

        response = api_client.get("/api/shadow/predictions")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert data["predictions"][0]["id"] == "pred-001"
        assert data["predictions"][1]["id"] == "pred-002"

    def test_get_predictions_pagination(self, api_hub, api_client):
        """Limit and offset params are passed through."""
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=[])

        api_client.get("/api/shadow/predictions?limit=10&offset=5")

        api_hub.cache.get_recent_predictions.assert_called_once_with(limit=10, offset=5)

    def test_get_predictions_error(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get_recent_predictions = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/shadow/predictions")
        assert response.status_code == 500


# ============================================================================
# GET /api/shadow/accuracy
# ============================================================================


class TestGetAccuracy:
    def test_get_accuracy_empty(self, api_hub, api_client):
        """Returns zeroes when no data exists."""
        api_hub.cache.get_accuracy_stats = AsyncMock(
            return_value={
                "overall_accuracy": 0.0,
                "total_resolved": 0,
                "per_outcome": {},
                "daily_trend": [],
            }
        )
        api_hub.cache.get_pipeline_state = AsyncMock(
            return_value={
                "current_stage": "backtest",
                "stage_entered_at": "2026-02-12T00:00:00",
                "updated_at": "2026-02-12T00:00:00",
            }
        )

        response = api_client.get("/api/shadow/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["overall_accuracy"] == 0.0
        assert data["predictions_total"] == 0
        assert data["predictions_correct"] == 0
        assert data["predictions_disagreement"] == 0
        assert data["predictions_nothing"] == 0
        assert data["by_type"] == {}
        assert data["stage"] == "backtest"

    def test_get_accuracy_with_stats(self, api_hub, api_client):
        """Returns accuracy breakdown with correct mapping."""
        api_hub.cache.get_accuracy_stats = AsyncMock(
            return_value={
                "overall_accuracy": 0.75,
                "total_resolved": 20,
                "per_outcome": {
                    "correct": 15,
                    "disagreement": 3,
                    "nothing": 2,
                },
                "daily_trend": [
                    {"date": "2026-02-12", "correct": 15, "total": 20, "accuracy": 0.75},
                ],
            }
        )
        api_hub.cache.get_pipeline_state = AsyncMock(
            return_value={
                "current_stage": "shadow",
                "stage_entered_at": "2026-02-10T00:00:00",
                "updated_at": "2026-02-12T00:00:00",
            }
        )

        response = api_client.get("/api/shadow/accuracy")
        assert response.status_code == 200

        data = response.json()
        assert data["overall_accuracy"] == 0.75
        assert data["predictions_total"] == 20
        assert data["predictions_correct"] == 15
        assert data["predictions_disagreement"] == 3
        assert data["predictions_nothing"] == 2
        assert data["stage"] == "shadow"
        assert len(data["daily_trend"]) == 1

    def test_get_accuracy_error(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get_accuracy_stats = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/shadow/accuracy")
        assert response.status_code == 500


# ============================================================================
# GET /api/shadow/disagreements
# ============================================================================


class TestGetDisagreements:
    def test_get_disagreements_empty(self, api_hub, api_client):
        """Returns empty list when no disagreements exist."""
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=[])

        response = api_client.get("/api/shadow/disagreements")
        assert response.status_code == 200

        data = response.json()
        assert data["disagreements"] == []
        assert data["count"] == 0

    def test_get_disagreements_sorted_by_confidence(self, api_hub, api_client):
        """Returns disagreements sorted by confidence descending."""
        disagreements = [
            {"id": "p1", "confidence": 0.5, "outcome": "disagreement"},
            {"id": "p2", "confidence": 0.9, "outcome": "disagreement"},
            {"id": "p3", "confidence": 0.7, "outcome": "disagreement"},
        ]
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=disagreements)

        response = api_client.get("/api/shadow/disagreements")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3
        # Highest confidence first
        assert data["disagreements"][0]["id"] == "p2"
        assert data["disagreements"][1]["id"] == "p3"
        assert data["disagreements"][2]["id"] == "p1"

    def test_get_disagreements_respects_limit(self, api_hub, api_client):
        """Limit param caps the number of returned disagreements."""
        disagreements = [{"id": f"p{i}", "confidence": i * 0.1, "outcome": "disagreement"} for i in range(10)]
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=disagreements)

        response = api_client.get("/api/shadow/disagreements?limit=3")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 3

    def test_get_disagreements_uses_outcome_filter(self, api_hub, api_client):
        """Verifies the cache call uses outcome_filter='disagreement'."""
        api_hub.cache.get_recent_predictions = AsyncMock(return_value=[])

        api_client.get("/api/shadow/disagreements")

        api_hub.cache.get_recent_predictions.assert_called_once_with(limit=200, outcome_filter="disagreement")

    def test_get_disagreements_error(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get_recent_predictions = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/shadow/disagreements")
        assert response.status_code == 500


# ============================================================================
# GET /api/pipeline
# ============================================================================


class TestGetPipeline:
    def test_get_pipeline_not_initialized(self, api_hub, api_client):
        """Returns default state when pipeline is None."""
        api_hub.cache.get_pipeline_state = AsyncMock(return_value=None)

        response = api_client.get("/api/pipeline")
        assert response.status_code == 200

        data = response.json()
        assert data["current_stage"] == "shadow"
        assert data["gates"] == {}
        assert "message" in data

    def test_get_pipeline_with_state(self, api_hub, api_client):
        """Returns full pipeline state when initialized."""
        pipeline = {
            "id": 1,
            "current_stage": "shadow",
            "stage_entered_at": "2026-02-10T00:00:00",
            "backtest_accuracy": 0.92,
            "shadow_accuracy_7d": 0.85,
            "suggest_approval_rate_14d": None,
            "autonomous_contexts": None,
            "updated_at": "2026-02-12T10:00:00",
        }
        api_hub.cache.get_pipeline_state = AsyncMock(return_value=pipeline)

        response = api_client.get("/api/pipeline")
        assert response.status_code == 200

        data = response.json()
        assert data["current_stage"] == "shadow"
        assert data["backtest_accuracy"] == 0.92
        assert data["shadow_accuracy_7d"] == 0.85
        assert data["updated_at"] == "2026-02-12T10:00:00"

    def test_get_pipeline_error(self, api_hub, api_client):
        """Returns 500 on cache error."""
        api_hub.cache.get_pipeline_state = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/pipeline")
        assert response.status_code == 500


# ============================================================================
# POST /api/pipeline/advance
# ============================================================================


class TestPipelineAdvance:
    def _pipeline(self, stage="backtest", **overrides):
        """Build a pipeline state dict."""
        state = {
            "id": 1,
            "current_stage": stage,
            "stage_entered_at": "2026-02-10T00:00:00",
            "backtest_accuracy": None,
            "shadow_accuracy_7d": None,
            "suggest_approval_rate_14d": None,
            "autonomous_contexts": None,
            "updated_at": "2026-02-12T10:00:00",
        }
        state.update(overrides)
        return state

    def test_pipeline_advance_success(self, api_hub, api_client):
        """Advances from backtest to shadow when gate is met."""
        initial = self._pipeline("backtest", backtest_accuracy=0.55)
        advanced = self._pipeline("shadow", backtest_accuracy=0.55)

        api_hub.cache.get_pipeline_state = AsyncMock(side_effect=[initial, advanced])
        api_hub.cache.update_pipeline_state = AsyncMock()
        api_hub.publish = AsyncMock()

        response = api_client.post("/api/pipeline/advance")
        assert response.status_code == 200

        data = response.json()
        assert data["current_stage"] == "shadow"

        api_hub.cache.update_pipeline_state.assert_called_once()
        call_kwargs = api_hub.cache.update_pipeline_state.call_args[1]
        assert call_kwargs["current_stage"] == "shadow"
        assert "stage_entered_at" in call_kwargs

        api_hub.publish.assert_called_once_with("pipeline_updated", advanced)

    def test_pipeline_advance_gate_not_met(self, api_hub, api_client):
        """Returns 400 with structured error when gate threshold not met."""
        initial = self._pipeline("backtest", backtest_accuracy=0.25)

        api_hub.cache.get_pipeline_state = AsyncMock(return_value=initial)

        response = api_client.post("/api/pipeline/advance")
        assert response.status_code == 400

        data = response.json()
        assert data["error"] == "Gate not met"
        assert data["required"] == 0.40
        assert data["current"] == 0.25

    def test_pipeline_advance_at_final_stage(self, api_hub, api_client):
        """Returns 400 when already at autonomous."""
        initial = self._pipeline("autonomous")

        api_hub.cache.get_pipeline_state = AsyncMock(return_value=initial)

        response = api_client.post("/api/pipeline/advance")
        assert response.status_code == 400


# ============================================================================
# POST /api/pipeline/retreat
# ============================================================================


class TestPipelineRetreat:
    def _pipeline(self, stage="shadow", **overrides):
        """Build a pipeline state dict."""
        state = {
            "id": 1,
            "current_stage": stage,
            "stage_entered_at": "2026-02-10T00:00:00",
            "backtest_accuracy": None,
            "shadow_accuracy_7d": None,
            "suggest_approval_rate_14d": None,
            "autonomous_contexts": None,
            "updated_at": "2026-02-12T10:00:00",
        }
        state.update(overrides)
        return state

    def test_pipeline_retreat_success(self, api_hub, api_client):
        """Retreats from shadow to backtest with no gate check."""
        initial = self._pipeline("shadow")
        retreated = self._pipeline("backtest")

        api_hub.cache.get_pipeline_state = AsyncMock(side_effect=[initial, retreated])
        api_hub.cache.update_pipeline_state = AsyncMock()
        api_hub.publish = AsyncMock()

        response = api_client.post("/api/pipeline/retreat")
        assert response.status_code == 200

        data = response.json()
        assert data["current_stage"] == "backtest"

        api_hub.cache.update_pipeline_state.assert_called_once()
        call_kwargs = api_hub.cache.update_pipeline_state.call_args[1]
        assert call_kwargs["current_stage"] == "backtest"

        api_hub.publish.assert_called_once_with("pipeline_updated", retreated)

    def test_pipeline_retreat_at_first_stage(self, api_hub, api_client):
        """Returns 400 when already at backtest."""
        initial = self._pipeline("backtest")

        api_hub.cache.get_pipeline_state = AsyncMock(return_value=initial)

        response = api_client.post("/api/pipeline/retreat")
        assert response.status_code == 400
