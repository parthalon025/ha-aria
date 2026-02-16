"""Tests for config and curation API endpoints."""

from unittest.mock import AsyncMock

# ============================================================================
# GET /api/config
# ============================================================================


class TestGetAllConfig:
    def test_get_all_config_empty(self, api_hub, api_client):
        """Returns empty list when no config exists."""
        api_hub.cache.get_all_config = AsyncMock(return_value=[])

        response = api_client.get("/api/config")
        assert response.status_code == 200

        data = response.json()
        assert data["configs"] == []

    def test_get_all_config_with_data(self, api_hub, api_client):
        """Returns grouped config data."""
        configs = [
            {"key": "shadow.min_confidence", "value": "0.3", "source": "default"},
            {"key": "shadow.window_seconds", "value": "300", "source": "user"},
        ]
        api_hub.cache.get_all_config = AsyncMock(return_value=configs)

        response = api_client.get("/api/config")
        assert response.status_code == 200

        data = response.json()
        assert len(data["configs"]) == 2
        assert data["configs"][0]["key"] == "shadow.min_confidence"
        assert data["configs"][1]["key"] == "shadow.window_seconds"


# ============================================================================
# GET /api/config/{key}
# ============================================================================


class TestGetConfig:
    def test_get_config_found(self, api_hub, api_client):
        """Returns config dict when key exists."""
        config = {"key": "shadow.min_confidence", "value": "0.3", "source": "default"}
        api_hub.cache.get_config = AsyncMock(return_value=config)

        response = api_client.get("/api/config/shadow.min_confidence")
        assert response.status_code == 200

        data = response.json()
        assert data["key"] == "shadow.min_confidence"
        assert data["value"] == "0.3"

    def test_get_config_not_found(self, api_hub, api_client):
        """Returns 404 when key does not exist."""
        api_hub.cache.get_config = AsyncMock(return_value=None)

        response = api_client.get("/api/config/nonexistent.key")
        assert response.status_code == 404


# ============================================================================
# PUT /api/config/{key}
# ============================================================================


class TestPutConfig:
    def test_put_config_success(self, api_hub, api_client):
        """Updates config and returns result."""
        result = {"key": "shadow.min_confidence", "value": "0.5", "source": "user"}
        api_hub.cache.set_config = AsyncMock(return_value=result)

        response = api_client.put(
            "/api/config/shadow.min_confidence",
            json={"value": "0.5", "changed_by": "user"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["value"] == "0.5"
        api_hub.cache.set_config.assert_called_once_with("shadow.min_confidence", "0.5", changed_by="user")

    def test_put_config_validation_error(self, api_hub, api_client):
        """Returns 400 on ValueError from set_config."""
        api_hub.cache.set_config = AsyncMock(side_effect=ValueError("Value must be between 0 and 1"))

        response = api_client.put(
            "/api/config/shadow.min_confidence",
            json={"value": "99"},
        )
        assert response.status_code == 400
        assert "Value must be between 0 and 1" in response.json()["detail"]

    def test_put_config_not_found(self, api_hub, api_client):
        """Returns 400 when key doesn't exist (ValueError from set_config)."""
        api_hub.cache.set_config = AsyncMock(side_effect=ValueError("Unknown config key: bad.key"))

        response = api_client.put(
            "/api/config/bad.key",
            json={"value": "anything"},
        )
        assert response.status_code == 400

    def test_put_config_server_error(self, api_hub, api_client):
        """Returns 500 on unexpected error."""
        api_hub.cache.set_config = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.put(
            "/api/config/shadow.min_confidence",
            json={"value": "0.5"},
        )
        assert response.status_code == 500


# ============================================================================
# POST /api/config/reset/{key}
# ============================================================================


class TestResetConfig:
    def test_reset_config_success(self, api_hub, api_client):
        """Resets config and returns default value."""
        result = {"key": "shadow.min_confidence", "value": "0.3", "source": "default"}
        api_hub.cache.reset_config = AsyncMock(return_value=result)

        response = api_client.post("/api/config/reset/shadow.min_confidence")
        assert response.status_code == 200

        data = response.json()
        assert data["value"] == "0.3"
        assert data["source"] == "default"

    def test_reset_config_not_found(self, api_hub, api_client):
        """Returns 400 when key doesn't exist."""
        api_hub.cache.reset_config = AsyncMock(side_effect=ValueError("Unknown config key: bad.key"))

        response = api_client.post("/api/config/reset/bad.key")
        assert response.status_code == 400


# ============================================================================
# GET /api/config-history
# ============================================================================


class TestGetConfigHistory:
    def test_get_history_empty(self, api_hub, api_client):
        """Returns empty list when no history exists."""
        api_hub.cache.get_config_history = AsyncMock(return_value=[])

        response = api_client.get("/api/config-history")
        assert response.status_code == 200

        data = response.json()
        assert data["history"] == []
        assert data["count"] == 0

    def test_get_history_with_data(self, api_hub, api_client):
        """Returns history entries."""
        history = [
            {
                "key": "shadow.min_confidence",
                "old_value": "0.3",
                "new_value": "0.5",
                "changed_by": "user",
                "changed_at": "2026-02-12T10:00:00",
            },
        ]
        api_hub.cache.get_config_history = AsyncMock(return_value=history)

        response = api_client.get("/api/config-history")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 1
        assert data["history"][0]["key"] == "shadow.min_confidence"

    def test_get_history_filtered_by_key(self, api_hub, api_client):
        """Passes key filter param through to cache."""
        api_hub.cache.get_config_history = AsyncMock(return_value=[])

        api_client.get("/api/config-history?key=shadow.min_confidence&limit=10")

        api_hub.cache.get_config_history.assert_called_once_with(key="shadow.min_confidence", limit=10)


# ============================================================================
# GET /api/curation
# ============================================================================


class TestGetAllCuration:
    def test_get_all_curation_empty(self, api_hub, api_client):
        """Returns empty list when no curations exist."""
        api_hub.cache.get_all_curation = AsyncMock(return_value=[])

        response = api_client.get("/api/curation")
        assert response.status_code == 200

        data = response.json()
        assert data["curations"] == []

    def test_get_all_curation_with_data(self, api_hub, api_client):
        """Returns curation records."""
        curations = [
            {"entity_id": "light.living_room", "status": "tracked", "tier": "primary"},
            {"entity_id": "sensor.temp", "status": "excluded", "tier": "noise"},
        ]
        api_hub.cache.get_all_curation = AsyncMock(return_value=curations)

        response = api_client.get("/api/curation")
        assert response.status_code == 200

        data = response.json()
        assert len(data["curations"]) == 2
        assert data["curations"][0]["entity_id"] == "light.living_room"


# ============================================================================
# GET /api/curation/summary
# ============================================================================


class TestGetCurationSummary:
    def test_get_curation_summary(self, api_hub, api_client):
        """Returns tier/status count summary."""
        summary = {
            "by_tier": {"primary": 10, "secondary": 5, "noise": 20},
            "by_status": {"tracked": 15, "excluded": 20},
            "total": 35,
        }
        api_hub.cache.get_curation_summary = AsyncMock(return_value=summary)

        response = api_client.get("/api/curation/summary")
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 35
        assert data["by_tier"]["primary"] == 10


# ============================================================================
# PUT /api/curation/{entity_id}
# ============================================================================


class TestPutCuration:
    def test_put_curation_override(self, api_hub, api_client):
        """Upserts with human_override=True."""
        result = {
            "entity_id": "light.living_room",
            "status": "tracked",
            "decided_by": "user",
            "human_override": True,
        }
        api_hub.cache.upsert_curation = AsyncMock(return_value=result)

        response = api_client.put(
            "/api/curation/light.living_room",
            json={"status": "tracked", "decided_by": "user"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["human_override"] is True
        api_hub.cache.upsert_curation.assert_called_once_with(
            "light.living_room",
            status="tracked",
            decided_by="user",
            human_override=True,
        )


# ============================================================================
# POST /api/curation/bulk
# ============================================================================


class TestBulkUpdateCuration:
    def test_bulk_update_success(self, api_hub, api_client):
        """Returns count of updated entities."""
        api_hub.cache.bulk_update_curation = AsyncMock(return_value=3)

        response = api_client.post(
            "/api/curation/bulk",
            json={
                "entity_ids": ["light.a", "light.b", "light.c"],
                "status": "tracked",
                "decided_by": "user",
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["updated"] == 3

    def test_bulk_update_empty(self, api_hub, api_client):
        """Returns 0 when no entities provided."""
        api_hub.cache.bulk_update_curation = AsyncMock(return_value=0)

        response = api_client.post(
            "/api/curation/bulk",
            json={"entity_ids": [], "status": "tracked"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["updated"] == 0


# ============================================================================
# Error handling
# ============================================================================


class TestErrorHandling:
    def test_config_get_all_500_on_error(self, api_hub, api_client):
        """Config endpoints return 500 on cache exception."""
        api_hub.cache.get_all_config = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/config")
        assert response.status_code == 500

    def test_curation_get_all_500_on_error(self, api_hub, api_client):
        """Curation endpoints return 500 on cache exception."""
        api_hub.cache.get_all_curation = AsyncMock(side_effect=RuntimeError("db error"))

        response = api_client.get("/api/curation")
        assert response.status_code == 500
