"""Unit tests for HaAutomationSync.

Tests periodic HA automation fetching, incremental hash-based normalization,
entity_id format normalization, and ha_automations cache storage.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.shared.ha_automation_sync import HaAutomationSync

# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock for automation sync tests."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self.logger = MagicMock()

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = {
            "data": data,
            "metadata": metadata,
            "last_updated": datetime.now().isoformat(),
        }

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        return self._cache.get(category)


# ============================================================================
# Fixtures
# ============================================================================


def make_ha_automation(  # noqa: PLR0913
    id: str = "automation.evening_lights",
    alias: str = "Evening Lights",
    trigger: list | None = None,
    action: list | None = None,
    condition: list | None = None,
    mode: str = "single",
    enabled: bool = True,
) -> dict[str, Any]:
    """Helper to build a mock HA automation config."""
    return {
        "id": id,
        "alias": alias,
        "trigger": trigger or [{"platform": "time", "at": "21:00:00"}],
        "action": action or [{"service": "light.turn_on", "target": {"area_id": "bedroom"}}],
        "condition": condition or [],
        "mode": mode,
        "enabled": enabled if enabled is not None else True,
    }


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    return session


def make_sync(hub, session=None):
    """Create an HaAutomationSync with mock dependencies."""
    return HaAutomationSync(
        hub=hub,
        ha_url="http://localhost:8123",
        ha_token="fake-token",
        session=session,
    )


def make_response(data, status=200):
    """Create a mock HTTP response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=data)
    resp.text = AsyncMock(return_value=json.dumps(data) if isinstance(data, list | dict) else str(data))
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


# ============================================================================
# Basic Sync
# ============================================================================


class TestBasicSync:
    """Test basic sync() behavior."""

    @pytest.mark.asyncio
    async def test_sync_fetches_and_stores(self, hub):
        """sync() fetches automations and stores normalized versions in cache."""
        automations = [make_ha_automation()]
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response(automations))

        sync = make_sync(hub, session)
        result = await sync.sync()

        assert result["success"] is True
        assert result["count"] == 1

        cached = await hub.get_cache("ha_automations")
        assert cached is not None
        assert len(cached["data"]["automations"]) == 1

    @pytest.mark.asyncio
    async def test_sync_stores_metadata(self, hub):
        """sync() stores sync metadata (timestamp, count)."""
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([make_ha_automation()]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        assert "last_sync" in cached["data"]
        assert cached["data"]["count"] == 1

    @pytest.mark.asyncio
    async def test_sync_empty_list(self, hub):
        """sync() handles empty automation list."""
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([]))

        sync = make_sync(hub, session)
        result = await sync.sync()

        assert result["success"] is True
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_sync_http_failure(self, hub):
        """sync() returns failure on HTTP error."""
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response("Server Error", status=500))

        sync = make_sync(hub, session)
        result = await sync.sync()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_sync_network_error(self, hub):
        """sync() returns failure on network exception."""
        import aiohttp

        session = AsyncMock()
        session.get = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

        sync = make_sync(hub, session)
        result = await sync.sync()

        assert result["success"] is False


# ============================================================================
# Force Sync
# ============================================================================


class TestForceSync:
    """Test force_sync() behavior."""

    @pytest.mark.asyncio
    async def test_force_sync_clears_hashes(self, hub):
        """force_sync() re-normalizes all automations regardless of hash."""
        automations = [make_ha_automation()]
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response(automations))

        sync = make_sync(hub, session)

        # First sync — caches hashes
        await sync.sync()
        assert len(sync._hashes) == 1

        # Force sync — should still process all automations
        result = await sync.force_sync()
        assert result["success"] is True
        assert result["count"] == 1
        assert result.get("changes", 0) == 1  # all re-processed


# ============================================================================
# Incremental Hash-Based Normalization
# ============================================================================


class TestIncrementalSync:
    """Test hash-based change detection."""

    @pytest.mark.asyncio
    async def test_unchanged_automation_not_renormalized(self, hub):
        """Automations with unchanged hash are not re-normalized."""
        automations = [make_ha_automation()]
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response(automations))

        sync = make_sync(hub, session)

        # First sync
        result1 = await sync.sync()
        assert result1["changes"] >= 1

        # Second sync — same data, no changes expected
        result2 = await sync.sync()
        assert result2["changes"] == 0

    @pytest.mark.asyncio
    async def test_changed_automation_is_renormalized(self, hub):
        """Automations with changed hash are re-normalized."""
        auto1 = make_ha_automation(alias="Version 1")
        auto2 = make_ha_automation(alias="Version 2")

        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto1]))

        sync = make_sync(hub, session)
        await sync.sync()

        # Second sync with changed data
        session.get = MagicMock(return_value=make_response([auto2]))
        result = await sync.sync()
        assert result["changes"] == 1

    @pytest.mark.asyncio
    async def test_new_automation_detected(self, hub):
        """New automations are detected and normalized."""
        auto1 = make_ha_automation(id="auto.first")
        auto2 = make_ha_automation(id="auto.second")

        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto1]))

        sync = make_sync(hub, session)
        await sync.sync()

        # Second sync with additional automation
        session.get = MagicMock(return_value=make_response([auto1, auto2]))
        result = await sync.sync()
        assert result["changes"] == 1  # only the new one

    @pytest.mark.asyncio
    async def test_removed_automation_cleaned_up(self, hub):
        """Removed automations are cleaned from cache."""
        auto1 = make_ha_automation(id="auto.first")
        auto2 = make_ha_automation(id="auto.second")

        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto1, auto2]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        assert len(cached["data"]["automations"]) == 2

        # Second sync with one automation removed
        session.get = MagicMock(return_value=make_response([auto1]))
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        assert len(cached["data"]["automations"]) == 1


# ============================================================================
# Entity ID Normalization
# ============================================================================


class TestEntityNormalization:
    """Test entity_id format normalization in automations."""

    @pytest.mark.asyncio
    async def test_normalizes_entity_ids_in_actions(self, hub):
        """Entity IDs in action targets are normalized to lowercase."""
        auto = make_ha_automation(action=[{"service": "light.turn_on", "target": {"entity_id": "Light.Bedroom_Main"}}])
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        actions = cached["data"]["automations"][0]["action"]
        entity = actions[0]["target"]["entity_id"]
        assert entity == "light.bedroom_main"

    @pytest.mark.asyncio
    async def test_normalizes_entity_ids_in_triggers(self, hub):
        """Entity IDs in triggers are normalized to lowercase."""
        auto = make_ha_automation(trigger=[{"platform": "state", "entity_id": "Binary_Sensor.Front_Door"}])
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        triggers = cached["data"]["automations"][0]["trigger"]
        assert triggers[0]["entity_id"] == "binary_sensor.front_door"

    @pytest.mark.asyncio
    async def test_normalizes_entity_id_lists(self, hub):
        """Entity ID lists in targets are normalized."""
        auto = make_ha_automation(
            action=[{"service": "light.turn_on", "target": {"entity_id": ["Light.A", "Light.B"]}}]
        )
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        entity_ids = cached["data"]["automations"][0]["action"][0]["target"]["entity_id"]
        assert entity_ids == ["light.a", "light.b"]

    @pytest.mark.asyncio
    async def test_normalizes_entity_ids_in_conditions(self, hub):
        """Entity IDs in conditions are normalized."""
        auto = make_ha_automation(condition=[{"condition": "state", "entity_id": "Person.Alice", "state": "home"}])
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        conditions = cached["data"]["automations"][0]["condition"]
        assert conditions[0]["entity_id"] == "person.alice"

    @pytest.mark.asyncio
    async def test_handles_missing_entity_ids_gracefully(self, hub):
        """Automations without entity_id fields don't crash normalization."""
        auto = make_ha_automation(
            trigger=[{"platform": "time", "at": "21:00:00"}],
            action=[{"service": "notify.persistent_notification", "data": {"message": "Hello"}}],
        )
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        result = await sync.sync()
        assert result["success"] is True


# ============================================================================
# Disabled Automations
# ============================================================================


class TestDisabledAutomations:
    """Test handling of disabled HA automations."""

    @pytest.mark.asyncio
    async def test_disabled_automations_preserved_with_flag(self, hub):
        """Disabled automations are stored with enabled=False flag."""
        auto = make_ha_automation(enabled=False)
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([auto]))

        sync = make_sync(hub, session)
        await sync.sync()

        cached = await hub.get_cache("ha_automations")
        stored = cached["data"]["automations"][0]
        assert stored.get("enabled") is False


# ============================================================================
# Immediate Cache Update (used by orchestrator)
# ============================================================================


class TestImmediateCacheUpdate:
    """Test add_automation() for immediate cache updates."""

    @pytest.mark.asyncio
    async def test_add_automation_to_empty_cache(self, hub):
        """add_automation() works when cache is empty."""
        sync = make_sync(hub)
        auto = make_ha_automation(id="pattern_abc")

        await sync.add_automation(auto)

        cached = await hub.get_cache("ha_automations")
        assert cached is not None
        assert len(cached["data"]["automations"]) == 1
        assert cached["data"]["automations"][0]["id"] == "pattern_abc"

    @pytest.mark.asyncio
    async def test_add_automation_to_existing_cache(self, hub):
        """add_automation() appends to existing cache."""
        # Pre-populate cache
        session = AsyncMock()
        session.get = MagicMock(return_value=make_response([make_ha_automation(id="existing")]))
        sync = make_sync(hub, session)
        await sync.sync()

        # Add new automation
        new_auto = make_ha_automation(id="pattern_new")
        await sync.add_automation(new_auto)

        cached = await hub.get_cache("ha_automations")
        assert len(cached["data"]["automations"]) == 2

    @pytest.mark.asyncio
    async def test_add_automation_normalizes_entities(self, hub):
        """add_automation() normalizes entity IDs."""
        sync = make_sync(hub)
        auto = make_ha_automation(
            id="pattern_abc",
            action=[{"service": "light.turn_on", "target": {"entity_id": "Light.Bedroom"}}],
        )

        await sync.add_automation(auto)

        cached = await hub.get_cache("ha_automations")
        entity = cached["data"]["automations"][0]["action"][0]["target"]["entity_id"]
        assert entity == "light.bedroom"

    @pytest.mark.asyncio
    async def test_add_automation_updates_existing(self, hub):
        """add_automation() replaces automation with same ID."""
        sync = make_sync(hub)

        await sync.add_automation(make_ha_automation(id="auto1", alias="V1"))
        await sync.add_automation(make_ha_automation(id="auto1", alias="V2"))

        cached = await hub.get_cache("ha_automations")
        assert len(cached["data"]["automations"]) == 1
        assert cached["data"]["automations"][0]["alias"] == "V2"


# ---------------------------------------------------------------------------
# Regression tests for #240 and #245
# ---------------------------------------------------------------------------


class TestIssue240TypeValidation:
    """#240 — non-list HA API response causes TypeError without isinstance guard."""

    @pytest.mark.asyncio
    async def test_non_list_response_returns_failure_closes_240(self, hub):
        """Sync must not raise TypeError when HA returns a dict instead of a list."""
        sync = HaAutomationSync(hub=hub, ha_url="http://ha", ha_token="tok")
        # Patch _fetch_automations to return a dict (simulates a malformed HA response)
        sync._fetch_automations = AsyncMock(return_value={"error": "not a list"})

        result = await sync.sync()

        assert result["success"] is False
        assert "Unexpected response type" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_none_response_returns_failure(self, hub):
        """Sync returns failure dict when _fetch_automations returns None."""
        sync = HaAutomationSync(hub=hub, ha_url="http://ha", ha_token="tok")
        sync._fetch_automations = AsyncMock(return_value=None)

        result = await sync.sync()

        assert result["success"] is False


class TestIssue245SessionNoneWarning:
    """#245 — _session=None returns None with no log — silent sync failure."""

    @pytest.mark.asyncio
    async def test_session_none_returns_empty_list_not_none_closes_245(self, hub):
        """_fetch_automations must return [] (not None) — sync completes without TypeError."""
        sync = HaAutomationSync(hub=hub, ha_url="http://ha", ha_token="tok")
        assert sync._session is None

        # Must NOT raise TypeError from iterating None; returns success with 0 automations
        result = await sync.sync()
        assert result.get("success") is True
        assert result.get("count", 0) == 0

    @pytest.mark.asyncio
    async def test_session_none_logs_warning_closes_245(self, hub, caplog):
        """_fetch_automations must emit a warning when _session is None."""
        import logging

        sync = HaAutomationSync(hub=hub, ha_url="http://ha", ha_token="tok")

        with caplog.at_level(logging.WARNING, logger="aria.shared.ha_automation_sync"):
            result = await sync.sync()

        assert any("session" in r.message.lower() or "initialized" in r.message.lower() for r in caplog.records), (
            "Expected a warning about session not being initialized"
        )
        assert result.get("success") is True  # empty list → success with 0 automations
