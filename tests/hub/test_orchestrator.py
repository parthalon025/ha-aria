"""Unit tests for OrchestratorModule.

Tests pattern-to-automation suggestion generation, deduplication,
safety guardrails, approval/rejection flow, cache interaction,
lifecycle, and event handling.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.modules.orchestrator import OrchestratorModule

# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock for orchestrator tests."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._running = True
        self._scheduled_tasks: list[dict[str, Any]] = []
        self._published_events: list[dict[str, Any]] = []
        self.logger = MagicMock()
        self.modules = {}

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = {
            "data": data,
            "metadata": metadata,
            "last_updated": datetime.now().isoformat(),
        }

    async def get_cache(self, category: str) -> dict[str, Any] | None:
        return self._cache.get(category)

    async def get_cache_fresh(self, category: str, max_age=None, caller="") -> dict[str, Any] | None:
        return self._cache.get(category)

    def is_running(self) -> bool:
        return self._running

    async def schedule_task(self, **kwargs):
        self._scheduled_tasks.append(kwargs)

    def register_module(self, mod):
        self.modules[mod.module_id] = mod

    async def publish(self, event_type: str, data: dict[str, Any]):
        self._published_events.append({"event_type": event_type, "data": data})


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def module(hub):
    """Create an OrchestratorModule with mock hub and dummy HA credentials."""
    return OrchestratorModule(
        hub=hub,
        ha_url="http://192.168.1.35:8123",
        ha_token="fake-token",
        min_confidence=0.7,
    )


def make_pattern(  # noqa: PLR0913
    pattern_id="bedroom_cluster_1",
    name="Evening Bedroom",
    area="bedroom",
    typical_time="21:30",
    variance_minutes=15,
    frequency=5,
    total_days=7,
    confidence=0.85,
    associated_signals=None,
    llm_description="Bedroom lights on at 9:30pm",
):
    """Helper to build a pattern dictionary."""
    return {
        "pattern_id": pattern_id,
        "name": name,
        "area": area,
        "typical_time": typical_time,
        "variance_minutes": variance_minutes,
        "frequency": frequency,
        "total_days": total_days,
        "confidence": confidence,
        "associated_signals": associated_signals or ["bedroom_light_on_h21"],
        "llm_description": llm_description,
    }


async def seed_patterns(hub, patterns):
    """Seed pattern cache with given pattern list."""
    await hub.set_cache("patterns", {"patterns": patterns})


# ============================================================================
# Initialization & Lifecycle
# ============================================================================


class TestInitialization:
    """Test module constructor and init/shutdown lifecycle."""

    def test_module_id(self, module):
        """Module ID should be 'orchestrator'."""
        assert module.module_id == "orchestrator"

    def test_constructor_stores_config(self, module):
        """Constructor stores HA URL, token, and min_confidence."""
        assert module.ha_url == "http://192.168.1.35:8123"
        assert module.ha_token == "fake-token"
        assert module.min_confidence == 0.7

    def test_constructor_strips_trailing_slash(self, hub):
        """HA URL trailing slash is stripped."""
        mod = OrchestratorModule(hub, "http://ha:8123/", "token")
        assert mod.ha_url == "http://ha:8123"

    def test_default_min_confidence(self, hub):
        """Default min_confidence is 0.7."""
        mod = OrchestratorModule(hub, "http://ha:8123", "token")
        assert mod.min_confidence == 0.7

    @pytest.mark.asyncio
    async def test_initialize_creates_http_session(self, module, hub):
        """initialize() creates an aiohttp ClientSession."""
        await seed_patterns(hub, [])
        await module.initialize()

        assert module._session is not None
        # Verify scheduled task was registered
        assert len(hub._scheduled_tasks) == 1
        assert hub._scheduled_tasks[0]["task_id"] == "orchestrator_suggestions"

        await module.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_generates_initial_suggestions(self, module, hub):
        """initialize() generates suggestions from cached patterns."""
        await seed_patterns(hub, [make_pattern(confidence=0.9)])
        await module.initialize()

        cached = await hub.get_cache("automation_suggestions")
        assert cached is not None
        assert cached["data"]["count"] >= 1

        await module.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_handles_suggestion_failure(self, module, hub):
        """initialize() logs error but doesn't crash if suggestion generation fails."""
        # No patterns cache at all — will hit the "No patterns found" branch
        await module.initialize()

        # Initialization succeeded despite missing patterns — session was created
        assert module._session is not None
        # Scheduled task was still registered
        assert len(hub._scheduled_tasks) == 1

        await module.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_closes_session(self, module, hub):
        """shutdown() closes the HTTP session and sets it to None."""
        await seed_patterns(hub, [])
        await module.initialize()
        assert module._session is not None

        await module.shutdown()
        assert module._session is None

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize(self, module):
        """shutdown() is safe to call without initialize."""
        await module.shutdown()
        assert module._session is None


# ============================================================================
# Suggestion Generation
# ============================================================================


class TestSuggestionGeneration:
    """Test generate_suggestions with various pattern inputs."""

    @pytest.mark.asyncio
    async def test_generates_suggestion_from_pattern(self, module, hub):
        """A single pattern above threshold produces one suggestion."""
        await seed_patterns(hub, [make_pattern(confidence=0.85)])

        suggestions = await module.generate_suggestions()

        assert len(suggestions) == 1
        s = suggestions[0]
        assert s["pattern_id"] == "bedroom_cluster_1"
        assert s["status"] == "pending"
        assert "automation_yaml" in s
        assert s["automation_yaml"]["alias"] == "Pattern: Evening Bedroom"

    @pytest.mark.asyncio
    async def test_filters_below_confidence(self, module, hub):
        """Patterns below min_confidence are excluded."""
        patterns = [
            make_pattern(pattern_id="low", confidence=0.3),
            make_pattern(pattern_id="high", confidence=0.9),
        ]
        await seed_patterns(hub, patterns)

        suggestions = await module.generate_suggestions()

        assert len(suggestions) == 1
        assert suggestions[0]["pattern_id"] == "high"

    @pytest.mark.asyncio
    async def test_empty_patterns_returns_empty(self, module, hub):
        """Empty patterns list returns empty suggestions."""
        await seed_patterns(hub, [])

        suggestions = await module.generate_suggestions()
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_no_patterns_cache_returns_empty(self, module, hub):
        """Missing patterns cache returns empty suggestions."""
        suggestions = await module.generate_suggestions()
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_multiple_patterns_generate_multiple_suggestions(self, module, hub):
        """Each eligible pattern generates one suggestion."""
        patterns = [
            make_pattern(pattern_id="a", typical_time="07:00", confidence=0.8),
            make_pattern(pattern_id="b", typical_time="21:00", confidence=0.9),
        ]
        await seed_patterns(hub, patterns)

        suggestions = await module.generate_suggestions()
        assert len(suggestions) == 2

    @pytest.mark.asyncio
    async def test_stores_suggestions_in_cache(self, module, hub):
        """Generated suggestions are stored in automation_suggestions cache."""
        await seed_patterns(hub, [make_pattern()])

        await module.generate_suggestions()

        cached = await hub.get_cache("automation_suggestions")
        assert cached is not None
        data = cached["data"]
        assert data["count"] >= 1
        assert data["total_patterns"] == 1
        assert data["eligible_patterns"] == 1

    @pytest.mark.asyncio
    async def test_cache_metadata_includes_source(self, module, hub):
        """Cache metadata includes orchestrator source and min_confidence."""
        await seed_patterns(hub, [make_pattern()])

        await module.generate_suggestions()

        cached = await hub.get_cache("automation_suggestions")
        assert cached["metadata"]["source"] == "orchestrator"
        assert cached["metadata"]["min_confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_malformed_pattern_skipped(self, module, hub):
        """Patterns missing required fields are skipped with error logging."""
        patterns = [
            {"confidence": 0.9},  # missing pattern_id
            make_pattern(confidence=0.8),
        ]
        await seed_patterns(hub, patterns)

        suggestions = await module.generate_suggestions()
        # Good pattern should still produce a suggestion
        assert len(suggestions) == 1


# ============================================================================
# Suggestion Deduplication (Merge with Existing)
# ============================================================================


class TestSuggestionDeduplication:
    """Test merge logic that preserves approval status of existing suggestions."""

    @pytest.mark.asyncio
    async def test_preserves_approved_status(self, module, hub):
        """Regeneration preserves 'approved' status from previous run."""
        pattern = make_pattern()
        await seed_patterns(hub, [pattern])

        # First generation
        suggestions = await module.generate_suggestions()
        suggestion_id = suggestions[0]["suggestion_id"]

        # Manually mark as approved in cache
        cached = await hub.get_cache("automation_suggestions")
        cached["data"]["suggestions"][0]["status"] = "approved"
        cached["data"]["suggestions"][0]["automation_id"] = "pattern_abc"
        await hub.set_cache("automation_suggestions", cached["data"])

        # Regenerate — same pattern produces same suggestion_id
        suggestions2 = await module.generate_suggestions()
        assert suggestions2[0]["suggestion_id"] == suggestion_id
        assert suggestions2[0]["status"] == "approved"
        assert suggestions2[0]["automation_id"] == "pattern_abc"

    @pytest.mark.asyncio
    async def test_new_suggestions_start_pending(self, module, hub):
        """New suggestions that weren't in previous cache start as 'pending'."""
        await seed_patterns(hub, [make_pattern(pattern_id="new_one", typical_time="06:00")])

        suggestions = await module.generate_suggestions()
        assert suggestions[0]["status"] == "pending"


# ============================================================================
# Pattern-to-Suggestion Conversion
# ============================================================================


class TestPatternToSuggestion:
    """Test _pattern_to_suggestion automation YAML generation."""

    @pytest.mark.asyncio
    async def test_automation_yaml_structure(self, module):
        """Generated YAML has alias, description, trigger, condition, action."""
        pattern = make_pattern(typical_time="21:30")
        suggestion = await module._pattern_to_suggestion(pattern)

        yaml = suggestion["automation_yaml"]
        assert yaml["alias"] == "Pattern: Evening Bedroom"
        assert "Auto-generated" in yaml["description"]
        assert yaml["trigger"][0]["platform"] == "time"
        assert yaml["trigger"][0]["at"] == "21:30:00"
        assert yaml["condition"] == []
        assert len(yaml["action"]) >= 1

    @pytest.mark.asyncio
    async def test_suggestion_id_is_deterministic(self, module):
        """Same pattern_id + typical_time produces same suggestion_id."""
        pattern = make_pattern(pattern_id="test", typical_time="12:00")
        s1 = await module._pattern_to_suggestion(pattern)
        s2 = await module._pattern_to_suggestion(pattern)
        assert s1["suggestion_id"] == s2["suggestion_id"]

    @pytest.mark.asyncio
    async def test_suggestion_id_differs_for_different_times(self, module):
        """Different typical_time produces different suggestion_id."""
        s1 = await module._pattern_to_suggestion(make_pattern(typical_time="07:00"))
        s2 = await module._pattern_to_suggestion(make_pattern(typical_time="21:00"))
        assert s1["suggestion_id"] != s2["suggestion_id"]

    @pytest.mark.asyncio
    async def test_confidence_from_frequency_ratio(self, module):
        """Confidence = frequency / total_days."""
        pattern = make_pattern(frequency=5, total_days=10)
        suggestion = await module._pattern_to_suggestion(pattern)
        assert suggestion["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_confidence_zero_total_days(self, module):
        """Confidence is 0 when total_days is 0."""
        pattern = make_pattern(frequency=5, total_days=0)
        suggestion = await module._pattern_to_suggestion(pattern)
        assert suggestion["confidence"] == 0

    @pytest.mark.asyncio
    async def test_metadata_captured(self, module):
        """Suggestion metadata includes area, time, frequency, etc."""
        pattern = make_pattern(area="kitchen", typical_time="18:00", variance_minutes=20)
        suggestion = await module._pattern_to_suggestion(pattern)

        meta = suggestion["metadata"]
        assert meta["area"] == "kitchen"
        assert meta["typical_time"] == "18:00"
        assert meta["variance_minutes"] == 20


# ============================================================================
# Signal-to-Action Conversion
# ============================================================================


class TestSignalsToActions:
    """Test _signals_to_actions entity signal parsing."""

    def test_light_on_signal(self, module):
        """bedroom_light_on_h21 produces light.turn_on for bedroom."""
        actions = module._signals_to_actions("bedroom", ["bedroom_light_on_h21"])
        assert len(actions) == 1
        assert actions[0]["service"] == "light.turn_on"
        assert actions[0]["target"]["area_id"] == "bedroom"

    def test_light_off_signal(self, module):
        """bedroom_light_off_h23 produces light.turn_off for bedroom."""
        actions = module._signals_to_actions("bedroom", ["bedroom_light_off_h23"])
        assert len(actions) == 1
        assert actions[0]["service"] == "light.turn_off"

    def test_cross_area_signal_ignored(self, module):
        """Signals from a different area are ignored."""
        actions = module._signals_to_actions("bedroom", ["kitchen_light_on_h7"])
        # Should fall through to default notification
        assert len(actions) == 1
        assert actions[0]["service"] == "notify.persistent_notification"

    def test_short_signal_skipped(self, module):
        """Signals with fewer than 3 parts are skipped."""
        actions = module._signals_to_actions("bedroom", ["ab"])
        assert actions[0]["service"] == "notify.persistent_notification"

    def test_empty_signals_gets_default(self, module):
        """No signals produces default notification action."""
        actions = module._signals_to_actions("kitchen", [])
        assert len(actions) == 1
        assert actions[0]["service"] == "notify.persistent_notification"
        assert "kitchen" in actions[0]["data"]["message"].lower()

    def test_multiple_signals_multiple_actions(self, module):
        """Multiple matching signals produce multiple actions."""
        signals = ["bedroom_light_on_h21", "bedroom_light_off_h23"]
        actions = module._signals_to_actions("bedroom", signals)
        assert len(actions) == 2
        services = {a["service"] for a in actions}
        assert "light.turn_on" in services
        assert "light.turn_off" in services


# ============================================================================
# Safety Guardrails
# ============================================================================


class TestSafetyGuardrails:
    """Test _check_safety_guardrails for restricted domains."""

    def test_restricted_lock_domain(self, module):
        """lock.* service requires explicit approval."""
        actions = [{"service": "lock.lock", "target": {"entity_id": "lock.front_door"}}]
        assert module._check_safety_guardrails(actions) is True

    def test_restricted_cover_domain(self, module):
        """cover.* service requires explicit approval."""
        actions = [{"service": "cover.close_cover", "target": {}}]
        assert module._check_safety_guardrails(actions) is True

    def test_restricted_alarm_domain(self, module):
        """alarm_control_panel.* requires explicit approval."""
        actions = [{"service": "alarm_control_panel.arm_away", "target": {}}]
        assert module._check_safety_guardrails(actions) is True

    def test_safe_light_domain(self, module):
        """light.* does not require explicit approval."""
        actions = [{"service": "light.turn_on", "target": {}}]
        assert module._check_safety_guardrails(actions) is False

    def test_safe_notification_domain(self, module):
        """notify.* does not require explicit approval."""
        actions = [{"service": "notify.persistent_notification", "data": {}}]
        assert module._check_safety_guardrails(actions) is False

    def test_empty_actions(self, module):
        """Empty actions list is safe."""
        assert module._check_safety_guardrails([]) is False

    def test_mixed_safe_and_restricted(self, module):
        """If any action is restricted, the whole set is restricted."""
        actions = [
            {"service": "light.turn_on", "target": {}},
            {"service": "lock.lock", "target": {}},
        ]
        assert module._check_safety_guardrails(actions) is True

    def test_no_service_key(self, module):
        """Actions without 'service' key are treated as safe."""
        actions = [{"data": {"message": "test"}}]
        assert module._check_safety_guardrails(actions) is False


# ============================================================================
# Approval Flow
# ============================================================================


class TestApprovalFlow:
    """Test approve_suggestion and reject_suggestion."""

    @pytest.mark.asyncio
    async def test_approve_creates_automation(self, module, hub):
        """Approving a suggestion calls _create_automation and updates status."""
        await seed_patterns(hub, [make_pattern(confidence=0.9)])
        suggestions = await module.generate_suggestions()
        suggestion_id = suggestions[0]["suggestion_id"]

        # Mock _create_automation to succeed
        module._create_automation = AsyncMock(
            return_value={
                "success": True,
                "automation_id": f"pattern_{suggestion_id}",
            }
        )
        # Mock session to avoid real HTTP
        module._session = MagicMock()

        result = await module.approve_suggestion(suggestion_id)

        assert result["success"] is True
        assert result["automation_id"] == f"pattern_{suggestion_id}"

        # Verify event published
        approval_events = [e for e in hub._published_events if e["event_type"] == "automation_approved"]
        assert len(approval_events) == 1

    @pytest.mark.asyncio
    async def test_approve_nonexistent_suggestion(self, module, hub):
        """Approving a nonexistent suggestion returns error."""
        await hub.set_cache("automation_suggestions", {"suggestions": []})

        result = await module.approve_suggestion("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_approve_already_approved(self, module, hub):
        """Approving an already-approved suggestion returns error."""
        await hub.set_cache(
            "automation_suggestions",
            {
                "suggestions": [
                    {
                        "suggestion_id": "abc",
                        "pattern_id": "p1",
                        "status": "approved",
                        "automation_id": "pattern_abc",
                        "automation_yaml": {},
                    }
                ]
            },
        )

        result = await module.approve_suggestion("abc")
        assert result["success"] is False
        assert "already approved" in result["error"]

    @pytest.mark.asyncio
    async def test_approve_no_cache_returns_error(self, module, hub):
        """Approving with no suggestion cache returns error."""
        result = await module.approve_suggestion("abc")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_reject_suggestion(self, module, hub):
        """Rejecting a suggestion updates status and publishes event."""
        await hub.set_cache(
            "automation_suggestions",
            {
                "suggestions": [
                    {
                        "suggestion_id": "abc",
                        "pattern_id": "p1",
                        "status": "pending",
                        "automation_yaml": {},
                    }
                ]
            },
        )

        result = await module.reject_suggestion("abc")
        assert result["success"] is True

        # Verify status updated in cache
        cached = await hub.get_cache("automation_suggestions")
        assert cached["data"]["suggestions"][0]["status"] == "rejected"

        # Verify event published
        rejection_events = [e for e in hub._published_events if e["event_type"] == "automation_rejected"]
        assert len(rejection_events) == 1

    @pytest.mark.asyncio
    async def test_reject_nonexistent_suggestion(self, module, hub):
        """Rejecting a nonexistent suggestion returns error."""
        await hub.set_cache("automation_suggestions", {"suggestions": []})

        result = await module.reject_suggestion("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]


# ============================================================================
# Event Handling
# ============================================================================


class TestEventHandling:
    """Test on_event triggering suggestion regeneration."""

    @pytest.mark.asyncio
    async def test_patterns_cache_update_triggers_regeneration(self, module, hub):
        """cache_updated event for 'patterns' triggers generate_suggestions."""
        await seed_patterns(hub, [make_pattern(confidence=0.9)])

        await module.on_event("cache_updated", {"category": "patterns"})

        cached = await hub.get_cache("automation_suggestions")
        assert cached is not None
        assert cached["data"]["count"] >= 1

    @pytest.mark.asyncio
    async def test_non_patterns_cache_update_ignored(self, module, hub):
        """cache_updated for other categories does not trigger regeneration."""
        await module.on_event("cache_updated", {"category": "entities"})

        cached = await hub.get_cache("automation_suggestions")
        assert cached is None

    @pytest.mark.asyncio
    async def test_non_cache_event_ignored(self, module, hub):
        """Non cache_updated events are ignored."""
        await module.on_event("state_changed", {"entity_id": "light.kitchen"})

        cached = await hub.get_cache("automation_suggestions")
        assert cached is None

    @pytest.mark.asyncio
    async def test_event_handler_error_does_not_crash(self, module, hub):
        """Errors during event-driven regeneration are caught."""
        # Intentionally no assertion — verifies error resilience (no crash on missing data)
        # No patterns cache → generate_suggestions returns [] but doesn't crash
        await module.on_event("cache_updated", {"category": "patterns"})


# ============================================================================
# Get Suggestions
# ============================================================================


class TestGetSuggestions:
    """Test get_suggestions with status filtering."""

    @pytest.mark.asyncio
    async def test_get_all_suggestions(self, module, hub):
        """get_suggestions() returns all suggestions without filter."""
        await hub.set_cache(
            "automation_suggestions",
            {
                "suggestions": [
                    {"suggestion_id": "a", "status": "pending"},
                    {"suggestion_id": "b", "status": "approved"},
                ]
            },
        )

        result = await module.get_suggestions()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_pending_only(self, module, hub):
        """get_suggestions(status_filter='pending') returns only pending."""
        await hub.set_cache(
            "automation_suggestions",
            {
                "suggestions": [
                    {"suggestion_id": "a", "status": "pending"},
                    {"suggestion_id": "b", "status": "approved"},
                ]
            },
        )

        result = await module.get_suggestions(status_filter="pending")
        assert len(result) == 1
        assert result[0]["suggestion_id"] == "a"

    @pytest.mark.asyncio
    async def test_get_suggestions_empty_cache(self, module, hub):
        """get_suggestions() returns empty list when no cache exists."""
        result = await module.get_suggestions()
        assert result == []


# ============================================================================
# Created Automations Tracking
# ============================================================================


class TestCreatedAutomations:
    """Test get_created_automations tracking."""

    @pytest.mark.asyncio
    async def test_get_created_automations_empty(self, module, hub):
        """Returns empty dict when no automations tracked."""
        result = await module.get_created_automations()
        assert result == {}

    @pytest.mark.asyncio
    async def test_track_and_retrieve(self, module, hub):
        """_track_created_automation stores and get_created_automations retrieves."""
        await module._track_created_automation("pattern_abc", "suggestion_abc")

        result = await module.get_created_automations()
        assert "pattern_abc" in result
        assert result["pattern_abc"]["suggestion_id"] == "suggestion_abc"
        assert result["pattern_abc"]["status"] == "active"


# ============================================================================
# HTTP Automation Creation
# ============================================================================


class TestCreateAutomation:
    """Test _create_automation HTTP interaction."""

    @pytest.mark.asyncio
    async def test_successful_creation(self, module, hub):
        """HTTP 200 returns success."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        module._session = mock_session

        result = await module._create_automation("test_id", {"alias": "Test"})
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_401_stores_for_manual_creation(self, module, hub):
        """HTTP 401 stores automation for manual creation (still returns success)."""
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        module._session = mock_session

        result = await module._create_automation("test_id", {"alias": "Test"})
        assert result["success"] is True
        assert result["manual_creation_required"] is True

    @pytest.mark.asyncio
    async def test_500_returns_failure(self, module, hub):
        """HTTP 500 returns failure with error message."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Internal Server Error")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        module._session = mock_session

        result = await module._create_automation("test_id", {"alias": "Test"})
        assert result["success"] is False
        assert "500" in result["error"]

    @pytest.mark.asyncio
    async def test_network_error_returns_failure(self, module, hub):
        """Network error returns failure."""
        import aiohttp

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
        module._session = mock_session

        result = await module._create_automation("test_id", {"alias": "Test"})
        assert result["success"] is False
        assert "Network error" in result["error"]


# ============================================================================
# Pending Automations Storage
# ============================================================================


class TestPendingAutomations:
    """Test _store_pending_automation cache behavior."""

    @pytest.mark.asyncio
    async def test_stores_pending_automation(self, module, hub):
        """Stores automation YAML in pending_automations cache."""
        await module._store_pending_automation("auto_1", {"alias": "Test"})

        cached = await hub.get_cache("pending_automations")
        assert cached is not None
        assert "auto_1" in cached["data"]["automations"]
        assert cached["data"]["automations"]["auto_1"]["yaml"]["alias"] == "Test"

    @pytest.mark.asyncio
    async def test_appends_to_existing_pending(self, module, hub):
        """Multiple pending automations accumulate in cache."""
        await module._store_pending_automation("auto_1", {"alias": "First"})
        await module._store_pending_automation("auto_2", {"alias": "Second"})

        cached = await hub.get_cache("pending_automations")
        assert len(cached["data"]["automations"]) == 2


# ============================================================================
# Pattern Detection Sensor
# ============================================================================


class TestPatternSensor:
    """Test update_pattern_detection_sensor HA API call."""

    @pytest.mark.asyncio
    async def test_sensor_update_success(self, module):
        """Successful sensor update posts to HA states API."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        module._session = mock_session

        await module.update_pattern_detection_sensor("Evening Routine", "p1", 0.9)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "sensor.ha_hub_pattern_detected" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_sensor_update_failure_does_not_crash(self, module):
        """Sensor update failure is logged but doesn't raise."""
        # Intentionally no assertion — verifies error resilience (network failure doesn't propagate)
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("Network error"))
        module._session = mock_session

        # Should not raise
        await module.update_pattern_detection_sensor("Test", "p1", 0.5)
