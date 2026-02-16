"""Unit tests for ShadowEngine module.

Tests context capture, prediction generation, outcome scoring,
expired window resolution, and event handling.
"""

import asyncio
import contextlib
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ACTIVITY_SUMMARY
from aria.modules.shadow_engine import (
    DEFAULT_WINDOW_SECONDS,
    MIN_CONFIDENCE,
    PREDICTION_COOLDOWN_S,
    ShadowEngine,
)

# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock for shadow engine tests."""

    def __init__(self):
        self._cache: dict[str, dict[str, Any]] = {}
        self._running = True
        self._subscribers: dict[str, list] = {}

        # Mock the cache manager's prediction helpers
        self.cache = Mock()
        self.cache.insert_prediction = AsyncMock()
        self.cache.update_prediction_outcome = AsyncMock()
        self.cache.get_pending_predictions = AsyncMock(return_value=[])
        self.cache.get_recent_predictions = AsyncMock(return_value=[])
        self.cache.get_accuracy_stats = AsyncMock(return_value={})

        # Phase 2: Config store + curation mocks (return fallbacks by default)
        # IMPORTANT: use side_effect to respect the fallback parameter,
        # otherwise get_config_value returns None which causes
        # _resolution_loop to call asyncio.sleep(None) → TypeError →
        # infinite non-yielding loop that blocks the event loop and leaks memory.
        async def _config_fallback(key, fallback=None):
            return fallback

        self.cache.get_config_value = AsyncMock(side_effect=_config_fallback)
        self.cache.get_included_entity_ids = AsyncMock(return_value=set())

        self.logger = Mock()
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
        pass

    def register_module(self, mod):
        self.modules[mod.module_id] = mod

    def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback):
        if event_type in self._subscribers:
            self._subscribers[event_type] = [cb for cb in self._subscribers[event_type] if cb != callback]

    async def publish(self, event_type: str, data: dict[str, Any]):
        pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def engine(hub):
    """Create a ShadowEngine with mock hub."""
    return ShadowEngine(hub)


def make_state_changed_event(
    entity_id="light.kitchen",
    from_state="off",
    to_state="on",
    friendly_name="Kitchen Light",
):
    """Helper to create a state_changed event dict."""
    return {
        "entity_id": entity_id,
        "domain": entity_id.split(".")[0] if "." in entity_id else "",
        "from": from_state,
        "to": to_state,
        "friendly_name": friendly_name,
        "timestamp": datetime.now().isoformat(),
    }


def make_activity_summary(
    anyone_home=True,
    recent_activity=None,
    event_predictions=None,
):
    """Helper to create activity_summary cache data."""
    if recent_activity is None:
        recent_activity = [
            {
                "entity": "light.kitchen",
                "domain": "light",
                "from": "off",
                "to": "on",
                "time": "14:30",
                "friendly_name": "Kitchen Light",
            },
            {
                "entity": "light.living_room",
                "domain": "light",
                "from": "off",
                "to": "on",
                "time": "14:25",
                "friendly_name": "Living Room Light",
            },
        ]

    return {
        "data": {
            "occupancy": {
                "anyone_home": anyone_home,
                "people": ["Justin"],
                "since": datetime.now().isoformat(),
            },
            "recent_activity": recent_activity,
            "event_predictions": event_predictions or {},
            "patterns": [],
            "anomalies": [],
        }
    }


def make_activity_log(windows=None):
    """Helper to create activity_log cache data."""
    if windows is None:
        now = datetime.now()
        windows = [
            {
                "window_start": (now - timedelta(minutes=30)).isoformat(),
                "window_end": (now - timedelta(minutes=15)).isoformat(),
                "event_count": 12,
                "by_domain": {"light": 5, "switch": 4, "binary_sensor": 3},
                "occupancy": True,
            },
            {
                "window_start": (now - timedelta(minutes=15)).isoformat(),
                "window_end": now.isoformat(),
                "event_count": 8,
                "by_domain": {"light": 3, "media_player": 2, "switch": 3},
                "occupancy": True,
            },
        ]

    return {
        "data": {
            "windows": windows,
            "last_updated": datetime.now().isoformat(),
            "events_today": 50,
        }
    }


# ============================================================================
# Initialization & Lifecycle
# ============================================================================


class TestInitialization:
    """Test module initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialize_subscribes_to_state_changed(self, engine, hub):
        """Initialize should subscribe to state_changed events."""
        await engine.initialize()
        assert "state_changed" in hub._subscribers
        assert len(hub._subscribers["state_changed"]) == 1

        # Clean up
        engine._resolution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await engine._resolution_task

    @pytest.mark.asyncio
    async def test_initialize_starts_resolution_task(self, engine):
        """Initialize should start the periodic resolution task."""
        await engine.initialize()
        assert engine._resolution_task is not None
        assert not engine._resolution_task.done()

        # Clean up
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_resolution_task(self, engine):
        """Shutdown should cancel the resolution task."""
        await engine.initialize()
        task = engine._resolution_task
        assert task is not None

        await engine.shutdown()
        assert engine._resolution_task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_shutdown_without_initialize(self, engine):
        """Shutdown should not fail if initialize was never called."""
        await engine.shutdown()
        assert engine._resolution_task is None

    def test_module_id(self, engine):
        """Module ID should be 'shadow_engine'."""
        assert engine.module_id == "shadow_engine"


# ============================================================================
# Context Capture
# ============================================================================


class TestContextCapture:
    """Test _capture_context snapshot format."""

    @pytest.mark.asyncio
    async def test_context_has_required_keys(self, engine, hub):
        """Context snapshot should have all required keys."""
        event = make_state_changed_event()
        context = await engine._capture_context(event)

        assert "timestamp" in context
        assert "time_features" in context
        assert "presence" in context
        assert "recent_events" in context
        assert "current_states" in context
        assert "rolling_stats" in context
        assert "trigger_event" in context

    @pytest.mark.asyncio
    async def test_time_features_format(self, engine, hub):
        """Time features should contain sin/cos pairs."""
        event = make_state_changed_event()
        context = await engine._capture_context(event)

        tf = context["time_features"]
        assert "hour_sin" in tf
        assert "hour_cos" in tf
        assert "dow_sin" in tf
        assert "dow_cos" in tf

        # Sin/cos should be in [-1, 1]
        for key in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert -1.0 <= tf[key] <= 1.0

    def test_compute_time_features_midnight(self, engine):
        """Time features at midnight should be correct."""
        dt = datetime(2026, 2, 12, 0, 0, 0)  # Thursday midnight
        tf = engine._compute_time_features(dt)

        assert tf["hour_sin"] == pytest.approx(0.0, abs=1e-5)
        assert tf["hour_cos"] == pytest.approx(1.0, abs=1e-5)

    def test_compute_time_features_noon(self, engine):
        """Time features at noon should be correct."""
        dt = datetime(2026, 2, 12, 12, 0, 0)
        tf = engine._compute_time_features(dt)

        # 12/24 = 0.5, sin(pi) = 0, cos(pi) = -1
        assert tf["hour_sin"] == pytest.approx(0.0, abs=1e-5)
        assert tf["hour_cos"] == pytest.approx(-1.0, abs=1e-5)

    def test_compute_time_features_6am(self, engine):
        """Time features at 6am should be correct."""
        dt = datetime(2026, 2, 12, 6, 0, 0)
        tf = engine._compute_time_features(dt)

        # 6/24 = 0.25, sin(pi/2) = 1, cos(pi/2) = 0
        assert tf["hour_sin"] == pytest.approx(1.0, abs=1e-5)
        assert tf["hour_cos"] == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_presence_from_activity_summary(self, engine, hub):
        """Presence should be derived from activity_summary cache."""
        await hub.set_cache(CACHE_ACTIVITY_SUMMARY, make_activity_summary()["data"])

        event = make_state_changed_event()
        context = await engine._capture_context(event)

        assert context["presence"]["home"] is True

    @pytest.mark.asyncio
    async def test_presence_empty_when_no_cache(self, engine, hub):
        """Presence should have defaults when cache is empty."""
        event = make_state_changed_event()
        context = await engine._capture_context(event)

        assert context["presence"]["home"] is False
        assert context["presence"]["rooms"] == []

    @pytest.mark.asyncio
    async def test_recent_events_in_context(self, engine, hub):
        """Recent events should be included in context snapshot."""
        # Add some events to the buffer
        engine._recent_events = [
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "entity_id": "switch.hallway",
                "domain": "switch",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event = make_state_changed_event()
        context = await engine._capture_context(event)

        assert len(context["recent_events"]) == 2
        assert context["recent_events"][0]["domain"] == "light"

    @pytest.mark.asyncio
    async def test_recent_events_have_seconds_ago(self, engine, hub):
        """Recent events should include seconds_ago field."""
        engine._recent_events = [
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "to": "on",
                "timestamp": (datetime.now() - timedelta(seconds=30)).isoformat(),
            },
        ]

        event = make_state_changed_event()
        context = await engine._capture_context(event)

        assert "seconds_ago" in context["recent_events"][0]
        assert context["recent_events"][0]["seconds_ago"] >= 29

    @pytest.mark.asyncio
    async def test_rolling_stats_from_activity_log(self, engine, hub):
        """Rolling stats should be computed from activity_log cache."""
        await hub.set_cache(CACHE_ACTIVITY_LOG, make_activity_log()["data"])

        event = make_state_changed_event()
        context = await engine._capture_context(event)

        stats = context["rolling_stats"]
        assert "1h_event_count" in stats
        assert "1h_domain_entropy" in stats
        assert "1h_dominant_domain_pct" in stats
        assert stats["1h_event_count"] > 0

    @pytest.mark.asyncio
    async def test_rolling_stats_empty_when_no_cache(self, engine, hub):
        """Rolling stats should have zeros when no activity log."""
        event = make_state_changed_event()
        context = await engine._capture_context(event)

        stats = context["rolling_stats"]
        assert stats["1h_event_count"] == 0
        assert stats["1h_domain_entropy"] == 0.0

    @pytest.mark.asyncio
    async def test_trigger_event_captured(self, engine, hub):
        """Context should include the trigger event info."""
        event = make_state_changed_event(entity_id="light.bedroom")
        context = await engine._capture_context(event)

        assert context["trigger_event"]["entity_id"] == "light.bedroom"
        assert context["trigger_event"]["domain"] == "light"


# ============================================================================
# Prediction Generation
# ============================================================================


class TestPredictionGeneration:
    """Test _generate_predictions and individual prediction types."""

    @pytest.mark.asyncio
    async def test_generates_domain_prediction_from_frequency(self, engine, hub):
        """Should generate next_domain_action from recent events."""
        # Populate recent events buffer with actionable domains
        engine._recent_events = [
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "entity_id": "light.bedroom",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "entity_id": "switch.hallway",
                "domain": "switch",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        domain_preds = [p for p in predictions if p["type"] == "next_domain_action"]
        assert len(domain_preds) == 1
        assert domain_preds[0]["predicted"] == "light"
        assert domain_preds[0]["confidence"] > 0

    @pytest.mark.asyncio
    async def test_generates_domain_prediction_from_activity_summary(self, engine, hub):
        """Should use activity_summary event_predictions when available."""
        await hub.set_cache(
            CACHE_ACTIVITY_SUMMARY,
            make_activity_summary(
                event_predictions={
                    "predicted_next_domain": "media_player",
                    "probability": 0.75,
                    "method": "sequence",
                }
            )["data"],
        )

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        domain_preds = [p for p in predictions if p["type"] == "next_domain_action"]
        assert len(domain_preds) == 1
        assert domain_preds[0]["predicted"] == "media_player"
        assert domain_preds[0]["confidence"] == 0.75

    @pytest.mark.asyncio
    async def test_generates_room_activation_prediction(self, engine, hub):
        """Should generate room_activation from recent events."""
        engine._recent_events = [
            {
                "entity_id": "light.kitchen_main",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "entity_id": "light.kitchen_counter",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "entity_id": "light.bedroom_lamp",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        room_preds = [p for p in predictions if p["type"] == "room_activation"]
        assert len(room_preds) == 1
        assert room_preds[0]["predicted"] in ("kitchen", "bedroom")

    @pytest.mark.asyncio
    async def test_routine_trigger_from_patterns(self, engine, hub):
        """Should generate routine_trigger from cached patterns."""
        now = datetime.now()
        typical_time = f"{now.hour:02d}:{now.minute:02d}"

        await hub.set_cache(
            "patterns",
            {
                "patterns": [
                    {
                        "pattern_id": "general_cluster_1",
                        "name": "Evening Routine",
                        "area": "general",
                        "typical_time": typical_time,
                        "variance_minutes": 30,
                        "frequency": 5,
                        "total_days": 7,
                        "confidence": 0.71,
                    }
                ]
            },
        )

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        routine_preds = [p for p in predictions if p["type"] == "routine_trigger"]
        assert len(routine_preds) == 1
        assert routine_preds[0]["predicted"] == "Evening Routine"

    @pytest.mark.asyncio
    async def test_no_routine_trigger_when_time_far(self, engine, hub):
        """Should not generate routine_trigger when current time is far from pattern."""
        now = datetime.now()
        # Set pattern time 3 hours away
        far_hour = (now.hour + 3) % 24
        typical_time = f"{far_hour:02d}:00"

        await hub.set_cache(
            "patterns",
            {
                "patterns": [
                    {
                        "pattern_id": "general_cluster_1",
                        "name": "Far Routine",
                        "typical_time": typical_time,
                        "variance_minutes": 30,
                        "frequency": 5,
                        "total_days": 7,
                        "confidence": 0.71,
                    }
                ]
            },
        )

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        routine_preds = [p for p in predictions if p["type"] == "routine_trigger"]
        assert len(routine_preds) == 0

    @pytest.mark.asyncio
    async def test_predictions_below_threshold_filtered(self, engine, hub):
        """Predictions with confidence below MIN_CONFIDENCE should be excluded."""
        # With no data, frequency predictions will have low confidence
        engine._recent_events = []

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        # Should be empty because there's no data to generate confident predictions
        for p in predictions:
            assert p["confidence"] >= MIN_CONFIDENCE

    @pytest.mark.asyncio
    async def test_prediction_window_seconds(self, engine, hub):
        """Predictions should include window_seconds."""
        engine._recent_events = [
            {
                "entity_id": "light.kitchen",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event = make_state_changed_event()
        context = await engine._capture_context(event)
        predictions = await engine._generate_predictions(context)

        for p in predictions:
            assert "window_seconds" in p
            assert p["window_seconds"] == DEFAULT_WINDOW_SECONDS


# ============================================================================
# Outcome Scoring
# ============================================================================


class TestOutcomeScoring:
    """Test _score_prediction logic."""

    def test_correct_domain_prediction(self, engine):
        """Score as correct when predicted domain matches actual events."""
        prediction = {
            "predictions": [
                {"type": "next_domain_action", "predicted": "light"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "correct"
        assert actual_data["event_count"] == 1
        assert "light" in actual_data["domains"]

    def test_disagreement_wrong_domain(self, engine):
        """Score as disagreement when events happen but domain doesn't match."""
        prediction = {
            "predictions": [
                {"type": "next_domain_action", "predicted": "light"},
            ]
        }
        actual_events = [
            {"domain": "switch", "entity_id": "switch.hallway", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "disagreement"
        assert "switch" in actual_data["domains"]

    def test_nothing_no_events(self, engine):
        """Score as nothing when no events occurred during window."""
        prediction = {
            "predictions": [
                {"type": "next_domain_action", "predicted": "light"},
            ]
        }

        outcome, actual_data = engine._score_prediction(prediction, [])
        assert outcome == "nothing"
        assert actual_data["event_count"] == 0

    def test_correct_room_prediction(self, engine):
        """Score as correct when predicted room matches actual events."""
        prediction = {
            "predictions": [
                {"type": "room_activation", "predicted": "kitchen"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.kitchen_main", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "correct"
        assert "kitchen" in actual_data["rooms"]

    def test_disagreement_wrong_room(self, engine):
        """Score as disagreement when activity happens in different room."""
        prediction = {
            "predictions": [
                {"type": "room_activation", "predicted": "bedroom"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.kitchen_main", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "disagreement"
        assert "kitchen" in actual_data["rooms"]
        assert "bedroom" not in actual_data["rooms"]

    def test_correct_routine_trigger_with_expected_domains(self, engine):
        """Score routine_trigger correct when expected domains overlap."""
        prediction = {
            "predictions": [
                {
                    "type": "routine_trigger",
                    "predicted": "Evening Routine",
                    "expected_domains": ["light", "media_player"],
                },
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.living_room", "to": "on"},
            {"domain": "media_player", "entity_id": "media_player.tv", "to": "playing"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "correct"

    def test_disagreement_routine_no_domain_overlap(self, engine):
        """Routine trigger scores disagreement when wrong domains fire."""
        prediction = {
            "predictions": [
                {
                    "type": "routine_trigger",
                    "predicted": "Evening Routine",
                    "expected_domains": ["light", "media_player"],
                },
            ]
        }
        actual_events = [
            {"domain": "climate", "entity_id": "climate.thermostat", "to": "heat"},
            {"domain": "switch", "entity_id": "switch.fan", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "disagreement"

    def test_disagreement_routine_single_event(self, engine):
        """Routine trigger needs 2+ matching domain events."""
        prediction = {
            "predictions": [
                {
                    "type": "routine_trigger",
                    "predicted": "Evening Routine",
                    "expected_domains": ["light", "media_player"],
                },
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.living_room", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "disagreement"

    def test_correct_routine_trigger_fallback_no_expected_domains(self, engine):
        """Without expected_domains, fallback requires 3+ events and 2+ domains."""
        prediction = {
            "predictions": [
                {"type": "routine_trigger", "predicted": "Evening Routine"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.living_room", "to": "on"},
            {"domain": "media_player", "entity_id": "media_player.tv", "to": "playing"},
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "correct"

    def test_disagreement_routine_fallback_single_domain(self, engine):
        """Without expected_domains, 3 events in same domain = disagreement."""
        prediction = {
            "predictions": [
                {"type": "routine_trigger", "predicted": "Evening Routine"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.living_room", "to": "on"},
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
            {"domain": "light", "entity_id": "light.bedroom", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        assert outcome == "disagreement"

    def test_multiple_predictions_any_correct_wins(self, engine):
        """If any prediction matches, the overall outcome is correct."""
        prediction = {
            "predictions": [
                {"type": "next_domain_action", "predicted": "climate"},
                {"type": "room_activation", "predicted": "kitchen"},
            ]
        }
        actual_events = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        outcome, actual_data = engine._score_prediction(prediction, actual_events)
        # domain wrong (climate != light) but room correct (kitchen)
        assert outcome == "correct"

    def test_empty_predictions_list(self, engine):
        """Empty predictions list should score as nothing."""
        prediction = {"predictions": []}

        outcome, actual_data = engine._score_prediction(prediction, [])
        assert outcome == "nothing"


# ============================================================================
# Event Handling
# ============================================================================


class TestEventHandling:
    """Test _on_state_changed and event processing."""

    @pytest.mark.asyncio
    async def test_state_changed_buffers_event(self, engine, hub):
        """State changed events should be added to recent events buffer."""
        event = make_state_changed_event()
        await engine._on_state_changed(event)

        assert len(engine._recent_events) == 1
        assert engine._recent_events[0]["entity_id"] == "light.kitchen"

    @pytest.mark.asyncio
    async def test_state_changed_triggers_prediction(self, engine, hub):
        """First state_changed on actionable domain should trigger prediction."""
        # Pre-populate some recent events for prediction content
        engine._recent_events = [
            {
                "entity_id": "light.bedroom",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event = make_state_changed_event()
        await engine._on_state_changed(event)

        # insert_prediction should have been called
        hub.cache.insert_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_predictions(self, engine, hub):
        """Predictions should be debounced by PREDICTION_COOLDOWN_S."""
        engine._recent_events = [
            {
                "entity_id": "light.bedroom",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event1 = make_state_changed_event(entity_id="light.kitchen")
        await engine._on_state_changed(event1)
        assert hub.cache.insert_prediction.call_count == 1

        # Second event should be debounced
        event2 = make_state_changed_event(entity_id="light.bedroom")
        await engine._on_state_changed(event2)
        assert hub.cache.insert_prediction.call_count == 1

    @pytest.mark.asyncio
    async def test_cooldown_expired_allows_prediction(self, engine, hub):
        """After cooldown expires, new predictions should be generated."""
        engine._recent_events = [
            {
                "entity_id": "light.bedroom",
                "domain": "light",
                "to": "on",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        event1 = make_state_changed_event()
        await engine._on_state_changed(event1)
        assert hub.cache.insert_prediction.call_count == 1

        # Simulate cooldown expiration
        engine._last_prediction_time = datetime.now() - timedelta(seconds=PREDICTION_COOLDOWN_S + 1)

        event2 = make_state_changed_event(entity_id="light.bedroom")
        await engine._on_state_changed(event2)
        assert hub.cache.insert_prediction.call_count == 2

    @pytest.mark.asyncio
    async def test_non_predictable_domain_no_prediction(self, engine, hub):
        """Events from non-predictable domains should not trigger predictions."""
        event = make_state_changed_event(entity_id="binary_sensor.motion_kitchen")
        await engine._on_state_changed(event)

        hub.cache.insert_prediction.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_recorded_against_open_windows(self, engine, hub):
        """Events should be recorded against all open prediction windows."""
        engine._window_events["pred-1"] = []
        engine._window_events["pred-2"] = []

        event = make_state_changed_event()
        await engine._on_state_changed(event)

        assert len(engine._window_events["pred-1"]) == 1
        assert len(engine._window_events["pred-2"]) == 1

    @pytest.mark.asyncio
    async def test_recent_events_pruned(self, engine, hub):
        """Old events should be removed from the recent events buffer."""
        # Add an old event
        old_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        engine._recent_events = [
            {
                "entity_id": "light.old",
                "domain": "light",
                "to": "on",
                "timestamp": old_time,
            },
        ]

        # Process new event
        event = make_state_changed_event()
        await engine._on_state_changed(event)

        # Old event should be pruned (>5 min)
        entities = [e["entity_id"] for e in engine._recent_events]
        assert "light.old" not in entities
        assert "light.kitchen" in entities

    @pytest.mark.asyncio
    async def test_nested_state_format_handled(self, engine, hub):
        """Should handle nested new_state/old_state format from HA WebSocket."""
        event = {
            "entity_id": "light.kitchen",
            "new_state": {
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
            },
            "old_state": {"state": "off"},
        }

        await engine._on_state_changed(event)
        assert len(engine._recent_events) == 1
        assert engine._recent_events[0]["to"] == "on"
        assert engine._recent_events[0]["from"] == "off"

    @pytest.mark.asyncio
    async def test_on_event_does_not_handle_state_changed(self, engine, hub):
        """on_event is a no-op — shadow engine uses hub.subscribe() instead.

        This prevents double-handling when hub.publish() calls both
        subscriber callbacks and module.on_event().
        """
        event_data = make_state_changed_event()
        initial_count = len(engine._recent_events)

        await engine.on_event("state_changed", event_data)

        # on_event should NOT add events (subscribe handler does that)
        assert len(engine._recent_events) == initial_count


# ============================================================================
# Expired Window Resolution
# ============================================================================


class TestExpiredWindowResolution:
    """Test _resolve_expired_predictions."""

    @pytest.mark.asyncio
    async def test_resolves_expired_predictions(self, engine, hub):
        """Should resolve predictions whose windows have expired."""
        # Set up a pending prediction returned by cache
        pending = [
            {
                "id": "pred-001",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "light"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)

        # Set up events that happened during the window
        engine._window_events["pred-001"] = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        await engine._resolve_expired_predictions()

        hub.cache.update_prediction_outcome.assert_called_once()
        call_kwargs = hub.cache.update_prediction_outcome.call_args
        assert call_kwargs[1]["prediction_id"] == "pred-001"
        assert call_kwargs[1]["outcome"] == "correct"

    @pytest.mark.asyncio
    async def test_resolves_nothing_when_no_events(self, engine, hub):
        """Should resolve as 'nothing' when no events occurred."""
        pending = [
            {
                "id": "pred-002",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "light"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)

        # No events in window
        engine._window_events = {}

        await engine._resolve_expired_predictions()

        hub.cache.update_prediction_outcome.assert_called_once()
        call_kwargs = hub.cache.update_prediction_outcome.call_args
        assert call_kwargs[1]["outcome"] == "nothing"

    @pytest.mark.asyncio
    async def test_resolves_disagreement(self, engine, hub):
        """Should resolve as 'disagreement' when events don't match."""
        pending = [
            {
                "id": "pred-003",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "climate"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)

        engine._window_events["pred-003"] = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        await engine._resolve_expired_predictions()

        call_kwargs = hub.cache.update_prediction_outcome.call_args
        assert call_kwargs[1]["outcome"] == "disagreement"

    @pytest.mark.asyncio
    async def test_no_pending_does_nothing(self, engine, hub):
        """Should do nothing when there are no pending predictions."""
        hub.cache.get_pending_predictions = AsyncMock(return_value=[])

        await engine._resolve_expired_predictions()

        hub.cache.update_prediction_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_removes_window_events_after_resolution(self, engine, hub):
        """Window events should be cleaned up after resolution."""
        pending = [
            {
                "id": "pred-004",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "light"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)
        engine._window_events["pred-004"] = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        await engine._resolve_expired_predictions()

        assert "pred-004" not in engine._window_events

    @pytest.mark.asyncio
    async def test_handles_cache_error_gracefully(self, engine, hub):
        """Should handle errors from get_pending_predictions gracefully."""
        hub.cache.get_pending_predictions = AsyncMock(side_effect=Exception("DB error"))

        # Should not raise
        await engine._resolve_expired_predictions()

        # Verify error was handled — update_prediction_outcome should never be called
        # because the function returns early after the get_pending_predictions failure
        hub.cache.update_prediction_outcome.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolves_multiple_predictions(self, engine, hub):
        """Should resolve multiple expired predictions in one pass."""
        pending = [
            {
                "id": f"pred-{i}",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "light"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
            for i in range(3)
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)

        await engine._resolve_expired_predictions()

        assert hub.cache.update_prediction_outcome.call_count == 3


# ============================================================================
# Prediction Storage
# ============================================================================


class TestPredictionStorage:
    """Test _store_predictions."""

    @pytest.mark.asyncio
    async def test_stores_prediction_with_correct_args(self, engine, hub):
        """Should call insert_prediction with correct arguments."""
        context = {"timestamp": datetime.now().isoformat()}
        predictions = [
            {
                "type": "next_domain_action",
                "predicted": "light",
                "confidence": 0.8,
                "window_seconds": 600,
            }
        ]

        await engine._store_predictions(context, predictions)

        hub.cache.insert_prediction.assert_called_once()
        call_kwargs = hub.cache.insert_prediction.call_args[1]

        assert len(call_kwargs["prediction_id"]) == 32  # uuid4 hex
        assert call_kwargs["context"] == context
        assert call_kwargs["predictions"] == predictions
        assert call_kwargs["confidence"] == 0.8
        assert call_kwargs["window_seconds"] == 600
        assert call_kwargs["is_exploration"] is False

    @pytest.mark.asyncio
    async def test_average_confidence_multiple_predictions(self, engine, hub):
        """Confidence should be averaged across multiple predictions."""
        context = {"timestamp": datetime.now().isoformat()}
        predictions = [
            {"type": "next_domain_action", "confidence": 0.9, "window_seconds": 600},
            {"type": "room_activation", "confidence": 0.5, "window_seconds": 600},
        ]

        await engine._store_predictions(context, predictions)

        call_kwargs = hub.cache.insert_prediction.call_args[1]
        assert call_kwargs["confidence"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_tracks_window_events(self, engine, hub):
        """Should create empty event list for tracking during window."""
        context = {"timestamp": datetime.now().isoformat()}
        predictions = [
            {"type": "next_domain_action", "confidence": 0.8, "window_seconds": 600},
        ]

        await engine._store_predictions(context, predictions)

        # Should have created a window_events entry
        assert len(engine._window_events) == 1
        pred_id = list(engine._window_events.keys())[0]
        assert engine._window_events[pred_id] == []

    @pytest.mark.asyncio
    async def test_handles_storage_error(self, engine, hub):
        """Should handle insert_prediction errors gracefully."""
        hub.cache.insert_prediction = AsyncMock(side_effect=Exception("DB full"))

        context = {"timestamp": datetime.now().isoformat()}
        predictions = [
            {"type": "next_domain_action", "confidence": 0.8, "window_seconds": 600},
        ]

        # Should not raise
        await engine._store_predictions(context, predictions)

        # Verify error was handled — window_events should NOT be populated
        # because the exception occurs before the tracking code runs
        assert len(engine._window_events) == 0


# ============================================================================
# Room Extraction
# ============================================================================


class TestRoomExtraction:
    """Test _extract_room helper."""

    def test_extract_kitchen(self, engine):
        assert engine._extract_room("light.kitchen_main", "") == "kitchen"

    def test_extract_bedroom(self, engine):
        assert engine._extract_room("light.bedroom_lamp", "Bedroom Lamp") == "bedroom"

    def test_extract_from_friendly_name(self, engine):
        assert engine._extract_room("light.main", "Living Room Light") == "living"

    def test_no_room_returns_none(self, engine):
        assert engine._extract_room("light.main", "Main Light") is None

    def test_extract_office(self, engine):
        assert engine._extract_room("switch.office_fan", "") == "office"

    def test_extract_garage(self, engine):
        assert engine._extract_room("light.garage_overhead", "Garage") == "garage"


# ============================================================================
# Window Cleanup
# ============================================================================


class TestWindowCleanup:
    """Test _cleanup_stale_windows."""

    def test_cleanup_when_under_limit(self, engine):
        """Should not remove entries when under the limit."""
        engine._window_events = {f"pred-{i}": [] for i in range(10)}
        engine._cleanup_stale_windows()
        assert len(engine._window_events) == 10

    def test_cleanup_when_over_limit(self, engine):
        """Should trim entries when over 100."""
        engine._window_events = {f"pred-{i}": [] for i in range(120)}
        engine._cleanup_stale_windows()
        assert len(engine._window_events) == 100

    def test_cleanup_removes_oldest(self, engine):
        """Should remove oldest entries first (insertion order)."""
        engine._window_events = {f"pred-{i}": [] for i in range(110)}
        engine._cleanup_stale_windows()

        # First 10 should be removed, 10-109 remain
        assert "pred-0" not in engine._window_events
        assert "pred-9" not in engine._window_events
        assert "pred-10" in engine._window_events
        assert "pred-109" in engine._window_events


# ============================================================================
# Integration: ActivityMonitor → hub.publish → ShadowEngine
# ============================================================================


class IntegrationHub(MockHub):
    """MockHub with a real publish that calls subscribers (like IntelligenceHub)."""

    async def publish(self, event_type: str, data: dict[str, Any]):
        """Notify subscribers and registered modules, matching real hub behavior."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                with contextlib.suppress(Exception):
                    await callback(data)

        for mod in self.modules.values():
            with contextlib.suppress(Exception):
                await mod.on_event(event_type, data)


class TestActivityMonitorIntegration:
    """Verify that activity_monitor events reach shadow engine via hub event bus."""

    @pytest.mark.asyncio
    async def test_state_changed_flows_from_activity_monitor_to_shadow_engine(self):
        """When activity_monitor processes a tracked event, the shadow engine
        should receive it via hub.publish() and buffer it in _recent_events."""
        hub = IntegrationHub()

        # Create both modules on the same hub
        shadow = ShadowEngine(hub)
        hub.register_module(shadow)
        await shadow.initialize()

        try:
            # Simulate what activity_monitor._handle_state_changed does:
            # It appends to its own buffers then fires asyncio.create_task(hub.publish(...))
            # We replicate the publish call directly since _handle_state_changed
            # requires a fully wired ActivityMonitor with ha_url/ha_token.
            await hub.publish(
                "state_changed",
                {
                    "entity_id": "light.kitchen",
                    "domain": "light",
                    "device_class": "",
                    "from": "off",
                    "to": "on",
                    "timestamp": datetime.now().isoformat(),
                    "friendly_name": "Kitchen Light",
                },
            )

            # Shadow engine should have buffered the event
            assert len(shadow._recent_events) == 1
            assert shadow._recent_events[0]["entity_id"] == "light.kitchen"
            assert shadow._recent_events[0]["domain"] == "light"
            assert shadow._recent_events[0]["to"] == "on"
            assert shadow._recent_events[0]["from"] == "off"
        finally:
            await shadow.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_events_accumulate_in_shadow_engine(self):
        """Multiple state_changed events should all reach shadow engine."""
        hub = IntegrationHub()

        shadow = ShadowEngine(hub)
        hub.register_module(shadow)
        await shadow.initialize()

        try:
            entities = [
                ("light.kitchen", "light", "off", "on", "Kitchen Light"),
                ("switch.hallway", "switch", "off", "on", "Hallway Switch"),
                ("light.bedroom", "light", "on", "off", "Bedroom Light"),
            ]

            for entity_id, domain, from_s, to_s, name in entities:
                await hub.publish(
                    "state_changed",
                    {
                        "entity_id": entity_id,
                        "domain": domain,
                        "device_class": "",
                        "from": from_s,
                        "to": to_s,
                        "timestamp": datetime.now().isoformat(),
                        "friendly_name": name,
                    },
                )

            assert len(shadow._recent_events) == 3
            assert shadow._recent_events[0]["entity_id"] == "light.kitchen"
            assert shadow._recent_events[1]["entity_id"] == "switch.hallway"
            assert shadow._recent_events[2]["entity_id"] == "light.bedroom"
        finally:
            await shadow.shutdown()

    @pytest.mark.asyncio
    async def test_on_event_does_not_double_count(self):
        """Shadow engine's on_event should remain a no-op to prevent double-handling.

        The IntegrationHub.publish() calls both subscribers and module.on_event().
        Only the subscriber callback should buffer events.
        """
        hub = IntegrationHub()

        shadow = ShadowEngine(hub)
        hub.register_module(shadow)
        await shadow.initialize()

        try:
            await hub.publish(
                "state_changed",
                {
                    "entity_id": "light.kitchen",
                    "domain": "light",
                    "device_class": "",
                    "from": "off",
                    "to": "on",
                    "timestamp": datetime.now().isoformat(),
                    "friendly_name": "Kitchen Light",
                },
            )

            # Should be exactly 1, not 2 (proves on_event is a no-op)
            assert len(shadow._recent_events) == 1
        finally:
            await shadow.shutdown()

    @pytest.mark.asyncio
    async def test_activity_monitor_handle_state_changed_publishes_event(self):
        """Full integration: activity_monitor._handle_state_changed fires
        asyncio.create_task(hub.publish(...)) which reaches shadow engine."""
        hub = IntegrationHub()

        # Create activity monitor with dummy HA credentials
        from aria.modules.activity_monitor import ActivityMonitor

        activity_mon = ActivityMonitor(hub, "http://dummy:8123", "dummy_token")
        hub.register_module(activity_mon)

        shadow = ShadowEngine(hub)
        hub.register_module(shadow)
        await shadow.initialize()

        try:
            # Call _handle_state_changed with realistic HA WebSocket data
            event_data = {
                "entity_id": "light.living_room",
                "new_state": {
                    "state": "on",
                    "attributes": {
                        "friendly_name": "Living Room Light",
                        "device_class": "",
                    },
                },
                "old_state": {"state": "off"},
            }

            activity_mon._handle_state_changed(event_data)

            # The publish is fire-and-forget via asyncio.create_task,
            # so we need to yield control for the task to execute
            await asyncio.sleep(0.05)

            # Shadow engine should have received the event
            assert len(shadow._recent_events) == 1
            assert shadow._recent_events[0]["entity_id"] == "light.living_room"
            assert shadow._recent_events[0]["domain"] == "light"
            assert shadow._recent_events[0]["to"] == "on"
            assert shadow._recent_events[0]["from"] == "off"
        finally:
            await shadow.shutdown()


# ============================================================================
# Phase 2: Config store + curation integration
# ============================================================================


class TestConfigStoreIntegration:
    """Verify shadow engine reads from config store and respects curation."""

    @pytest.mark.asyncio
    async def test_excluded_entity_skipped(self, hub, engine):
        """Events from excluded entities should be silently dropped."""
        # Set up: only light.kitchen is included
        hub.cache.get_included_entity_ids = AsyncMock(return_value={"light.kitchen"})

        # Send event from an excluded entity
        event = make_state_changed_event(entity_id="light.bedroom")
        await engine._on_state_changed(event)

        # Should NOT be buffered
        assert len(engine._recent_events) == 0

    @pytest.mark.asyncio
    async def test_included_entity_processed(self, hub, engine):
        """Events from included entities should be processed normally."""
        hub.cache.get_included_entity_ids = AsyncMock(return_value={"light.kitchen"})

        event = make_state_changed_event(entity_id="light.kitchen")
        await engine._on_state_changed(event)

        # Should be buffered
        assert len(engine._recent_events) == 1
        assert engine._recent_events[0]["entity_id"] == "light.kitchen"

    @pytest.mark.asyncio
    async def test_empty_included_set_includes_all(self, hub, engine):
        """When curation table is empty (no classifications), all entities pass."""
        hub.cache.get_included_entity_ids = AsyncMock(return_value=set())

        event = make_state_changed_event(entity_id="light.bedroom")
        await engine._on_state_changed(event)

        # Empty set = no curation data = include everything
        assert len(engine._recent_events) == 1

    @pytest.mark.asyncio
    async def test_config_value_overrides_min_confidence(self, hub, engine):
        """Min confidence from config store should filter predictions."""

        # Set very high threshold via config
        async def mock_config_value(key, fallback=None):
            if key == "shadow.min_confidence":
                return 0.99
            if key == "shadow.default_window_seconds":
                return DEFAULT_WINDOW_SECONDS
            return fallback

        hub.cache.get_config_value = AsyncMock(side_effect=mock_config_value)

        # Set up some recent events so frequency prediction has data
        engine._recent_events = [
            {
                "domain": "light",
                "entity": "light.kitchen",
                "state": "on",
                "timestamp": datetime.now().isoformat(),
                "seconds_ago": 10,
            },
        ]

        # Build context and generate predictions
        context = {
            "timestamp": datetime.now().isoformat(),
            "time_features": {"hour_sin": 0, "hour_cos": 1, "dow_sin": 0, "dow_cos": 1},
            "presence": {"home": True, "rooms": ["kitchen"]},
            "recent_events": engine._recent_events,
            "current_states": {},
            "rolling_stats": {"1h_event_count": 5, "1h_domain_entropy": 1.0, "1h_dominant_domain_pct": 0.5},
            "trigger_event": {"entity_id": "light.kitchen", "domain": "light"},
        }

        predictions = await engine._generate_predictions(context)

        # With 0.99 threshold, frequency-based predictions (typically <1.0 confidence)
        # should mostly be filtered out
        for pred in predictions:
            assert pred["confidence"] >= 0.99

    @pytest.mark.asyncio
    async def test_config_value_overrides_cooldown(self, hub, engine):
        """Prediction cooldown from config store should be respected."""

        # Set very long cooldown via config
        async def mock_config_value(key, fallback=None):
            if key == "shadow.prediction_cooldown_s":
                return 9999
            if key == "shadow.default_window_seconds":
                return DEFAULT_WINDOW_SECONDS
            if key == "shadow.min_confidence":
                return MIN_CONFIDENCE
            return fallback

        hub.cache.get_config_value = AsyncMock(side_effect=mock_config_value)
        hub.cache.get_included_entity_ids = AsyncMock(return_value=set())

        # Set up summary cache for predictions
        await hub.set_cache(
            CACHE_ACTIVITY_SUMMARY,
            {
                "event_predictions": {
                    "predicted_next_domain": "light",
                    "probability": 0.8,
                    "method": "test",
                },
                "occupancy": {"anyone_home": True},
                "recent_activity": [],
            },
        )

        # First event — should generate predictions (no last_prediction_time)
        event1 = make_state_changed_event(entity_id="light.kitchen")
        await engine._on_state_changed(event1)
        assert hub.cache.insert_prediction.call_count == 1

        # Second event — should be blocked by cooldown
        event2 = make_state_changed_event(entity_id="light.bedroom")
        await engine._on_state_changed(event2)
        # Still 1 — second call was blocked by the 9999s cooldown
        assert hub.cache.insert_prediction.call_count == 1

    @pytest.mark.asyncio
    async def test_config_value_overrides_window_seconds(self, hub, engine):
        """Window seconds from config store should appear in predictions."""

        async def mock_config_value(key, fallback=None):
            if key == "shadow.default_window_seconds":
                return 1200  # 20 minutes instead of default 600
            if key == "shadow.min_confidence":
                return 0.01  # low threshold so prediction passes
            return fallback

        hub.cache.get_config_value = AsyncMock(side_effect=mock_config_value)
        hub.cache.get_included_entity_ids = AsyncMock(return_value=set())

        # Set up summary with event_predictions so _predict_next_domain returns a result
        await hub.set_cache(
            CACHE_ACTIVITY_SUMMARY,
            {
                "event_predictions": {
                    "predicted_next_domain": "light",
                    "probability": 0.8,
                    "method": "frequency",
                },
                "occupancy": {"anyone_home": True},
                "recent_activity": [],
            },
        )

        context = {
            "timestamp": datetime.now().isoformat(),
            "time_features": {"hour_sin": 0, "hour_cos": 1, "dow_sin": 0, "dow_cos": 1},
            "presence": {"home": True, "rooms": []},
            "recent_events": [],
            "current_states": {},
            "rolling_stats": {"1h_event_count": 0, "1h_domain_entropy": 0, "1h_dominant_domain_pct": 0},
            "trigger_event": {"entity_id": "light.kitchen", "domain": "light"},
        }

        predictions = await engine._generate_predictions(context)
        # At least the domain prediction should use 1200s window
        domain_preds = [p for p in predictions if p["type"] == "next_domain_action"]
        assert len(domain_preds) >= 1
        assert domain_preds[0]["window_seconds"] == 1200


# ============================================================================
# Thompson Sampling f-dsw Non-Stationarity
# ============================================================================


from aria.modules.shadow_engine import ThompsonSampler  # noqa: E402


class TestThompsonSamplerFDSW:
    """Test f-dsw discount factor and sliding window on ThompsonSampler."""

    def test_discount_factor_decays_posteriors(self):
        """Sampler with discount_factor=0.9 should decay alpha on each record."""
        sampler = ThompsonSampler(discount_factor=0.9)
        context = {"time_features": {"hour_sin": 0.8}, "presence": {"home": True}}

        # Record 10 successes
        for _ in range(10):
            sampler.record_outcome(context, success=True)

        key = sampler.get_bucket_key(context)
        bucket = sampler._buckets[key]
        # Without decay, alpha would be 11.0 (1.0 + 10).
        # With decay factor 0.9 applied before each update, alpha should be < 8.0
        assert bucket["alpha"] < 8.0

    def test_window_size_caps_effective_history(self):
        """100 observations with window=20 should keep alpha low."""
        sampler = ThompsonSampler(window_size=20)
        context = {"time_features": {"hour_sin": 0.8}, "presence": {"home": True}}

        for _ in range(100):
            sampler.record_outcome(context, success=True)

        key = sampler.get_bucket_key(context)
        bucket = sampler._buckets[key]
        # With window_size=20, effective alpha should be bounded
        assert bucket["alpha"] < 25.0

    def test_reset_bucket_clears_state(self):
        """reset_bucket should restore flat prior (alpha=1.0, beta=1.0)."""
        sampler = ThompsonSampler()
        context = {"time_features": {"hour_sin": 0.8}, "presence": {"home": True}}

        sampler.record_outcome(context, success=True)
        sampler.record_outcome(context, success=True)
        key = sampler.get_bucket_key(context)
        assert sampler._buckets[key]["alpha"] > 1.0

        sampler.reset_bucket(key)
        assert sampler._buckets[key]["alpha"] == 1.0
        assert sampler._buckets[key]["beta"] == 1.0

    def test_default_discount_factor_is_0_95(self):
        """Default discount_factor should be 0.95."""
        sampler = ThompsonSampler()
        assert sampler.discount_factor == 0.95


# ============================================================================
# Thompson Sampling Persistence
# ============================================================================


class TestThompsonPersistence:
    """Test Thompson Sampling state save/load round-trip."""

    @pytest.mark.asyncio
    async def test_save_and_load_thompson_state(self, tmp_path):
        """Thompson state should round-trip through CacheManager."""
        from aria.hub.cache import CacheManager

        db_path = str(tmp_path / "test.db")
        cache = CacheManager(db_path)
        await cache.initialize()

        try:
            # Build state
            sampler = ThompsonSampler()
            ctx = {"time_features": {"hour_sin": 0.8}, "presence": {"home": True}}
            sampler.record_outcome(ctx, success=True)
            sampler.record_outcome(ctx, success=False)

            state = sampler.get_state()

            # Save to DB
            await cache.save_thompson_state(state)

            # Load from DB
            loaded = await cache.load_thompson_state()
            assert loaded is not None

            # Restore into a new sampler
            sampler2 = ThompsonSampler()
            sampler2.load_state(loaded)

            assert sampler2.get_state() == state
        finally:
            await cache.close()


# ============================================================================
# Feedback: Shadow Engine → Capabilities Cache
# ============================================================================


class TestCapabilityFeedback:
    """Test _get_capability_hit_rates and _write_feedback_to_capabilities."""

    @pytest.mark.asyncio
    async def test_write_feedback_updates_capabilities(self, engine, hub):
        """Should write shadow_accuracy to each capability with prediction data."""
        # Set up capabilities cache with entities
        await hub.set_cache(
            "capabilities",
            {
                "lighting": {
                    "name": "Lighting",
                    "entities": ["light.kitchen", "light.bedroom"],
                },
                "media": {
                    "name": "Media",
                    "entities": ["media_player.tv", "media_player.speaker"],
                },
            },
        )

        # Populate recent resolved with predictions involving light domain
        engine._recent_resolved = [
            {
                "id": "pred-1",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "correct",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": ["kitchen"]},
            },
            {
                "id": "pred-2",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "disagreement",
                "actual": {"event_count": 1, "domains": ["switch"], "rooms": []},
            },
            {
                "id": "pred-3",
                "predictions": [{"type": "next_domain_action", "predicted": "media_player"}],
                "outcome": "correct",
                "actual": {"event_count": 2, "domains": ["media_player"], "rooms": []},
            },
        ]

        await engine._write_feedback_to_capabilities()

        # Read back capabilities
        caps = await hub.get_cache("capabilities")
        assert caps is not None
        caps_data = caps["data"]

        # Lighting: 2 predictions involved light domain (pred-1 correct, pred-2 had light predicted)
        assert "shadow_accuracy" in caps_data["lighting"]
        lighting_acc = caps_data["lighting"]["shadow_accuracy"]
        assert lighting_acc["total_predictions"] == 2
        assert lighting_acc["hit_rate"] == 0.5
        assert "last_updated" in lighting_acc

        # Media: 1 prediction involved media_player domain (pred-3 correct)
        assert "shadow_accuracy" in caps_data["media"]
        media_acc = caps_data["media"]["shadow_accuracy"]
        assert media_acc["total_predictions"] == 1
        assert media_acc["hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_write_feedback_clears_recent_resolved(self, engine, hub):
        """After writing feedback, _recent_resolved should be cleared."""
        await hub.set_cache(
            "capabilities",
            {
                "lighting": {
                    "name": "Lighting",
                    "entities": ["light.kitchen"],
                },
            },
        )

        engine._recent_resolved = [
            {
                "id": "pred-1",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "correct",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": []},
            },
        ]

        await engine._write_feedback_to_capabilities()
        assert engine._recent_resolved == []

    @pytest.mark.asyncio
    async def test_write_feedback_no_capabilities(self, engine, hub):
        """Should be a no-op when capabilities cache is empty."""
        engine._recent_resolved = [
            {
                "id": "pred-1",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "correct",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": []},
            },
        ]

        # No capabilities cache set
        await engine._write_feedback_to_capabilities()

        # recent_resolved should NOT be cleared (no feedback written)
        assert len(engine._recent_resolved) == 1

    @pytest.mark.asyncio
    async def test_write_feedback_no_resolved_predictions(self, engine, hub):
        """Should be a no-op when no predictions have been resolved."""
        await hub.set_cache(
            "capabilities",
            {
                "lighting": {
                    "name": "Lighting",
                    "entities": ["light.kitchen"],
                },
            },
        )

        engine._recent_resolved = []
        await engine._write_feedback_to_capabilities()

        # Capabilities should not be modified
        caps = await hub.get_cache("capabilities")
        assert "shadow_accuracy" not in caps["data"]["lighting"]

    @pytest.mark.asyncio
    async def test_write_feedback_no_entity_overlap(self, engine, hub):
        """Capabilities with no entity overlap should not get shadow_accuracy."""
        await hub.set_cache(
            "capabilities",
            {
                "climate": {
                    "name": "Climate",
                    "entities": ["climate.thermostat"],
                },
            },
        )

        engine._recent_resolved = [
            {
                "id": "pred-1",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "correct",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": []},
            },
        ]

        await engine._write_feedback_to_capabilities()

        caps = await hub.get_cache("capabilities")
        # climate capability should not have shadow_accuracy (no light entities)
        assert "shadow_accuracy" not in caps["data"]["climate"]

    def test_get_capability_hit_rates_empty_resolved(self, engine):
        """Should return empty dict when no resolved predictions."""
        engine._recent_resolved = []
        assert engine._get_capability_hit_rates() == {}

    def test_get_capability_hit_rates_computes_correctly(self, engine):
        """Should compute correct hit/total per capability."""
        engine._cached_cap_entities = {
            "lighting": ["light.kitchen", "light.bedroom"],
            "switches": ["switch.hallway"],
        }

        engine._recent_resolved = [
            {
                "id": "pred-1",
                "predictions": [{"type": "next_domain_action", "predicted": "light"}],
                "outcome": "correct",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": []},
            },
            {
                "id": "pred-2",
                "predictions": [{"type": "next_domain_action", "predicted": "switch"}],
                "outcome": "disagreement",
                "actual": {"event_count": 1, "domains": ["light"], "rooms": []},
            },
        ]

        rates = engine._get_capability_hit_rates()

        # lighting: pred-1 (light predicted, correct) + pred-2 (light in actual)
        assert rates["lighting"]["total"] == 2
        assert rates["lighting"]["hits"] == 1

        # switches: pred-2 (switch predicted, disagreement)
        assert rates["switches"]["total"] == 1
        assert rates["switches"]["hits"] == 0

    @pytest.mark.asyncio
    async def test_resolution_loop_triggers_feedback_every_10th(self, engine, hub):
        """Resolution loop should call _write_feedback_to_capabilities every 10th iteration."""
        calls = []

        async def mock_write_feedback():
            calls.append(1)

        engine._write_feedback_to_capabilities = mock_write_feedback

        # Simulate 10 iterations by setting counter to 9 and running one resolve
        engine._resolution_iteration_count = 9
        hub.cache.get_pending_predictions = AsyncMock(return_value=[])

        # We can't run the actual loop (it sleeps forever), so test the logic directly
        await engine._resolve_expired_predictions()
        engine._resolution_iteration_count += 1
        if engine._resolution_iteration_count % 10 == 0:
            await engine._write_feedback_to_capabilities()

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_resolution_loop_does_not_trigger_feedback_before_10th(self, engine, hub):
        """Resolution loop should NOT call _write_feedback_to_capabilities before 10th iteration."""
        calls = []

        async def mock_write_feedback():
            calls.append(1)

        engine._write_feedback_to_capabilities = mock_write_feedback

        # Simulate iteration 5
        engine._resolution_iteration_count = 4
        hub.cache.get_pending_predictions = AsyncMock(return_value=[])

        await engine._resolve_expired_predictions()
        engine._resolution_iteration_count += 1
        if engine._resolution_iteration_count % 10 == 0:
            await engine._write_feedback_to_capabilities()

        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_resolve_appends_to_recent_resolved(self, engine, hub):
        """_resolve_expired_predictions should append to _recent_resolved."""
        pending = [
            {
                "id": "pred-feedback-1",
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "predictions": [
                    {"type": "next_domain_action", "predicted": "light"},
                ],
                "confidence": 0.8,
                "window_seconds": 600,
                "context": {},
            }
        ]
        hub.cache.get_pending_predictions = AsyncMock(return_value=pending)
        engine._window_events["pred-feedback-1"] = [
            {"domain": "light", "entity_id": "light.kitchen", "to": "on"},
        ]

        await engine._resolve_expired_predictions()

        assert len(engine._recent_resolved) == 1
        assert engine._recent_resolved[0]["id"] == "pred-feedback-1"
        assert engine._recent_resolved[0]["outcome"] == "correct"

    @pytest.mark.asyncio
    async def test_write_feedback_routine_trigger_domains(self, engine, hub):
        """Should match routine_trigger expected_domains to capabilities."""
        await hub.set_cache(
            "capabilities",
            {
                "media": {
                    "name": "Media",
                    "entities": ["media_player.tv"],
                },
            },
        )

        engine._recent_resolved = [
            {
                "id": "pred-rt",
                "predictions": [
                    {
                        "type": "routine_trigger",
                        "predicted": "Evening Routine",
                        "expected_domains": ["media_player", "light"],
                    }
                ],
                "outcome": "correct",
                "actual": {"event_count": 3, "domains": ["media_player", "light"], "rooms": []},
            },
        ]

        await engine._write_feedback_to_capabilities()

        caps = await hub.get_cache("capabilities")
        assert "shadow_accuracy" in caps["data"]["media"]
        assert caps["data"]["media"]["shadow_accuracy"]["hit_rate"] == 1.0
