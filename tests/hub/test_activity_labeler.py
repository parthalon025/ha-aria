"""Unit tests for ActivityLabeler module.

Tests LLM prediction, classifier fallback, label recording, stats tracking,
classifier training at threshold, feature extraction, and time-of-day logic.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.modules.activity_labeler import (
    ActivityLabeler,
    CLASSIFIER_THRESHOLD,
)


# ============================================================================
# Mock Hub
# ============================================================================


class MockHub:
    """Lightweight hub mock for activity labeler tests."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._scheduled: List[dict] = []
        self._running = True

        self.cache = Mock()
        self.logger = Mock()
        self.modules = {}

    async def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    async def set_cache(self, key: str, data: dict, metadata: Optional[dict] = None) -> int:
        self._cache[key] = {"data": data, "version": 1, "last_updated": "now"}
        return 1

    async def schedule_task(self, **kwargs):
        self._scheduled.append(kwargs)

    def register_module(self, mod):
        self.modules[mod.module_id] = mod


# ============================================================================
# Helpers
# ============================================================================


def make_context(
    power_watts: float = 200.0,
    lights_on: int = 3,
    motion_rooms: str = "kitchen,living_room",
    hour: int = 14,
    occupancy: str = "home",
    recent_events: str = "light.kitchen turned on",
) -> dict:
    """Build a sensor context dict for testing."""
    return {
        "power_watts": power_watts,
        "lights_on": lights_on,
        "motion_rooms": motion_rooms,
        "hour": hour,
        "occupancy": occupancy,
        "recent_events": recent_events,
    }


def make_label(
    predicted: str = "cooking",
    actual: str = "cleaning",
    source: str = "corrected",
    context: Optional[dict] = None,
) -> dict:
    """Build a label dict matching the cache format."""
    return {
        "id": "abc123",
        "timestamp": datetime.now().isoformat(),
        "sensor_context": context or make_context(),
        "predicted_activity": predicted,
        "actual_activity": actual,
        "source": source,
    }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hub():
    return MockHub()


@pytest.fixture
def labeler(hub):
    module = ActivityLabeler(hub)
    return module


# ============================================================================
# Tests
# ============================================================================


class TestPredictActivity:
    """Tests for predict_activity method."""

    @pytest.mark.asyncio
    async def test_predict_activity_uses_ollama(self, labeler):
        """When no classifier is ready, predict_activity uses Ollama."""
        ollama_result = {"activity": "cooking", "confidence": 0.85}
        labeler._query_ollama = AsyncMock(return_value=ollama_result)

        context = make_context(power_watts=450, lights_on=3)
        result = await labeler.predict_activity(context)

        assert result["predicted"] == "cooking"
        assert result["confidence"] == 0.85
        assert result["method"] == "ollama"
        assert result["sensor_context"] == context
        assert "predicted_at" in result
        labeler._query_ollama.assert_awaited_once_with(context)

    @pytest.mark.asyncio
    async def test_predict_activity_uses_classifier_when_ready(self, labeler):
        """When classifier is ready, predict_activity uses it instead of Ollama."""
        # Set up mock classifier
        mock_clf = Mock()
        mock_clf.predict.return_value = [0]
        mock_clf.predict_proba.return_value = [[0.9, 0.1]]

        mock_encoder = Mock()
        mock_encoder.inverse_transform.return_value = ["cooking"]

        labeler._classifier = mock_clf
        labeler._classifier_ready = True
        labeler._label_encoder = mock_encoder

        context = make_context(power_watts=450, lights_on=3)
        result = await labeler.predict_activity(context)

        assert result["predicted"] == "cooking"
        assert result["confidence"] == 0.9
        assert result["method"] == "classifier"
        assert result["sensor_context"] == context
        mock_clf.predict.assert_called_once()
        mock_clf.predict_proba.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_falls_back_to_ollama_on_classifier_error(self, labeler):
        """If classifier raises, falls back to Ollama gracefully."""
        mock_clf = Mock()
        mock_clf.predict.side_effect = RuntimeError("bad features")
        labeler._classifier = mock_clf
        labeler._classifier_ready = True
        labeler._label_encoder = Mock()

        ollama_result = {"activity": "relaxing", "confidence": 0.6}
        labeler._query_ollama = AsyncMock(return_value=ollama_result)

        context = make_context()
        result = await labeler.predict_activity(context)

        assert result["method"] == "ollama"
        assert result["predicted"] == "relaxing"


class TestRecordLabel:
    """Tests for record_label method."""

    @pytest.mark.asyncio
    async def test_record_label_stores_correction(self, labeler, hub):
        """Recording a correction stores the label and updates stats."""
        context = make_context()
        stats = await labeler.record_label(
            predicted="cooking",
            actual="cleaning",
            sensor_context=context,
            source="corrected",
        )

        assert stats["total_labels"] == 1
        assert stats["total_corrections"] == 1
        assert stats["accuracy"] == 0.0  # predicted != actual
        assert "cooking" in stats["activities_seen"]
        assert "cleaning" in stats["activities_seen"]

        # Verify cache was written
        cache_entry = hub._cache.get("activity_labels")
        assert cache_entry is not None
        data = cache_entry["data"]
        assert len(data["labels"]) == 1
        assert data["labels"][0]["source"] == "corrected"
        assert data["labels"][0]["predicted_activity"] == "cooking"
        assert data["labels"][0]["actual_activity"] == "cleaning"

    @pytest.mark.asyncio
    async def test_record_label_confirmed(self, labeler, hub):
        """Recording a confirmed label (predicted == actual) has accuracy=1.0."""
        context = make_context()
        stats = await labeler.record_label(
            predicted="cooking",
            actual="cooking",
            sensor_context=context,
            source="confirmed",
        )

        assert stats["total_labels"] == 1
        assert stats["total_corrections"] == 0  # confirmed, not corrected
        assert stats["accuracy"] == 1.0  # predicted == actual

        cache_entry = hub._cache.get("activity_labels")
        data = cache_entry["data"]
        assert data["labels"][0]["source"] == "confirmed"

    @pytest.mark.asyncio
    async def test_record_label_accumulates(self, labeler):
        """Multiple labels accumulate correctly in stats."""
        ctx = make_context()
        await labeler.record_label("cooking", "cooking", ctx, "confirmed")
        await labeler.record_label("cooking", "cleaning", ctx, "corrected")
        stats = await labeler.record_label("relaxing", "relaxing", ctx, "confirmed")

        assert stats["total_labels"] == 3
        assert stats["total_corrections"] == 1
        assert stats["accuracy"] == round(2 / 3, 3)  # 2 correct out of 3


class TestClassifierTraining:
    """Tests for classifier training at threshold."""

    @pytest.mark.asyncio
    async def test_classifier_trains_at_threshold(self, labeler, hub):
        """After recording CLASSIFIER_THRESHOLD labels, classifier becomes ready."""
        ctx = make_context()

        # Record labels up to threshold - 1
        for i in range(CLASSIFIER_THRESHOLD - 1):
            activity = "cooking" if i % 2 == 0 else "sleeping"
            # Vary context to give classifier something to learn
            varied_ctx = make_context(
                power_watts=400.0 if activity == "cooking" else 50.0,
                lights_on=3 if activity == "cooking" else 0,
                hour=18 if activity == "cooking" else 23,
                occupancy="home",
            )
            await labeler.record_label(activity, activity, varied_ctx, "confirmed")

        # Classifier should not be ready yet
        assert not labeler._classifier_ready

        # One more label crosses the threshold
        stats = await labeler.record_label("cooking", "cooking", ctx, "confirmed")

        assert labeler._classifier_ready
        assert labeler._classifier is not None
        assert labeler._label_encoder is not None
        assert stats["classifier_ready"] is True
        assert stats["last_trained"] is not None

    @pytest.mark.asyncio
    async def test_classifier_not_trained_below_threshold(self, labeler):
        """Below threshold, classifier stays inactive."""
        ctx = make_context()
        for i in range(5):
            activity = "cooking" if i % 2 == 0 else "sleeping"
            await labeler.record_label(activity, activity, ctx, "confirmed")

        assert not labeler._classifier_ready
        assert labeler._classifier is None


class TestContextToFeatures:
    """Tests for _context_to_features method."""

    def test_context_to_features(self, labeler):
        """Verify feature vector shape and values."""
        ctx = make_context(
            power_watts=450.0,
            lights_on=3,
            motion_rooms="kitchen,living_room",
            hour=14,
            occupancy="home",
        )
        features = labeler._context_to_features(ctx)

        assert len(features) == 5
        assert features[0] == 450.0  # power_watts
        assert features[1] == 3.0  # lights_on
        assert features[2] == 2.0  # motion_room_count (2 rooms)
        assert features[3] == 14.0  # hour
        assert features[4] == 1.0  # is_home

    def test_context_to_features_away(self, labeler):
        """Verify is_home=0 when occupancy is not home."""
        ctx = make_context(occupancy="away")
        features = labeler._context_to_features(ctx)
        assert features[4] == 0.0

    def test_context_to_features_empty_motion(self, labeler):
        """Verify motion_room_count is 0 for no motion."""
        ctx = make_context(motion_rooms="none")
        features = labeler._context_to_features(ctx)
        assert features[2] == 0.0

    def test_context_to_features_list_motion(self, labeler):
        """Verify motion_rooms as list is handled correctly."""
        ctx = make_context()
        ctx["motion_rooms"] = ["kitchen", "bedroom", "bathroom"]
        features = labeler._context_to_features(ctx)
        assert features[2] == 3.0

    def test_context_to_features_defaults(self, labeler):
        """Missing context keys produce sensible defaults."""
        features = labeler._context_to_features({})
        assert len(features) == 5
        assert features[0] == 0.0  # power_watts default
        assert features[1] == 0.0  # lights_on default
        assert features[2] == 0.0  # motion_room_count default
        assert features[4] == 0.0  # is_home default (unknown occupancy)


class TestTimeOfDay:
    """Tests for _time_of_day static method."""

    def test_time_of_day_night(self):
        """Hours 0-5 are night."""
        for hour in [0, 3, 5]:
            with patch("aria.modules.activity_labeler.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 2, 15, hour, 0)
                result = ActivityLabeler._time_of_day()
                assert result == "night", f"Hour {hour} should be night, got {result}"

    def test_time_of_day_morning(self):
        """Hours 6-11 are morning."""
        for hour in [6, 9, 11]:
            with patch("aria.modules.activity_labeler.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 2, 15, hour, 0)
                result = ActivityLabeler._time_of_day()
                assert result == "morning", f"Hour {hour} should be morning, got {result}"

    def test_time_of_day_afternoon(self):
        """Hours 12-17 are afternoon."""
        for hour in [12, 14, 17]:
            with patch("aria.modules.activity_labeler.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 2, 15, hour, 0)
                result = ActivityLabeler._time_of_day()
                assert result == "afternoon", f"Hour {hour} should be afternoon, got {result}"

    def test_time_of_day_evening(self):
        """Hours 18-23 are evening."""
        for hour in [18, 20, 23]:
            with patch("aria.modules.activity_labeler.datetime") as mock_dt:
                mock_dt.now.return_value = datetime(2026, 2, 15, hour, 0)
                result = ActivityLabeler._time_of_day()
                assert result == "evening", f"Hour {hour} should be evening, got {result}"


class TestInitialize:
    """Tests for module initialization."""

    @pytest.mark.asyncio
    async def test_initialize_schedules_prediction(self, labeler, hub):
        """Initialize schedules periodic prediction task."""
        await labeler.initialize()
        assert len(hub._scheduled) == 1
        assert hub._scheduled[0]["task_id"] == "activity_labeler_predict"
        assert hub._scheduled[0]["run_immediately"] is False

    @pytest.mark.asyncio
    async def test_initialize_restores_classifier(self, labeler, hub):
        """Initialize restores classifier when cached labels exist at threshold."""
        # Pre-populate cache with enough labels
        labels = []
        for i in range(CLASSIFIER_THRESHOLD):
            activity = "cooking" if i % 2 == 0 else "sleeping"
            labels.append(make_label(
                predicted=activity,
                actual=activity,
                source="confirmed",
                context=make_context(
                    power_watts=400.0 if activity == "cooking" else 50.0,
                    lights_on=3 if activity == "cooking" else 0,
                    hour=18 if activity == "cooking" else 23,
                ),
            ))

        hub._cache["activity_labels"] = {
            "data": {
                "current_activity": None,
                "labels": labels,
                "label_stats": {
                    "total_labels": CLASSIFIER_THRESHOLD,
                    "total_corrections": 0,
                    "accuracy": 1.0,
                    "activities_seen": ["cooking", "sleeping"],
                    "classifier_ready": True,
                    "last_trained": datetime.now().isoformat(),
                },
            },
        }

        await labeler.initialize()
        assert labeler._classifier_ready
        assert labeler._classifier is not None
