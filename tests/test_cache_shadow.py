"""Tests for shadow engine cache tables: predictions and pipeline_state."""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hub.cache import CacheManager
from hub.constants import CACHE_PREDICTIONS, CACHE_PIPELINE_STATE


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def cache(tmp_path):
    """Create and initialize a CacheManager with a temp DB."""
    db_path = str(tmp_path / "test_hub.db")
    cm = CacheManager(db_path)
    await cm.initialize()
    yield cm
    await cm.close()


def _make_prediction_kwargs(
    prediction_id="pred-001",
    timestamp=None,
    context=None,
    predictions=None,
    confidence=0.85,
    window_seconds=300,
    is_exploration=False,
):
    """Helper to build insert_prediction keyword args."""
    return {
        "prediction_id": prediction_id,
        "timestamp": timestamp or datetime.now().isoformat(),
        "context": context or {"room": "living_room", "time_of_day": "evening"},
        "predictions": predictions or [{"action": "light.turn_on", "target": "living_room"}],
        "confidence": confidence,
        "window_seconds": window_seconds,
        "is_exploration": is_exploration,
    }


# ============================================================================
# Table creation
# ============================================================================


class TestTableCreation:
    """Verify tables exist after initialize()."""

    @pytest.mark.asyncio
    async def test_predictions_table_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_pipeline_state_table_exists(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_state'"
        )
        row = await cursor.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_predictions_indexes_exist(self, cache):
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_predictions_%'"
        )
        rows = await cursor.fetchall()
        names = {row["name"] for row in rows}
        assert "idx_predictions_timestamp" in names
        assert "idx_predictions_outcome" in names

    @pytest.mark.asyncio
    async def test_reinitialize_is_safe(self, cache):
        """Calling initialize() again should not fail (IF NOT EXISTS)."""
        await cache.initialize()
        cursor = await cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        assert await cursor.fetchone() is not None


# ============================================================================
# Constants
# ============================================================================


class TestConstants:

    def test_predictions_constant(self):
        assert CACHE_PREDICTIONS == "predictions"

    def test_pipeline_state_constant(self):
        assert CACHE_PIPELINE_STATE == "pipeline_state"


# ============================================================================
# insert_prediction
# ============================================================================


class TestInsertPrediction:

    @pytest.mark.asyncio
    async def test_insert_basic(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        rows = await cache.get_recent_predictions(limit=10)
        assert len(rows) == 1
        assert rows[0]["id"] == "pred-001"
        assert rows[0]["confidence"] == 0.85
        assert rows[0]["window_seconds"] == 300
        assert rows[0]["is_exploration"] is False
        assert rows[0]["outcome"] is None
        assert rows[0]["actual"] is None
        assert rows[0]["resolved_at"] is None

    @pytest.mark.asyncio
    async def test_insert_context_is_json(self, cache):
        ctx = {"room": "kitchen", "occupancy": True, "devices": ["sensor.motion"]}
        kwargs = _make_prediction_kwargs(context=ctx)
        await cache.insert_prediction(**kwargs)

        rows = await cache.get_recent_predictions()
        assert rows[0]["context"] == ctx

    @pytest.mark.asyncio
    async def test_insert_predictions_array_is_json(self, cache):
        preds = [
            {"action": "light.turn_on", "target": "hall"},
            {"action": "climate.set_temp", "target": "bedroom", "value": 21},
        ]
        kwargs = _make_prediction_kwargs(predictions=preds)
        await cache.insert_prediction(**kwargs)

        rows = await cache.get_recent_predictions()
        assert rows[0]["predictions"] == preds

    @pytest.mark.asyncio
    async def test_insert_exploration_flag(self, cache):
        kwargs = _make_prediction_kwargs(is_exploration=True)
        await cache.insert_prediction(**kwargs)

        rows = await cache.get_recent_predictions()
        assert rows[0]["is_exploration"] is True

    @pytest.mark.asyncio
    async def test_insert_duplicate_id_fails(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        with pytest.raises(Exception):
            await cache.insert_prediction(**kwargs)

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self, tmp_path):
        cm = CacheManager(str(tmp_path / "uninit.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await cm.insert_prediction(**_make_prediction_kwargs())


# ============================================================================
# update_prediction_outcome
# ============================================================================


class TestUpdatePredictionOutcome:

    @pytest.mark.asyncio
    async def test_update_correct(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        await cache.update_prediction_outcome("pred-001", "correct")

        rows = await cache.get_recent_predictions()
        assert rows[0]["outcome"] == "correct"
        assert rows[0]["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_update_with_actual(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        actual = {"event": "light.turn_on", "entity": "light.living_room"}
        await cache.update_prediction_outcome("pred-001", "correct", actual=actual)

        rows = await cache.get_recent_predictions()
        assert rows[0]["actual"] == actual

    @pytest.mark.asyncio
    async def test_update_disagreement(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        await cache.update_prediction_outcome(
            "pred-001", "disagreement", actual={"event": "light.turn_off"}, propagated_count=2
        )

        rows = await cache.get_recent_predictions()
        assert rows[0]["outcome"] == "disagreement"
        assert rows[0]["propagated_count"] == 2

    @pytest.mark.asyncio
    async def test_update_nothing(self, cache):
        kwargs = _make_prediction_kwargs()
        await cache.insert_prediction(**kwargs)

        await cache.update_prediction_outcome("pred-001", "nothing")

        rows = await cache.get_recent_predictions()
        assert rows[0]["outcome"] == "nothing"
        assert rows[0]["actual"] is None

    @pytest.mark.asyncio
    async def test_update_nonexistent_is_noop(self, cache):
        """Updating a non-existent prediction doesn't raise — just 0 rows affected."""
        await cache.update_prediction_outcome("does-not-exist", "correct")
        rows = await cache.get_recent_predictions()
        assert len(rows) == 0


# ============================================================================
# get_recent_predictions
# ============================================================================


class TestGetRecentPredictions:

    @pytest.mark.asyncio
    async def test_empty(self, cache):
        rows = await cache.get_recent_predictions()
        assert rows == []

    @pytest.mark.asyncio
    async def test_ordered_by_timestamp_desc(self, cache):
        for i in range(3):
            ts = (datetime.now() - timedelta(minutes=10 - i)).isoformat()
            await cache.insert_prediction(
                **_make_prediction_kwargs(prediction_id=f"pred-{i}", timestamp=ts)
            )

        rows = await cache.get_recent_predictions()
        assert len(rows) == 3
        # Most recent first
        assert rows[0]["id"] == "pred-2"
        assert rows[2]["id"] == "pred-0"

    @pytest.mark.asyncio
    async def test_limit(self, cache):
        for i in range(5):
            await cache.insert_prediction(
                **_make_prediction_kwargs(prediction_id=f"pred-{i}")
            )

        rows = await cache.get_recent_predictions(limit=2)
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_outcome_filter(self, cache):
        await cache.insert_prediction(**_make_prediction_kwargs(prediction_id="p1"))
        await cache.insert_prediction(**_make_prediction_kwargs(prediction_id="p2"))
        await cache.insert_prediction(**_make_prediction_kwargs(prediction_id="p3"))

        await cache.update_prediction_outcome("p1", "correct")
        await cache.update_prediction_outcome("p2", "disagreement")
        # p3 remains pending (NULL outcome)

        rows = await cache.get_recent_predictions(outcome_filter="correct")
        assert len(rows) == 1
        assert rows[0]["id"] == "p1"

        rows = await cache.get_recent_predictions(outcome_filter="disagreement")
        assert len(rows) == 1
        assert rows[0]["id"] == "p2"


# ============================================================================
# get_pending_predictions
# ============================================================================


class TestGetPendingPredictions:

    @pytest.mark.asyncio
    async def test_empty(self, cache):
        rows = await cache.get_pending_predictions()
        assert rows == []

    @pytest.mark.asyncio
    async def test_returns_expired_window(self, cache):
        """Prediction with expired window should be returned."""
        past = (datetime.now() - timedelta(minutes=10)).isoformat()
        await cache.insert_prediction(
            **_make_prediction_kwargs(
                prediction_id="p-expired",
                timestamp=past,
                window_seconds=60,  # 1 min window, made 10 min ago
            )
        )

        rows = await cache.get_pending_predictions()
        assert len(rows) == 1
        assert rows[0]["id"] == "p-expired"

    @pytest.mark.asyncio
    async def test_excludes_future_window(self, cache):
        """Prediction whose window hasn't expired should NOT be returned."""
        now = datetime.now().isoformat()
        await cache.insert_prediction(
            **_make_prediction_kwargs(
                prediction_id="p-future",
                timestamp=now,
                window_seconds=3600,  # 1 hour window, just made
            )
        )

        rows = await cache.get_pending_predictions()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_excludes_resolved(self, cache):
        """Already resolved predictions should not appear."""
        past = (datetime.now() - timedelta(minutes=10)).isoformat()
        await cache.insert_prediction(
            **_make_prediction_kwargs(
                prediction_id="p-resolved",
                timestamp=past,
                window_seconds=60,
            )
        )
        await cache.update_prediction_outcome("p-resolved", "correct")

        rows = await cache.get_pending_predictions()
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_before_timestamp_filter(self, cache):
        """Use explicit before_timestamp to control cutoff."""
        ts = "2025-01-01T12:00:00"
        await cache.insert_prediction(
            **_make_prediction_kwargs(
                prediction_id="p1",
                timestamp=ts,
                window_seconds=60,
            )
        )

        # Check at a time after the window expired
        check_time = "2025-01-01T12:05:00"
        rows = await cache.get_pending_predictions(before_timestamp=check_time)
        assert len(rows) == 1

        # Check at a time before the window expired
        check_time_early = "2025-01-01T12:00:30"
        rows = await cache.get_pending_predictions(before_timestamp=check_time_early)
        assert len(rows) == 0

    @pytest.mark.asyncio
    async def test_ordered_by_timestamp_asc(self, cache):
        """Pending predictions should be oldest first."""
        for i in range(3):
            ts = (datetime.now() - timedelta(minutes=30 - i)).isoformat()
            await cache.insert_prediction(
                **_make_prediction_kwargs(
                    prediction_id=f"p-{i}",
                    timestamp=ts,
                    window_seconds=60,
                )
            )

        rows = await cache.get_pending_predictions()
        assert len(rows) == 3
        assert rows[0]["id"] == "p-0"  # oldest first
        assert rows[2]["id"] == "p-2"


# ============================================================================
# get_accuracy_stats
# ============================================================================


class TestGetAccuracyStats:

    @pytest.mark.asyncio
    async def test_empty(self, cache):
        stats = await cache.get_accuracy_stats()
        assert stats["overall_accuracy"] == 0.0
        assert stats["total_resolved"] == 0
        assert stats["per_outcome"] == {}
        assert stats["daily_trend"] == []

    @pytest.mark.asyncio
    async def test_all_correct(self, cache):
        for i in range(5):
            await cache.insert_prediction(
                **_make_prediction_kwargs(prediction_id=f"p-{i}")
            )
            await cache.update_prediction_outcome(f"p-{i}", "correct")

        stats = await cache.get_accuracy_stats()
        assert stats["overall_accuracy"] == 1.0
        assert stats["total_resolved"] == 5
        assert stats["per_outcome"]["correct"] == 5

    @pytest.mark.asyncio
    async def test_mixed_outcomes(self, cache):
        outcomes = ["correct", "correct", "disagreement", "nothing", "correct"]
        for i, outcome in enumerate(outcomes):
            await cache.insert_prediction(
                **_make_prediction_kwargs(prediction_id=f"p-{i}")
            )
            await cache.update_prediction_outcome(f"p-{i}", outcome)

        stats = await cache.get_accuracy_stats()
        assert stats["total_resolved"] == 5
        assert stats["per_outcome"]["correct"] == 3
        assert stats["per_outcome"]["disagreement"] == 1
        assert stats["per_outcome"]["nothing"] == 1
        assert stats["overall_accuracy"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_respects_days_window(self, cache):
        """Only predictions within the time window should count."""
        # Old prediction (outside 7-day window)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        await cache.insert_prediction(
            **_make_prediction_kwargs(prediction_id="p-old", timestamp=old_ts)
        )
        await cache.update_prediction_outcome("p-old", "correct")

        # Recent prediction
        await cache.insert_prediction(
            **_make_prediction_kwargs(prediction_id="p-new")
        )
        await cache.update_prediction_outcome("p-new", "disagreement")

        stats = await cache.get_accuracy_stats(days=7)
        assert stats["total_resolved"] == 1
        assert stats["per_outcome"].get("correct", 0) == 0
        assert stats["per_outcome"]["disagreement"] == 1

    @pytest.mark.asyncio
    async def test_daily_trend(self, cache):
        """Daily trend should group by resolved_at date."""
        for i in range(3):
            await cache.insert_prediction(
                **_make_prediction_kwargs(prediction_id=f"p-{i}")
            )
            await cache.update_prediction_outcome(f"p-{i}", "correct" if i < 2 else "nothing")

        stats = await cache.get_accuracy_stats()
        assert len(stats["daily_trend"]) >= 1
        today_entry = stats["daily_trend"][-1]
        assert today_entry["total"] == 3
        assert today_entry["correct"] == 2
        assert today_entry["accuracy"] == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_excludes_pending_predictions(self, cache):
        """Predictions without resolved_at should not count in stats."""
        await cache.insert_prediction(**_make_prediction_kwargs(prediction_id="p-pending"))
        await cache.insert_prediction(**_make_prediction_kwargs(prediction_id="p-resolved"))
        await cache.update_prediction_outcome("p-resolved", "correct")

        stats = await cache.get_accuracy_stats()
        assert stats["total_resolved"] == 1


# ============================================================================
# get_pipeline_state
# ============================================================================


class TestGetPipelineState:

    @pytest.mark.asyncio
    async def test_creates_default_on_first_call(self, cache):
        state = await cache.get_pipeline_state()
        assert state["id"] == 1
        assert state["current_stage"] == "backtest"
        assert state["stage_entered_at"] is not None
        assert state["updated_at"] is not None
        assert state["backtest_accuracy"] is None
        assert state["shadow_accuracy_7d"] is None
        assert state["suggest_approval_rate_14d"] is None
        assert state["autonomous_contexts"] is None

    @pytest.mark.asyncio
    async def test_returns_same_on_second_call(self, cache):
        state1 = await cache.get_pipeline_state()
        state2 = await cache.get_pipeline_state()
        assert state1["stage_entered_at"] == state2["stage_entered_at"]

    @pytest.mark.asyncio
    async def test_not_initialized_raises(self, tmp_path):
        cm = CacheManager(str(tmp_path / "uninit.db"))
        with pytest.raises(RuntimeError, match="not initialized"):
            await cm.get_pipeline_state()


# ============================================================================
# update_pipeline_state
# ============================================================================


class TestUpdatePipelineState:

    @pytest.mark.asyncio
    async def test_update_stage(self, cache):
        await cache.update_pipeline_state(current_stage="shadow")
        state = await cache.get_pipeline_state()
        assert state["current_stage"] == "shadow"

    @pytest.mark.asyncio
    async def test_update_accuracy_fields(self, cache):
        await cache.update_pipeline_state(
            backtest_accuracy=0.92,
            shadow_accuracy_7d=0.88,
        )
        state = await cache.get_pipeline_state()
        assert state["backtest_accuracy"] == pytest.approx(0.92)
        assert state["shadow_accuracy_7d"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_update_autonomous_contexts(self, cache):
        contexts = ["lighting", "climate"]
        await cache.update_pipeline_state(autonomous_contexts=contexts)
        state = await cache.get_pipeline_state()
        assert state["autonomous_contexts"] == contexts

    @pytest.mark.asyncio
    async def test_update_sets_updated_at(self, cache):
        state_before = await cache.get_pipeline_state()
        # Small delay so updated_at differs
        await cache.update_pipeline_state(backtest_accuracy=0.5)
        state_after = await cache.get_pipeline_state()
        assert state_after["updated_at"] >= state_before["updated_at"]

    @pytest.mark.asyncio
    async def test_unknown_field_raises(self, cache):
        with pytest.raises(ValueError, match="Unknown pipeline_state field"):
            await cache.update_pipeline_state(nonexistent_field=42)

    @pytest.mark.asyncio
    async def test_empty_update_is_noop(self, cache):
        state_before = await cache.get_pipeline_state()
        await cache.update_pipeline_state()
        state_after = await cache.get_pipeline_state()
        assert state_before["updated_at"] == state_after["updated_at"]

    @pytest.mark.asyncio
    async def test_creates_default_row_if_missing(self, cache):
        """update_pipeline_state should work even if get_pipeline_state was never called."""
        await cache.update_pipeline_state(current_stage="suggest")
        state = await cache.get_pipeline_state()
        assert state["current_stage"] == "suggest"

    @pytest.mark.asyncio
    async def test_update_suggest_approval_rate(self, cache):
        await cache.update_pipeline_state(suggest_approval_rate_14d=0.75)
        state = await cache.get_pipeline_state()
        assert state["suggest_approval_rate_14d"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_autonomous_contexts_none(self, cache):
        """Setting autonomous_contexts to None should store NULL."""
        await cache.update_pipeline_state(autonomous_contexts=["test"])
        await cache.update_pipeline_state(autonomous_contexts=None)
        state = await cache.get_pipeline_state()
        assert state["autonomous_contexts"] is None


# ============================================================================
# Existing cache operations still work
# ============================================================================


class TestExistingCacheUnaffected:
    """Verify the original cache/events tables still work after schema changes."""

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        await cache.set("test_category", {"key": "value"})
        result = await cache.get("test_category")
        assert result["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_log_and_get_events(self, cache):
        await cache.log_event("test_event", category="test", data={"x": 1})
        events = await cache.get_events(event_type="test_event")
        assert len(events) >= 1
        assert events[0]["data"] == {"x": 1}
