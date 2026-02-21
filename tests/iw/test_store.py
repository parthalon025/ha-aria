"""Tests for aria.iw.store — BehavioralStateStore. TDD: written before implementation."""

from __future__ import annotations

import pytest_asyncio

from aria.iw.models import BehavioralStateDefinition, BehavioralStateTracker, Indicator
from aria.iw.store import BehavioralStateStore


def _make_trigger() -> Indicator:
    return Indicator(entity_id="binary_sensor.door", role="trigger", mode="state_change")


def _make_definition(id: str = "bsd-001", name: str = "Morning Routine") -> BehavioralStateDefinition:
    return BehavioralStateDefinition(
        id=id,
        name=name,
        trigger=_make_trigger(),
        trigger_preconditions=[],
        confirming=[],
        deviations=[],
        areas=frozenset({"kitchen"}),
        day_types=frozenset({"weekday"}),
        person_attribution=None,
        typical_duration_minutes=30.0,
        expected_outcomes=(),
    )


def _make_tracker(definition_id: str = "bsd-001", lifecycle: str = "seed") -> BehavioralStateTracker:
    return BehavioralStateTracker(definition_id=definition_id, lifecycle=lifecycle)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create and initialize a BehavioralStateStore with a temp database."""
    db_path = str(tmp_path / "iw_test.db")
    s = BehavioralStateStore(db_path)
    await s.initialize()
    yield s
    await s.close()


# ── BehavioralStateDefinition CRUD ───────────────────────────────────────────


class TestDefinitionCRUD:
    async def test_save_and_get_definition(self, store) -> None:
        defn = _make_definition()
        await store.save_definition(defn)
        retrieved = await store.get_definition("bsd-001")
        assert retrieved is not None
        assert retrieved.id == "bsd-001"
        assert retrieved.name == "Morning Routine"

    async def test_get_missing_definition_returns_none(self, store) -> None:
        result = await store.get_definition("nonexistent")
        assert result is None

    async def test_list_definitions_returns_all(self, store) -> None:
        d1 = _make_definition("bsd-001", "Morning Routine")
        d2 = _make_definition("bsd-002", "Evening Relaxation")
        await store.save_definition(d1)
        await store.save_definition(d2)
        all_defs = await store.list_definitions()
        ids = {d.id for d in all_defs}
        assert "bsd-001" in ids
        assert "bsd-002" in ids

    async def test_list_definitions_empty(self, store) -> None:
        result = await store.list_definitions()
        assert result == []

    async def test_save_definition_overwrites(self, store) -> None:
        defn = _make_definition()
        await store.save_definition(defn)
        # Save again with same id but different name
        updated = BehavioralStateDefinition(
            id="bsd-001",
            name="Updated Name",
            trigger=_make_trigger(),
            trigger_preconditions=[],
            confirming=[],
            deviations=[],
            areas=frozenset(),
            day_types=frozenset(),
            person_attribution=None,
            typical_duration_minutes=0.0,
            expected_outcomes=(),
        )
        await store.save_definition(updated)
        retrieved = await store.get_definition("bsd-001")
        assert retrieved is not None
        assert retrieved.name == "Updated Name"

    async def test_delete_definition(self, store) -> None:
        defn = _make_definition()
        await store.save_definition(defn)
        await store.delete_definition("bsd-001")
        retrieved = await store.get_definition("bsd-001")
        assert retrieved is None

    async def test_delete_definition_cascades_tracker(self, store) -> None:
        defn = _make_definition()
        tracker = _make_tracker("bsd-001")
        await store.save_definition(defn)
        await store.save_tracker(tracker)
        await store.delete_definition("bsd-001")
        retrieved_tracker = await store.get_tracker("bsd-001")
        assert retrieved_tracker is None

    async def test_delete_definition_cascades_co_activations(self, store) -> None:
        d1 = _make_definition("bsd-001")
        d2 = _make_definition("bsd-002", "Evening")
        await store.save_definition(d1)
        await store.save_definition(d2)
        await store.record_co_activation("bsd-001", "bsd-002")
        await store.delete_definition("bsd-001")
        co_acts = await store.get_co_activations(min_count=1)
        assert len(co_acts) == 0


# ── BehavioralStateTracker CRUD ───────────────────────────────────────────────


class TestTrackerCRUD:
    async def test_save_and_get_tracker(self, store) -> None:
        tracker = _make_tracker("bsd-001", "emerging")
        tracker.record_observation("2026-01-01T08:00:00", 0.9)
        await store.save_tracker(tracker)
        retrieved = await store.get_tracker("bsd-001")
        assert retrieved is not None
        assert retrieved.definition_id == "bsd-001"
        assert retrieved.lifecycle == "emerging"
        assert retrieved.observation_count == 1
        assert abs(retrieved.consistency - 0.9) < 1e-9

    async def test_get_missing_tracker_returns_none(self, store) -> None:
        result = await store.get_tracker("nonexistent")
        assert result is None

    async def test_list_trackers_returns_all(self, store) -> None:
        t1 = _make_tracker("bsd-001", "seed")
        t2 = _make_tracker("bsd-002", "emerging")
        await store.save_tracker(t1)
        await store.save_tracker(t2)
        all_trackers = await store.list_trackers()
        ids = {t.definition_id for t in all_trackers}
        assert "bsd-001" in ids
        assert "bsd-002" in ids

    async def test_list_trackers_lifecycle_filter(self, store) -> None:
        t1 = _make_tracker("bsd-001", "seed")
        t2 = _make_tracker("bsd-002", "emerging")
        t3 = _make_tracker("bsd-003", "confirmed")
        await store.save_tracker(t1)
        await store.save_tracker(t2)
        await store.save_tracker(t3)
        emerging_only = await store.list_trackers(lifecycle_filter="emerging")
        assert len(emerging_only) == 1
        assert emerging_only[0].definition_id == "bsd-002"

    async def test_list_trackers_empty(self, store) -> None:
        result = await store.list_trackers()
        assert result == []


# ── Co-activation ─────────────────────────────────────────────────────────────


class TestCoActivation:
    async def test_record_co_activation_increments(self, store) -> None:
        await store.record_co_activation("bsd-001", "bsd-002")
        co_acts = await store.get_co_activations(min_count=1)
        assert len(co_acts) >= 1
        # Find the (bsd-001, bsd-002) or (bsd-002, bsd-001) pair
        pair = next(
            (ca for ca in co_acts if set(ca[:2]) == {"bsd-001", "bsd-002"}),
            None,
        )
        assert pair is not None
        assert pair[2] == 1

    async def test_record_co_activation_increments_again(self, store) -> None:
        await store.record_co_activation("bsd-001", "bsd-002")
        await store.record_co_activation("bsd-001", "bsd-002")
        co_acts = await store.get_co_activations(min_count=1)
        pair = next(
            (ca for ca in co_acts if set(ca[:2]) == {"bsd-001", "bsd-002"}),
            None,
        )
        assert pair is not None
        assert pair[2] == 2

    async def test_get_co_activations_respects_min_count(self, store) -> None:
        await store.record_co_activation("bsd-001", "bsd-002")
        await store.record_co_activation("bsd-003", "bsd-004")
        await store.record_co_activation("bsd-003", "bsd-004")
        # Only pairs with count >= 2
        result = await store.get_co_activations(min_count=2)
        assert len(result) == 1
        pair = result[0]
        assert set(pair[:2]) == {"bsd-003", "bsd-004"}
        assert pair[2] == 2

    async def test_get_co_activations_empty(self, store) -> None:
        result = await store.get_co_activations(min_count=1)
        assert result == []
