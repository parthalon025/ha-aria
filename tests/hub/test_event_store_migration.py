"""Tests for EventStore Phase 3 schema migration (context_parent_id)."""

import pytest

from aria.shared.event_store import EventStore


@pytest.fixture
async def store(tmp_path):
    s = EventStore(str(tmp_path / "test_events.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
class TestContextParentId:
    async def test_insert_with_context_parent_id(self, store):
        await store.insert_event(
            timestamp="2026-02-20T07:00:00",
            entity_id="light.bedroom",
            domain="light",
            old_state="off",
            new_state="on",
            context_parent_id=None,  # manual action
        )
        events = await store.query_events("2026-02-20T00:00:00", "2026-02-21T00:00:00")
        assert len(events) == 1
        assert events[0]["context_parent_id"] is None

    async def test_insert_automated_event(self, store):
        await store.insert_event(
            timestamp="2026-02-20T07:00:00",
            entity_id="light.bedroom",
            domain="light",
            old_state="off",
            new_state="on",
            context_parent_id="automation.morning_lights",
        )
        events = await store.query_events("2026-02-20T00:00:00", "2026-02-21T00:00:00")
        assert events[0]["context_parent_id"] == "automation.morning_lights"

    async def test_query_manual_only(self, store):
        # Insert manual + automated events
        await store.insert_event(
            timestamp="2026-02-20T07:00:00",
            entity_id="light.kitchen",
            domain="light",
            new_state="on",
            context_parent_id=None,
        )
        await store.insert_event(
            timestamp="2026-02-20T07:01:00",
            entity_id="light.bedroom",
            domain="light",
            new_state="on",
            context_parent_id="automation.morning",
        )
        manual = await store.query_manual_events("2026-02-20T00:00:00", "2026-02-21T00:00:00")
        assert len(manual) == 1
        assert manual[0]["entity_id"] == "light.kitchen"

    async def test_batch_insert_with_context(self, store):
        events = [
            ("2026-02-20T07:00:00", "light.a", "light", "off", "on", None, "bedroom", None, None),
            ("2026-02-20T07:01:00", "light.b", "light", "off", "on", None, "kitchen", None, "auto.x"),
        ]
        await store.insert_events_batch(events)
        all_events = await store.query_events("2026-02-20T00:00:00", "2026-02-21T00:00:00")
        assert len(all_events) == 2

    async def test_area_summary(self, store):
        """Test area-level aggregate query for performance tiering."""
        for i in range(10):
            await store.insert_event(
                timestamp=f"2026-02-20T07:{i:02d}:00",
                entity_id="light.bedroom",
                domain="light",
                new_state="on",
                area_id="bedroom",
            )
        for i in range(3):
            await store.insert_event(
                timestamp=f"2026-02-20T08:{i:02d}:00",
                entity_id="light.kitchen",
                domain="light",
                new_state="on",
                area_id="kitchen",
            )
        summary = await store.area_event_summary("2026-02-20T00:00:00", "2026-02-21T00:00:00")
        assert summary["bedroom"] >= 10
        assert summary["kitchen"] >= 3
