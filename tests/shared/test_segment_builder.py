"""Tests for SegmentBuilder — event-derived ML feature segments."""

import json
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from aria.shared.entity_graph import EntityGraph
from aria.shared.event_store import EventStore
from aria.shared.segment_builder import SegmentBuilder


@pytest_asyncio.fixture
async def event_store(tmp_path):
    es = EventStore(str(tmp_path / "events.db"))
    await es.initialize()
    yield es
    await es.close()


@pytest.fixture
def entity_graph():
    return EntityGraph()


@pytest.fixture
def builder(event_store, entity_graph):
    return SegmentBuilder(event_store, entity_graph)


class TestSegmentBuilderEmpty:
    @pytest.mark.asyncio
    async def test_empty_window_returns_zeros(self, builder):
        now = datetime.now(tz=UTC)
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["event_count"] == 0
        assert segment["light_transitions"] == 0
        assert segment["motion_events"] == 0
        assert segment["unique_entities_active"] == 0
        assert segment["domain_entropy"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_segment_has_start_end(self, builder):
        now = datetime.now(tz=UTC)
        start = (now - timedelta(minutes=15)).isoformat()
        end = now.isoformat()
        segment = await builder.build_segment(start, end)
        assert segment["start"] == start
        assert segment["end"] == end


class TestSegmentBuilderCounts:
    @pytest.mark.asyncio
    async def test_counts_events(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=10)).isoformat()
        for i in range(5):
            await event_store.insert_event(
                timestamp=ts,
                entity_id=f"light.test_{i}",
                domain="light",
                old_state="off",
                new_state="on",
            )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["event_count"] == 5

    @pytest.mark.asyncio
    async def test_unique_entities(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        # light.a appears twice, light.b once — 2 unique
        for eid in ["light.a", "light.b", "light.a"]:
            await event_store.insert_event(
                timestamp=ts,
                entity_id=eid,
                domain="light",
                old_state="off",
                new_state="on",
            )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["unique_entities_active"] == 2


class TestLightTransitions:
    @pytest.mark.asyncio
    async def test_counts_on_off_transitions(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.kitchen",
            domain="light",
            old_state="off",
            new_state="on",
        )
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.kitchen",
            domain="light",
            old_state="on",
            new_state="off",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["light_transitions"] == 2

    @pytest.mark.asyncio
    async def test_ignores_non_light_domains(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="switch.fan",
            domain="switch",
            old_state="off",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["light_transitions"] == 0

    @pytest.mark.asyncio
    async def test_ignores_same_state(self, builder, event_store):
        """on→on is not a transition."""
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.kitchen",
            domain="light",
            old_state="on",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["light_transitions"] == 0


class TestMotionEvents:
    @pytest.mark.asyncio
    async def test_counts_motion_events(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="binary_sensor.hallway_motion",
            domain="binary_sensor",
            old_state="off",
            new_state="on",
            attributes_json=json.dumps({"device_class": "motion"}),
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["motion_events"] == 1

    @pytest.mark.asyncio
    async def test_ignores_non_motion_binary_sensors(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="binary_sensor.door",
            domain="binary_sensor",
            old_state="off",
            new_state="on",
            attributes_json=json.dumps({"device_class": "door"}),
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["motion_events"] == 0

    @pytest.mark.asyncio
    async def test_handles_no_attributes(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="binary_sensor.hallway_motion",
            domain="binary_sensor",
            old_state="off",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["motion_events"] == 0


class TestDomainEntropy:
    @pytest.mark.asyncio
    async def test_single_domain_zero_entropy(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.a",
            domain="light",
            old_state="off",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["domain_entropy"] == 0.0

    @pytest.mark.asyncio
    async def test_multiple_domains_positive_entropy(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        for domain in ["light", "switch", "binary_sensor"]:
            await event_store.insert_event(
                timestamp=ts,
                entity_id=f"{domain}.test",
                domain=domain,
                old_state="off",
                new_state="on",
            )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        # 3 equally distributed domains → log2(3) ≈ 1.585
        assert segment["domain_entropy"] > 1.5
        assert segment["domain_entropy"] < 1.6


class TestPerAreaActivity:
    @pytest.mark.asyncio
    async def test_counts_per_area(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.kitchen",
            domain="light",
            old_state="off",
            new_state="on",
            area_id="kitchen",
        )
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.bedroom",
            domain="light",
            old_state="off",
            new_state="on",
            area_id="bedroom",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["per_area_activity"]["kitchen"] == 1
        assert segment["per_area_activity"]["bedroom"] == 1

    @pytest.mark.asyncio
    async def test_ignores_events_without_area(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.unknown",
            domain="light",
            old_state="off",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["per_area_activity"] == {}


class TestBuildSegments:
    @pytest.mark.asyncio
    async def test_consecutive_windows(self, builder, event_store):
        now = datetime.now(tz=UTC)
        start = now - timedelta(hours=1)
        # Insert one event per 15-min window
        for i in range(4):
            ts = (start + timedelta(minutes=15 * i + 5)).isoformat()
            await event_store.insert_event(
                timestamp=ts,
                entity_id="light.test",
                domain="light",
                old_state="off",
                new_state="on",
            )
        segments = await builder.build_segments(
            start.isoformat(),
            now.isoformat(),
            interval_minutes=15,
        )
        assert len(segments) == 4
        for seg in segments:
            assert seg["event_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_windows_included(self, builder, event_store):
        """All windows returned even if some have zero events."""
        now = datetime.now(tz=UTC)
        start = now - timedelta(hours=1)
        # Only insert in first window
        ts = (start + timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.test",
            domain="light",
            old_state="off",
            new_state="on",
        )
        segments = await builder.build_segments(
            start.isoformat(),
            now.isoformat(),
            interval_minutes=15,
        )
        assert len(segments) == 4
        assert segments[0]["event_count"] == 1
        assert segments[1]["event_count"] == 0
        assert segments[2]["event_count"] == 0
        assert segments[3]["event_count"] == 0

    @pytest.mark.asyncio
    async def test_per_domain_counts(self, builder, event_store):
        now = datetime.now(tz=UTC)
        ts = (now - timedelta(minutes=5)).isoformat()
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.a",
            domain="light",
            old_state="off",
            new_state="on",
        )
        await event_store.insert_event(
            timestamp=ts,
            entity_id="light.b",
            domain="light",
            old_state="off",
            new_state="on",
        )
        await event_store.insert_event(
            timestamp=ts,
            entity_id="switch.c",
            domain="switch",
            old_state="off",
            new_state="on",
        )
        segment = await builder.build_segment(
            (now - timedelta(minutes=15)).isoformat(),
            now.isoformat(),
        )
        assert segment["per_domain_counts"]["light"] == 2
        assert segment["per_domain_counts"]["switch"] == 1
