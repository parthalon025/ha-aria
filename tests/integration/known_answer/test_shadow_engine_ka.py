"""Known-answer tests for ShadowEngine.

Validates initialization/shutdown lifecycle, event processing through the
state_changed subscription, Thompson Sampling bucket accumulation from
deterministic events, and golden snapshot stability of internal state.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from aria.modules.shadow_engine import ShadowEngine
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Deterministic fixture events
# ---------------------------------------------------------------------------

# Entities that will be included via entity curation (tier 3 = default include)
CURATED_ENTITIES = [
    "light.living_room",
    "light.kitchen",
    "light.bedroom",
    "switch.coffee_maker",
    "switch.smart_plug",
    "media_player.tv",
]


def _make_state_changed(entity_id: str, from_state: str, to_state: str) -> dict:
    """Build a state_changed event dict matching what _on_state_changed expects.

    The handler reads: data["entity_id"], data["new_state"]["state"],
    data["old_state"]["state"], and also flat "from"/"to" via _normalize_event.
    """
    return {
        "entity_id": entity_id,
        "old_state": {"state": from_state, "attributes": {"friendly_name": entity_id.replace(".", " ").title()}},
        "new_state": {"state": to_state, "attributes": {"friendly_name": entity_id.replace(".", " ").title()}},
    }


# A morning routine: lights on, coffee maker, TV — spread across several entities
FIXTURE_EVENTS = [
    _make_state_changed("light.kitchen", "off", "on"),
    _make_state_changed("switch.coffee_maker", "off", "on"),
    _make_state_changed("light.living_room", "off", "on"),
    _make_state_changed("media_player.tv", "off", "playing"),
    _make_state_changed("light.bedroom", "on", "off"),
    _make_state_changed("switch.smart_plug", "off", "on"),
    _make_state_changed("light.kitchen", "on", "off"),
    _make_state_changed("light.living_room", "on", "off"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def shadow(hub):
    """Create a ShadowEngine with entity curation pre-populated.

    Does NOT call initialize() — tests that need subscription do so explicitly.
    This avoids the resolution loop timer running in the background.
    """
    # Pre-populate entity curation so events aren't filtered out
    for entity_id in CURATED_ENTITIES:
        await hub.cache.upsert_curation(
            entity_id=entity_id,
            status="included",
            tier=3,
            reason="known-answer fixture",
        )

    engine = ShadowEngine(hub=hub)
    yield engine

    # Ensure shutdown even if test didn't call it
    if engine._resolution_task and not engine._resolution_task.done():
        engine._resolution_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await engine._resolution_task


async def _feed_events(shadow: ShadowEngine, hub, events: list[dict]) -> None:
    """Feed events through hub.publish so the subscribe callback fires.

    Disables the prediction cooldown and stubs out prediction storage
    so we can focus on event processing and Thompson stats accumulation.
    """
    for evt in events:
        await hub.publish("state_changed", evt)
        # Small yield to let the async callback run
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initializes_and_shuts_down(shadow, hub):
    """ShadowEngine can initialize, subscribe to events, and shut down cleanly."""
    # Before init — no subscription, no resolution task
    assert shadow._resolution_task is None

    await shadow.initialize()

    # After init — resolution task is running
    assert shadow._resolution_task is not None
    assert not shadow._resolution_task.done()

    # Verify state_changed subscription is active by checking hub subscribers
    subs = hub.subscribers.get("state_changed", [])
    assert shadow._on_state_changed in subs, "Engine should be subscribed to state_changed"

    await shadow.shutdown()

    # After shutdown — task cancelled, unsubscribed
    assert shadow._resolution_task is None or shadow._resolution_task.done()
    subs_after = hub.subscribers.get("state_changed", [])
    assert shadow._on_state_changed not in subs_after, "Engine should be unsubscribed after shutdown"


@pytest.mark.asyncio
async def test_processes_state_changes(shadow, hub):
    """Feed events via hub.publish, verify internal state accumulates."""
    await shadow.initialize()

    try:
        # Stub out prediction generation so we don't need activity_summary cache
        shadow._generate_and_store_predictions = AsyncMock()

        # Feed events — _on_state_changed will buffer them
        for evt in FIXTURE_EVENTS:
            await hub.publish("state_changed", evt)
            await asyncio.sleep(0)

        # Recent events buffer should have accumulated events
        assert len(shadow._recent_events) > 0, "Recent events buffer should contain events"

        # All events should have normalized fields
        for evt in shadow._recent_events:
            assert "entity_id" in evt, "Event should have entity_id"
            assert "domain" in evt, "Event should have domain"
            assert "to" in evt, "Event should have 'to' state"
            assert "from" in evt, "Event should have 'from' state"
            assert "timestamp" in evt, "Event should have timestamp"

        # Verify domains are from our fixture entities
        domains = {evt["domain"] for evt in shadow._recent_events}
        assert domains.issubset({"light", "switch", "media_player"}), f"Unexpected domains: {domains}"
    finally:
        await shadow.shutdown()


@pytest.mark.asyncio
async def test_thompson_stats_accumulate(shadow, hub):
    """Thompson Sampling buckets accumulate after recording outcomes.

    Rather than running the full prediction→resolution loop, we directly
    exercise the ThompsonSampler with deterministic contexts and outcomes
    to verify bucket accumulation is deterministic.
    """
    sampler = shadow._thompson

    # Build deterministic contexts for different time periods
    morning_context = {
        "time_features": {"hour_sin": 0.7, "hour_cos": 0.7},
        "presence": {"home": True},
    }
    evening_context = {
        "time_features": {"hour_sin": -0.7, "hour_cos": 0.7},
        "presence": {"home": True},
    }

    # Record a series of outcomes — deterministic sequence
    outcomes = [
        (morning_context, True),
        (morning_context, True),
        (morning_context, False),
        (morning_context, True),
        (evening_context, False),
        (evening_context, True),
        (evening_context, False),
        (evening_context, False),
        (morning_context, True),
        (morning_context, True),
    ]

    for ctx, success in outcomes:
        sampler.record_outcome(ctx, success)

    stats = sampler.get_stats()

    # Verify both buckets exist
    assert "morning_home" in stats, f"Expected morning_home bucket, got keys: {list(stats.keys())}"
    assert "evening_home" in stats, f"Expected evening_home bucket, got keys: {list(stats.keys())}"

    # Morning had 5 successes, 1 failure — mean should be high
    morning = stats["morning_home"]
    assert morning["observations"] == 6, f"Expected 6 morning observations, got {morning['observations']}"
    assert morning["mean"] > 0.5, f"Morning mean should be > 0.5 (mostly successes), got {morning['mean']}"

    # Evening had 1 success, 3 failures — mean should be lower
    evening = stats["evening_home"]
    assert evening["observations"] == 4, f"Expected 4 evening observations, got {evening['observations']}"
    assert evening["mean"] < morning["mean"], (
        f"Evening mean ({evening['mean']}) should be less than morning ({morning['mean']})"
    )


@pytest.mark.asyncio
async def test_golden_snapshot(shadow, hub, update_golden):
    """Golden comparison of Thompson stats after deterministic outcome sequence."""
    sampler = shadow._thompson

    # Same deterministic sequence as test_thompson_stats_accumulate
    morning_context = {
        "time_features": {"hour_sin": 0.7, "hour_cos": 0.7},
        "presence": {"home": True},
    }
    evening_context = {
        "time_features": {"hour_sin": -0.7, "hour_cos": 0.7},
        "presence": {"home": True},
    }
    away_context = {
        "time_features": {"hour_sin": 0.7, "hour_cos": 0.7},
        "presence": {"home": False},
    }

    outcomes = [
        (morning_context, True),
        (morning_context, True),
        (morning_context, False),
        (morning_context, True),
        (evening_context, False),
        (evening_context, True),
        (evening_context, False),
        (evening_context, False),
        (morning_context, True),
        (morning_context, True),
        (away_context, True),
        (away_context, False),
        (away_context, True),
    ]

    for ctx, success in outcomes:
        sampler.record_outcome(ctx, success)

    stats = sampler.get_stats()

    # Build stable snapshot — stats are already rounded by get_stats()
    snapshot = {
        "bucket_count": len(stats),
        "buckets": {
            key: {
                "alpha": bucket["alpha"],
                "beta": bucket["beta"],
                "mean": bucket["mean"],
                "observations": bucket["observations"],
            }
            for key, bucket in sorted(stats.items())
        },
    }

    golden_compare(snapshot, "shadow_engine_thompson", update=update_golden)

    # Also snapshot the internal state serialization
    state = sampler.get_state()
    state_snapshot = {
        key: {
            "alpha": round(v["alpha"], 4),
            "beta": round(v["beta"], 4),
            "observations": v["observations"],
        }
        for key, v in sorted(state.items())
    }

    golden_compare(state_snapshot, "shadow_engine_state", update=update_golden)
