"""Known-answer tests for TrajectoryClassifier.

Validates initialization/activation lifecycle, shadow_resolved event processing
through the hub subscription, trajectory classification via the heuristic
fallback, and golden snapshot stability of classifier state.
"""

import asyncio
from unittest.mock import patch

import pytest

from aria.modules.trajectory_classifier import TrajectoryClassifier
from tests.integration.known_answer.conftest import golden_compare

# ---------------------------------------------------------------------------
# Deterministic fixture events
# ---------------------------------------------------------------------------

# shadow_resolved events with features dicts — the handler extracts sorted
# feature keys and builds a numeric vector from them.  We use "accuracy"
# and "power" as two simple numeric features so the heuristic classifier
# has something to trend on.

SHADOW_RESOLVED_EVENTS = [
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-01T07:00:00",
        "features": {"accuracy": 1.0, "power": 100.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-02T07:00:00",
        "features": {"accuracy": 1.0, "power": 110.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-03T07:00:00",
        "features": {"accuracy": 1.0, "power": 120.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-04T07:00:00",
        "features": {"accuracy": 0.0, "power": 130.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-05T07:00:00",
        "features": {"accuracy": 1.0, "power": 140.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-06T07:00:00",
        "features": {"accuracy": 1.0, "power": 150.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-07T07:00:00",
        "features": {"accuracy": 1.0, "power": 160.0},
    },
    {
        "target": "light.kitchen",
        "timestamp": "2026-02-08T07:00:00",
        "features": {"accuracy": 0.5, "power": 170.0},
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_scan_hardware():
    """Return a hardware profile that satisfies tier >= 3."""
    from aria.engine.hardware import HardwareProfile

    return HardwareProfile(
        ram_gb=32.0,
        cpu_cores=8,
        gpu_available=False,
    )


def _fake_recommend_tier(_profile):
    """Always return tier 3 — the minimum for trajectory_classifier."""
    return 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def classifier(hub):
    """Create a TrajectoryClassifier and initialize it with tier >= 3.

    Patches hardware detection so the module always activates regardless
    of the actual CI/test machine resources.
    """
    tc = TrajectoryClassifier(hub=hub)

    with (
        patch("aria.modules.trajectory_classifier.scan_hardware", _fake_scan_hardware),
        patch("aria.modules.trajectory_classifier.recommend_tier", _fake_recommend_tier),
    ):
        await tc.initialize()

    yield tc

    await tc.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initializes_and_activates(hub):
    """TrajectoryClassifier activates at tier >= 3 and subscribes to shadow_resolved."""
    tc = TrajectoryClassifier(hub=hub)
    assert tc.active is False

    with (
        patch("aria.modules.trajectory_classifier.scan_hardware", _fake_scan_hardware),
        patch("aria.modules.trajectory_classifier.recommend_tier", _fake_recommend_tier),
    ):
        await tc.initialize()

    try:
        assert tc.active is True

        # Verify subscription
        subs = hub.subscribers.get("shadow_resolved", [])
        assert tc._on_shadow_resolved in subs, "Should be subscribed to shadow_resolved"
    finally:
        await tc.shutdown()

    # Verify unsubscription (after cleanup)
    subs_after = hub.subscribers.get("shadow_resolved", [])
    assert tc._on_shadow_resolved not in subs_after, "Should be unsubscribed after shutdown"


@pytest.mark.asyncio
async def test_processes_shadow_resolved_events(classifier, hub):
    """Feed shadow_resolved events and verify internal state changes."""
    assert classifier.active is True

    # Feed all events through the hub
    for evt in SHADOW_RESOLVED_EVENTS:
        await hub.publish("shadow_resolved", evt)
        await asyncio.sleep(0)

    # Check stats — events should have been processed
    stats = classifier.get_stats()
    assert stats["active"] is True
    assert stats["shadow_events_processed"] == len(SHADOW_RESOLVED_EVENTS)
    assert "light.kitchen" in stats["window_count"]
    assert stats["window_count"]["light.kitchen"] == len(SHADOW_RESOLVED_EVENTS)

    # After 6+ events (default window_size), trajectory should be classified
    assert stats["current_trajectory"] is not None
    assert stats["current_trajectory"] in [
        "stable",
        "ramping_up",
        "winding_down",
        "anomalous_transition",
    ]

    # Check current state
    state = classifier.get_current_state()
    assert state["trajectory"] is not None
    assert state["shadow_events_processed"] == len(SHADOW_RESOLVED_EVENTS)
    assert "pattern_scales" in state
    assert len(state["pattern_scales"]) > 0


@pytest.mark.asyncio
async def test_golden_snapshot(classifier, hub, update_golden):
    """Golden comparison of classifier state after processing deterministic events."""
    # Feed all events
    for evt in SHADOW_RESOLVED_EVENTS:
        await hub.publish("shadow_resolved", evt)
        await asyncio.sleep(0)

    state = classifier.get_current_state()
    stats = classifier.get_stats()

    snapshot = {
        "trajectory": state["trajectory"],
        "shadow_events_processed": state["shadow_events_processed"],
        "pattern_scale_count": len(state["pattern_scales"]),
        "pattern_scales": state["pattern_scales"],
        "stats": {
            "active": stats["active"],
            "current_trajectory": stats["current_trajectory"],
            "shadow_events_processed": stats["shadow_events_processed"],
            "window_count": dict(stats["window_count"]),
            "sequence_classifier": stats["sequence_classifier"],
        },
    }

    golden_compare(snapshot, "trajectory_classifier", update=update_golden)
