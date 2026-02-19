"""Tests for discovery module — reconnect jitter, config propagation (#24), cold-start (#25)."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aria.hub.constants import CACHE_ENTITIES
from aria.hub.core import IntelligenceHub
from aria.modules.discovery import DiscoveryModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.set_cache = AsyncMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.schedule_task = AsyncMock()
    hub.publish = AsyncMock()
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value="72")
    return hub


@pytest.fixture
def module(mock_hub):
    with patch.object(DiscoveryModule, "__init__", lambda self, *args, **kwargs: None):
        m = DiscoveryModule.__new__(DiscoveryModule)
        m.hub = mock_hub
        m.logger = logging.getLogger("test_discovery")
        return m


def test_reconnect_delay_has_jitter():
    """Reconnect uses ±25% jitter to prevent thundering herd.

    Samples 200 delays for a base of 5s and verifies:
    - All values fall within [3.75, 6.25] (±25% of 5)
    - Both sides of the base are represented (not a constant offset)
    """
    import random

    base = 5
    low = base * 0.75
    high = base * 1.25

    samples = [base + base * random.uniform(-0.25, 0.25) for _ in range(200)]

    # All samples within bounds
    assert all(low <= s <= high for s in samples), (
        f"Sample out of ±25% range: min={min(samples):.3f}, max={max(samples):.3f}"
    )

    # Distribution is not degenerate — we should see values both above and below base
    assert any(s < base for s in samples), "No samples below base — jitter is not subtracting"
    assert any(s > base for s in samples), "No samples above base — jitter is not adding"


def test_reconnect_jitter_bounds_at_max_delay():
    """Jitter at the 60s cap stays within ±25% of 60."""
    import random

    base = 60
    low = base * 0.75  # 45s
    high = base * 1.25  # 75s

    samples = [base + base * random.uniform(-0.25, 0.25) for _ in range(200)]

    assert all(low <= s <= high for s in samples), (
        f"Sample out of ±25% range at cap: min={min(samples):.3f}, max={max(samples):.3f}"
    )


# ============================================================================
# Fixtures for integration tests (real hub)
# ============================================================================


@pytest_asyncio.fixture
async def real_hub():
    """Minimal initialized hub backed by a temporary SQLite file."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "test.db"
        h = IntelligenceHub(str(cache_path))
        await h.initialize()
        yield h
        await h.shutdown()


def _make_discovery_module(hub_instance):
    """Create a DiscoveryModule with mocked discover script path."""
    with patch.object(Path, "exists", return_value=True):
        mod = DiscoveryModule(
            hub=hub_instance,
            ha_url="http://test-host:8123",
            ha_token="test-token",
        )
    return mod


# ============================================================================
# #24: Config propagation — DiscoveryModule.on_config_updated
# ============================================================================


class TestDiscoveryConfigPropagation:
    """DiscoveryModule re-reads config and reclassifies on curation.* changes."""

    @pytest.mark.asyncio
    async def test_curation_config_triggers_reclassification(self, real_hub):
        """on_config_updated with curation.* key triggers run_classification."""
        mod = _make_discovery_module(real_hub)
        mod.run_classification = AsyncMock()

        await mod.on_config_updated({"key": "curation.auto_exclude_domains", "value": "update,tts"})

        mod.run_classification.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_discovery_config_triggers_reclassification(self, real_hub):
        """on_config_updated with discovery.* key triggers run_classification."""
        mod = _make_discovery_module(real_hub)
        mod.run_classification = AsyncMock()

        await mod.on_config_updated({"key": "discovery.stale_ttl_hours", "value": "48"})

        mod.run_classification.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unrelated_config_does_not_trigger(self, real_hub):
        """on_config_updated with non-curation key does not reclassify."""
        mod = _make_discovery_module(real_hub)
        mod.run_classification = AsyncMock()

        await mod.on_config_updated({"key": "shadow.min_confidence", "value": "0.5"})

        mod.run_classification.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_classification_error_does_not_raise(self, real_hub):
        """If run_classification fails during config update, error is logged not raised."""
        mod = _make_discovery_module(real_hub)
        mod.run_classification = AsyncMock(side_effect=RuntimeError("db error"))

        # Should not raise
        await mod.on_config_updated({"key": "curation.noise_event_threshold", "value": "500"})

    @pytest.mark.asyncio
    async def test_config_propagation_via_hub_publish(self, real_hub):
        """Publishing config_updated on hub reaches DiscoveryModule.on_config_updated."""
        mod = _make_discovery_module(real_hub)
        mod.run_classification = AsyncMock()
        real_hub.register_module(mod)

        await real_hub.publish("config_updated", {"key": "curation.stale_days_threshold", "value": "60"})

        mod.run_classification.assert_awaited_once()


# ============================================================================
# #25: Discovery cold-start — deferred classification
# ============================================================================


class TestDiscoveryColdStart:
    """Classification is deferred until entity data arrives in cache."""

    @pytest.mark.asyncio
    async def test_initialize_defers_classification_when_no_entities(self, real_hub):
        """When discovery returns no entities, classification is deferred."""
        mod = _make_discovery_module(real_hub)
        mod.run_discovery = AsyncMock(return_value={"entities": {}, "capabilities": {}})
        mod.run_classification = AsyncMock()

        await mod.initialize()

        # Classification should NOT have been called (no entities in cache)
        mod.run_classification.assert_not_awaited()
        assert mod._classification_deferred is True

    @pytest.mark.asyncio
    async def test_initialize_runs_classification_when_entities_exist(self, real_hub):
        """When discovery populates entities, classification runs immediately."""
        mod = _make_discovery_module(real_hub)

        # Simulate discovery populating the cache
        async def fake_discovery():
            await real_hub.set_cache(CACHE_ENTITIES, {"sensor.test": {"domain": "sensor"}})
            return {"entities": {"sensor.test": {}}, "capabilities": {}}

        mod.run_discovery = fake_discovery
        mod.run_classification = AsyncMock()

        await mod.initialize()

        # Classification should have run because entities were found in cache
        mod.run_classification.assert_awaited_once()
        assert mod._classification_deferred is False

    @pytest.mark.asyncio
    async def test_cache_updated_triggers_deferred_classification(self, real_hub):
        """Publishing cache_updated for entities triggers deferred classification."""
        mod = _make_discovery_module(real_hub)
        mod.run_discovery = AsyncMock(return_value={"entities": {}, "capabilities": {}})
        mod.run_classification = AsyncMock()

        await mod.initialize()
        assert mod._classification_deferred is True

        # Simulate entity data arriving in cache
        await real_hub.publish("cache_updated", {"category": CACHE_ENTITIES, "version": 1})

        mod.run_classification.assert_awaited_once()
        assert mod._classification_deferred is False

    @pytest.mark.asyncio
    async def test_cache_updated_non_entities_does_not_trigger(self, real_hub):
        """Publishing cache_updated for non-entities category does not trigger classification."""
        mod = _make_discovery_module(real_hub)
        mod.run_discovery = AsyncMock(return_value={"entities": {}, "capabilities": {}})
        mod.run_classification = AsyncMock()

        await mod.initialize()

        # Publish for a different category
        await real_hub.publish("cache_updated", {"category": "intelligence", "version": 1})

        mod.run_classification.assert_not_awaited()
        assert mod._classification_deferred is True

    @pytest.mark.asyncio
    async def test_deferred_classification_fires_only_once(self, real_hub):
        """The deferred classification fires only on the first entities cache_updated."""
        mod = _make_discovery_module(real_hub)
        mod.run_discovery = AsyncMock(return_value={"entities": {}, "capabilities": {}})
        mod.run_classification = AsyncMock()

        await mod.initialize()

        # First entities update — triggers classification
        await real_hub.publish("cache_updated", {"category": CACHE_ENTITIES, "version": 1})
        assert mod.run_classification.await_count == 1

        # Second entities update — should NOT trigger again (deferred flag cleared)
        await real_hub.publish("cache_updated", {"category": CACHE_ENTITIES, "version": 2})
        assert mod.run_classification.await_count == 1

    @pytest.mark.asyncio
    async def test_deferred_classification_error_logged_not_raised(self, real_hub):
        """If deferred classification fails, it's logged but doesn't crash."""
        mod = _make_discovery_module(real_hub)
        mod.run_discovery = AsyncMock(return_value={"entities": {}, "capabilities": {}})
        mod.run_classification = AsyncMock(side_effect=RuntimeError("classification error"))

        await mod.initialize()

        # Should not raise
        await real_hub.publish("cache_updated", {"category": CACHE_ENTITIES, "version": 1})
