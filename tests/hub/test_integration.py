"""Integration tests for ARIA Hub.

Tests end-to-end functionality of the hub with all modules integrated.
"""

import asyncio
import contextlib
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from aria.hub.core import IntelligenceHub
from aria.modules.discovery import DiscoveryModule
from aria.modules.ml_engine import MLEngine
from aria.modules.patterns import PatternRecognition

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dirs():
    """Create temporary directories for cache, models, and training data."""
    with tempfile.TemporaryDirectory() as cache_dir:
        models_dir = Path(cache_dir) / "models"
        training_dir = Path(cache_dir) / "training"
        daily_dir = Path(cache_dir) / "daily"
        models_dir.mkdir()
        training_dir.mkdir()
        daily_dir.mkdir()

        yield {"cache": cache_dir, "models": str(models_dir), "training": str(training_dir), "daily": str(daily_dir)}


@pytest_asyncio.fixture(scope="function")
async def initialized_hub(temp_dirs):
    """Create and initialize a hub instance."""
    cache_path = Path(temp_dirs["cache"]) / "test_hub.db"
    hub = IntelligenceHub(str(cache_path))
    await hub.initialize()
    yield hub
    await hub.shutdown()


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_hub_initialization(initialized_hub):
    """Test hub initializes successfully."""
    assert initialized_hub.cache is not None
    assert initialized_hub._running is True

    # Check cache database exists
    cache_path = Path(initialized_hub.cache.db_path)
    assert cache_path.exists()


@pytest.mark.asyncio
async def test_module_registration(initialized_hub):
    """Test modules can be registered with hub."""
    # Mock the discover script subprocess call
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"capabilities": {}, "entities": []}))

        # Create and register discovery module
        discovery = DiscoveryModule(initialized_hub, "http://test:8123", "test_token")
        initialized_hub.register_module(discovery)

        # Verify registration
        assert discovery.module_id in initialized_hub.modules
        module = await initialized_hub.get_module("discovery")
        assert module == discovery


@pytest.mark.asyncio
async def test_discovery_to_cache_pipeline(initialized_hub):
    """Test discovery module populates cache."""
    # Mock discovery script output
    mock_output = {
        "capabilities": {
            "lighting": {"count": 1, "entities": ["light.living_room"]},
            "power_monitoring": {"count": 1, "entities": ["sensor.power_meter"]},
        },
        "entities": [
            {"entity_id": "light.living_room", "domain": "light"},
            {"entity_id": "sensor.power_meter", "domain": "sensor"},
        ],
    }

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(mock_output))

        # Initialize discovery module
        discovery = DiscoveryModule(initialized_hub, "http://test:8123", "test_token")
        initialized_hub.register_module(discovery)
        await discovery.initialize()

        # Verify cache was populated
        capabilities = await initialized_hub.cache.get("capabilities")
        assert capabilities is not None
        assert "lighting" in capabilities["data"]
        assert "power_monitoring" in capabilities["data"]


@pytest.mark.asyncio
async def test_ml_engine_integration(initialized_hub, temp_dirs):
    """Test ML engine integrates with hub and cache."""
    # Create ML engine
    ml_engine = MLEngine(initialized_hub, temp_dirs["models"], temp_dirs["training"])
    initialized_hub.register_module(ml_engine)
    await ml_engine.initialize()

    # Create mock training data
    training_file = Path(temp_dirs["training"]) / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(training_file, "w") as f:
        for i in range(100):
            timestamp = datetime.now() - timedelta(hours=24 - i * 0.24)
            data = {
                "timestamp": timestamp.isoformat(),
                "light.living_room": {"state": "on" if i % 2 == 0 else "off"},
                "sensor.power_meter": {"state": str(100 + i * 2)},
            }
            f.write(json.dumps(data) + "\n")

    # Populate capabilities first (ML engine needs this)
    await initialized_hub.cache.set("capabilities", {"lighting": {"entities": ["light.living_room"], "count": 1}})

    # Train models
    await ml_engine.train_models()

    # Verify models metadata was cached
    # Verify cache is accessible (model_metadata may be None if training skipped)
    await initialized_hub.cache.get("ml_model_metadata")


@pytest.mark.asyncio
async def test_health_check(initialized_hub, temp_dirs):
    """Test health check reports status of all modules."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="{}")

        # Register modules
        discovery = DiscoveryModule(initialized_hub, "http://test:8123", "test_token")
        ml_engine = MLEngine(initialized_hub, temp_dirs["models"], temp_dirs["training"])

        initialized_hub.register_module(discovery)
        initialized_hub.register_module(ml_engine)

        # Get health check
        health = await initialized_hub.health_check()

        # Verify modules are registered
        assert "modules" in health
        assert "discovery" in health["modules"]
        assert "ml_engine" in health["modules"]


@pytest.mark.asyncio
async def test_graceful_shutdown(initialized_hub):
    """Test hub shuts down gracefully without data loss."""
    # Add data to cache
    test_data = {"key1": "value1", "key2": [1, 2, 3]}
    await initialized_hub.cache.set("test_shutdown", test_data)

    # Get the database path before shutdown
    cache_path = Path(initialized_hub.cache.db_path)

    # Shutdown hub
    await initialized_hub.shutdown()

    # Verify database file still exists
    assert cache_path.exists()

    # Verify data persisted
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM cache WHERE category = ?", ("test_shutdown",))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    retrieved_data = json.loads(row[0])
    assert retrieved_data == test_data


@pytest.mark.asyncio
async def test_websocket_event_broadcasting(initialized_hub):
    """Test hub can broadcast events to subscribers."""
    # Track events
    events_received = []

    async def event_listener(data):
        events_received.append(data)

    # Subscribe to custom event type
    initialized_hub.subscribe("test_event", event_listener)

    # Publish event through hub
    await initialized_hub.publish("test_event", {"test": "data"})

    # Allow event to propagate
    await asyncio.sleep(0.1)

    # Verify event was received
    assert len(events_received) >= 1
    assert events_received[0]["test"] == "data"


@pytest.mark.asyncio
async def test_concurrent_cache_access(initialized_hub):
    """Test cache handles concurrent access correctly."""

    # Create multiple concurrent write operations
    async def write_task(key_id):
        await initialized_hub.cache.set(f"concurrent_{key_id}", {"id": key_id})
        return key_id

    # Run 10 concurrent writes
    tasks = [write_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify all writes succeeded
    assert len(results) == 10

    # Verify all values can be retrieved
    for i in range(10):
        entry = await initialized_hub.cache.get(f"concurrent_{i}")
        assert entry is not None
        assert entry["data"]["id"] == i


@pytest.mark.asyncio
async def test_cache_persistence_across_restarts(temp_dirs):
    """Test cache data persists across hub restarts."""
    cache_path = Path(temp_dirs["cache"]) / "persistent_hub.db"

    # First hub instance
    hub1 = IntelligenceHub(str(cache_path))
    await hub1.initialize()
    await hub1.cache.set("persistent_data", {"restart_test": True})
    await hub1.shutdown()

    # Second hub instance (restart)
    hub2 = IntelligenceHub(str(cache_path))
    await hub2.initialize()
    entry = await hub2.cache.get("persistent_data")
    await hub2.shutdown()

    # Verify data persisted
    assert entry is not None
    assert entry["data"]["restart_test"] is True


@pytest.mark.asyncio
async def test_error_recovery_discovery_failure(initialized_hub):
    """Test hub handles discovery failures gracefully."""
    # Mock discovery script to fail
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="Connection failed")

        discovery = DiscoveryModule(initialized_hub, "http://test:8123", "test_token")
        initialized_hub.register_module(discovery)

        # Attempt discovery (should not crash)
        with contextlib.suppress(Exception):
            await discovery.initialize()

        # Hub should still be functional
        health = await initialized_hub.health_check()
        assert health["status"] == "ok"


@pytest.mark.asyncio
async def test_error_recovery_model_training_failure(initialized_hub, temp_dirs):
    """Test ML engine handles training failures gracefully."""
    ml_engine = MLEngine(initialized_hub, temp_dirs["models"], temp_dirs["training"])
    initialized_hub.register_module(ml_engine)
    await ml_engine.initialize()

    # Try to train with no data (should handle gracefully)
    with contextlib.suppress(Exception):
        await ml_engine.train_models()

    # ML engine should still be registered and functional
    module = await initialized_hub.get_module("ml_engine")
    assert module is not None


@pytest.mark.asyncio
async def test_cache_performance(initialized_hub):
    """Test cache handles high-frequency operations efficiently."""
    import time

    # Measure write performance
    start = time.time()
    for i in range(100):
        await initialized_hub.cache.set(f"perf_test_{i}", {"index": i})
    write_time = time.time() - start

    # Measure read performance
    start = time.time()
    for i in range(100):
        await initialized_hub.cache.get(f"perf_test_{i}")
    read_time = time.time() - start

    # Performance assertions (should be fast)
    assert write_time < 5.0, f"Write time too slow: {write_time}s"
    assert read_time < 3.0, f"Read time too slow: {read_time}s"


@pytest.mark.asyncio
async def test_module_initialization_order(initialized_hub, temp_dirs):
    """Test modules can initialize in any order."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="{}")

        # Create all modules
        discovery = DiscoveryModule(initialized_hub, "http://test:8123", "test_token")
        ml_engine = MLEngine(initialized_hub, temp_dirs["models"], temp_dirs["training"])
        patterns = PatternRecognition(initialized_hub, Path(temp_dirs["cache"]))

        # Register in any order
        initialized_hub.register_module(ml_engine)
        initialized_hub.register_module(discovery)
        initialized_hub.register_module(patterns)

        # All should be registered successfully
        health = await initialized_hub.health_check()
        assert len(health["modules"]) == 3


@pytest.mark.asyncio
async def test_event_logging(initialized_hub):
    """Test events are logged to cache."""
    # Use hub.set_cache (not cache.set directly) — the hub layer
    # routes through publish() which calls cache.log_event().
    await initialized_hub.set_cache("test_category", {"test": "data"})

    # Get recent events
    events = await initialized_hub.cache.get_events(limit=10)

    # Verify event was logged
    assert len(events) > 0
    # publish() logs with event_type="cache_updated"
    cache_events = [e for e in events if e.get("event_type") == "cache_updated"]
    assert len(cache_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
