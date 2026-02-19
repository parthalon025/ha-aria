"""Tests for performance batch: #51, #52, #53, #54, #55, #58."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aria.hub.cache import _EVENT_BUFFER_MAX_SIZE, CacheManager

# ============================================================================
# #51: Async Subprocess in Discovery
# ============================================================================


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.set_cache = AsyncMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.schedule_task = AsyncMock()
    hub.publish = AsyncMock()
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value="72")
    hub.cache.get_curations_batch = AsyncMock(return_value={})
    hub.cache.upsert_curations_batch = AsyncMock(return_value=0)
    return hub


@pytest.fixture
def discovery_module(mock_hub):
    from aria.modules.discovery import DiscoveryModule

    with patch.object(DiscoveryModule, "__init__", lambda self, *a, **kw: None):
        m = DiscoveryModule.__new__(DiscoveryModule)
        m.hub = mock_hub
        m.ha_url = "http://test-host:8123"
        m.ha_token = "test-token"
        m.discover_script = Path(__file__).parent.parent.parent / "bin" / "discover.py"
        m.logger = logging.getLogger("test_discovery_async")
        return m


@pytest.mark.asyncio
async def test_discovery_uses_async_subprocess(discovery_module):
    """#51: run_discovery() uses asyncio.create_subprocess_exec, not subprocess.run."""
    # The module should not import subprocess anymore
    import inspect

    import aria.modules.discovery as disc_mod

    source = inspect.getsource(disc_mod.DiscoveryModule.run_discovery)
    assert "subprocess.run" not in source, "run_discovery still uses subprocess.run"
    assert "create_subprocess_exec" in source, "run_discovery should use asyncio.create_subprocess_exec"


@pytest.mark.asyncio
async def test_discovery_async_subprocess_success(discovery_module):
    """#51: Async subprocess returns valid JSON output."""
    fake_output = json.dumps(
        {
            "entities": {},
            "devices": {},
            "areas": {},
            "capabilities": {},
            "entity_count": 0,
            "device_count": 0,
            "area_count": 0,
            "timestamp": "2026-01-01T00:00:00",
            "ha_version": "2024.1.0",
        }
    ).encode()

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(fake_output, b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await discovery_module.run_discovery()

    assert result["entity_count"] == 0
    assert "capabilities" in result


@pytest.mark.asyncio
async def test_discovery_async_subprocess_timeout(discovery_module):
    """#51: Async subprocess raises on timeout."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        patch("asyncio.wait_for", side_effect=TimeoutError()),
        pytest.raises(asyncio.TimeoutError),
    ):
        await discovery_module.run_discovery()


@pytest.mark.asyncio
async def test_discovery_async_subprocess_nonzero_exit(discovery_module):
    """#51: Async subprocess raises RuntimeError on non-zero exit."""
    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
    mock_proc.returncode = 1
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        pytest.raises(RuntimeError, match="Discovery failed"),
    ):
        await discovery_module.run_discovery()


# ============================================================================
# #52: Wrap Blocking I/O in asyncio.to_thread
# ============================================================================


@pytest.mark.asyncio
async def test_ml_engine_load_models_uses_to_thread():
    """#52: _load_models wraps pickle.load in asyncio.to_thread."""
    import inspect

    from aria.modules.ml_engine import MLEngine

    source = inspect.getsource(MLEngine._load_models)
    assert "asyncio.to_thread" in source, "_load_models should use asyncio.to_thread"


@pytest.mark.asyncio
async def test_ml_engine_save_model_uses_to_thread():
    """#52: _train_model_for_target wraps pickle.dump in asyncio.to_thread."""
    import inspect

    from aria.modules.ml_engine import MLEngine

    source = inspect.getsource(MLEngine._train_model_for_target)
    assert "asyncio.to_thread" in source, "_train_model_for_target should use asyncio.to_thread"


@pytest.mark.asyncio
async def test_ml_engine_save_anomaly_uses_to_thread():
    """#52: _train_anomaly_detector wraps pickle.dump in asyncio.to_thread."""
    import inspect

    from aria.modules.ml_engine import MLEngine

    source = inspect.getsource(MLEngine._train_anomaly_detector)
    assert "asyncio.to_thread" in source, "_train_anomaly_detector should use asyncio.to_thread"


@pytest.mark.asyncio
async def test_intelligence_read_uses_to_thread():
    """#52: intelligence initialize wraps _read_intelligence_data in asyncio.to_thread."""
    import inspect

    from aria.modules.intelligence import IntelligenceModule

    source = inspect.getsource(IntelligenceModule.initialize)
    assert "asyncio.to_thread" in source, "initialize should use asyncio.to_thread"


# ============================================================================
# #53: Batch SQLite Queries in Discovery Classification
# ============================================================================


@pytest_asyncio.fixture
async def cache():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test.db")
        cm = CacheManager(db_path)
        await cm.initialize()
        yield cm
        await cm.close()


@pytest.mark.asyncio
async def test_get_curations_batch_empty(cache):
    """#53: get_curations_batch returns empty dict for no matches."""
    result = await cache.get_curations_batch(["sensor.does_not_exist"])
    assert result == {}


@pytest.mark.asyncio
async def test_get_curations_batch_returns_matching(cache):
    """#53: get_curations_batch returns matching curations."""
    await cache.upsert_curation("sensor.a", "included", 3, reason="test")
    await cache.upsert_curation("sensor.b", "excluded", 1, reason="noise")
    await cache.upsert_curation("sensor.c", "included", 3, reason="general")

    result = await cache.get_curations_batch(["sensor.a", "sensor.c"])
    assert len(result) == 2
    assert result["sensor.a"]["status"] == "included"
    assert result["sensor.c"]["status"] == "included"
    assert "sensor.b" not in result


@pytest.mark.asyncio
async def test_upsert_curations_batch(cache):
    """#53: upsert_curations_batch inserts multiple records in one transaction."""
    records = [
        {
            "entity_id": "sensor.x",
            "status": "included",
            "tier": 3,
            "reason": "ok",
            "metrics": None,
            "group_id": "",
            "decided_by": "test",
        },
        {
            "entity_id": "sensor.y",
            "status": "excluded",
            "tier": 1,
            "reason": "noise",
            "metrics": {"rate": 100},
            "group_id": "",
            "decided_by": "test",
        },
    ]
    count = await cache.upsert_curations_batch(records)
    assert count == 2

    # Verify they were written
    result = await cache.get_curations_batch(["sensor.x", "sensor.y"])
    assert result["sensor.x"]["status"] == "included"
    assert result["sensor.y"]["tier"] == 1


@pytest.mark.asyncio
async def test_upsert_curations_batch_updates_existing(cache):
    """#53: upsert_curations_batch updates existing records (upsert)."""
    await cache.upsert_curation("sensor.a", "included", 3, reason="old")

    records = [
        {
            "entity_id": "sensor.a",
            "status": "excluded",
            "tier": 1,
            "reason": "new",
            "metrics": None,
            "group_id": "",
            "decided_by": "test",
        },
    ]
    await cache.upsert_curations_batch(records)

    result = await cache.get_curation("sensor.a")
    assert result["status"] == "excluded"
    assert result["reason"] == "new"


@pytest.mark.asyncio
async def test_classification_uses_batch_methods(mock_hub):
    """#53: run_classification uses get_curations_batch and upsert_curations_batch."""
    from aria.modules.discovery import (
        DEFAULT_AUTO_EXCLUDE_DOMAINS,
        DEFAULT_NOISE_EVENT_THRESHOLD,
        DEFAULT_STALE_DAYS_THRESHOLD,
        DEFAULT_UNAVAILABLE_GRACE_HOURS,
        DEFAULT_VEHICLE_PATTERNS,
        DiscoveryModule,
    )

    with patch.object(DiscoveryModule, "__init__", lambda self, *a, **kw: None):
        m = DiscoveryModule.__new__(DiscoveryModule)
        m.hub = mock_hub
        m.logger = logging.getLogger("test_classification_batch")

    # Set up entity data in mock cache
    entities = {
        "sensor.temp": {"domain": "sensor", "area_id": "kitchen", "device_class": "temperature"},
        "sensor.power": {"domain": "sensor", "area_id": "utility", "device_class": "power"},
    }
    mock_hub.get_cache = AsyncMock(
        side_effect=[
            {"data": entities},  # CACHE_ENTITIES
            None,  # CACHE_ACTIVITY_LOG
        ]
    )

    # Return proper typed defaults for config values
    config_defaults = {
        "curation.auto_exclude_domains": DEFAULT_AUTO_EXCLUDE_DOMAINS,
        "curation.noise_event_threshold": DEFAULT_NOISE_EVENT_THRESHOLD,
        "curation.stale_days_threshold": DEFAULT_STALE_DAYS_THRESHOLD,
        "curation.vehicle_patterns": DEFAULT_VEHICLE_PATTERNS,
        "curation.unavailable_grace_hours": DEFAULT_UNAVAILABLE_GRACE_HOURS,
    }

    async def _mock_config(key, default=None):
        return config_defaults.get(key, default)

    mock_hub.cache.get_config_value = AsyncMock(side_effect=_mock_config)
    mock_hub.cache.get_curations_batch = AsyncMock(return_value={})
    mock_hub.cache.upsert_curations_batch = AsyncMock(return_value=2)

    await m.run_classification()

    # Verify batch methods were called
    mock_hub.cache.get_curations_batch.assert_called_once()
    mock_hub.cache.upsert_curations_batch.assert_called_once()

    # Verify the batch contains the right number of records
    batch_arg = mock_hub.cache.upsert_curations_batch.call_args[0][0]
    assert len(batch_arg) == 2


# ============================================================================
# #54: Buffer Event Logging
# ============================================================================


@pytest.mark.asyncio
async def test_log_event_buffers(cache):
    """#54: log_event buffers events instead of writing immediately."""
    await cache.log_event("test_event", category="test", data={"key": "value"})

    # Event should be in buffer, not yet flushed to DB (< 50 threshold)
    assert len(cache._event_buffer) >= 1
    # Add another event
    await cache.log_event("buffered_event", category="test")
    assert len(cache._event_buffer) >= 2
    # The important thing is that events eventually appear in DB
    await cache._flush_event_buffer()

    events = await cache.get_events(event_type="buffered_event")
    assert len(events) >= 1


@pytest.mark.asyncio
async def test_log_event_flushes_at_max(cache):
    """#54: Buffer flushes when it reaches _EVENT_BUFFER_MAX_SIZE."""
    # Fill buffer to max
    for i in range(_EVENT_BUFFER_MAX_SIZE):
        await cache.log_event("bulk_event", category="test", data={"i": i})

    # At max size, buffer should have been flushed
    assert len(cache._event_buffer) == 0

    # Events should be in DB
    events = await cache.get_events(event_type="bulk_event", limit=200)
    assert len(events) == _EVENT_BUFFER_MAX_SIZE


@pytest.mark.asyncio
async def test_cache_set_atomic_version(cache):
    """#54: cache.set() increments version atomically without read-before-write."""
    v1 = await cache.set("test_cat", {"a": 1})
    assert v1 == 1

    v2 = await cache.set("test_cat", {"a": 2})
    assert v2 == 2

    v3 = await cache.set("test_cat", {"a": 3})
    assert v3 == 3

    # Verify data is correct
    entry = await cache.get("test_cat")
    assert entry["data"] == {"a": 3}
    assert entry["version"] == 3


@pytest.mark.asyncio
async def test_event_buffer_flush_on_close(cache):
    """#54: Closing the cache flushes remaining buffered events."""
    await cache.log_event("close_event", category="test")
    # Don't manually flush — close should do it

    # We need to verify events are flushed on close.
    # Close triggers flush via task cancellation.
    await cache.close()

    # Re-open to check
    cache2 = CacheManager(cache.db_path)
    await cache2.initialize()
    events = await cache2.get_events(event_type="close_event")
    assert len(events) >= 1
    await cache2.close()


# ============================================================================
# #55: Parallel Module Initialization
# ============================================================================


def test_register_modules_has_tier_comments():
    """#55: _register_modules organizes modules into dependency tiers."""
    import inspect

    from aria.cli import _register_modules

    source = inspect.getsource(_register_modules)
    assert "Tier 0" in source, "Should document Tier 0 (discovery)"
    assert "Tier 1" in source, "Should document Tier 1 (parallel core)"
    assert "asyncio.gather" in source, "Should use asyncio.gather for parallel init"


def test_register_modules_uses_gather():
    """#55: _register_modules uses asyncio.gather for tier 1 modules."""
    import inspect

    from aria.cli import _register_modules

    source = inspect.getsource(_register_modules)
    assert "return_exceptions=True" in source, "gather should use return_exceptions=True"


# ============================================================================
# #58: Shared aiohttp Session in Presence
# ============================================================================


@pytest.fixture
def presence_module():
    from aria.modules.presence import PresenceModule

    hub = MagicMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.set_cache = AsyncMock()
    hub.schedule_task = AsyncMock()
    hub.publish = AsyncMock()
    hub.is_running = MagicMock(return_value=True)

    with patch.object(PresenceModule, "__init__", lambda self, *a, **kw: None):
        m = PresenceModule.__new__(PresenceModule)
        m.hub = hub
        m.ha_url = "http://test-host:8123"
        m.ha_token = "test-token"
        m.mqtt_host = ""
        m.mqtt_port = 1883
        m.mqtt_user = ""
        m.mqtt_password = ""
        m._config_camera_rooms = None
        m.camera_rooms = {}
        m._person_states = {}
        m._room_signals = {}
        m._identified_persons = {}
        m._recent_detections = []
        m._max_recent_detections = 20
        m._frigate_url = "http://127.0.0.1:5000"
        m._frigate_camera_names = set()
        m._face_config = None
        m._labeled_faces = {}
        m._face_config_fetched = False
        m._mqtt_client = None
        m._mqtt_connected = False
        m._http_session = None
        m.logger = logging.getLogger("test_presence")
        # Import BayesianOccupancy for the occupancy instance
        from aria.engine.analysis.occupancy import BayesianOccupancy

        m._occupancy = BayesianOccupancy()
        return m


@pytest.mark.asyncio
async def test_presence_creates_session_on_init(presence_module):
    """#58: initialize() creates a shared aiohttp.ClientSession."""
    with (
        patch.object(presence_module, "_seed_presence_from_ha", new_callable=AsyncMock),
        patch.object(presence_module, "_refresh_camera_rooms", new_callable=AsyncMock),
        patch.object(presence_module, "_fetch_face_config", new_callable=AsyncMock),
    ):
        await presence_module.initialize()

    assert presence_module._http_session is not None
    # Clean up
    await presence_module._http_session.close()


@pytest.mark.asyncio
async def test_presence_closes_session_on_shutdown(presence_module):
    """#58: shutdown() closes the shared aiohttp session."""
    mock_session = MagicMock()
    mock_session.close = AsyncMock()
    presence_module._http_session = mock_session

    await presence_module.shutdown()

    # Session should have been closed and set to None
    mock_session.close.assert_called_once()
    assert presence_module._http_session is None


def test_presence_no_per_call_session():
    """#58: Presence methods use self._http_session, not aiohttp.ClientSession()."""
    import inspect

    from aria.modules.presence import PresenceModule

    # Check that _fetch_face_config, get_frigate_thumbnail, get_frigate_snapshot
    # no longer create new ClientSession instances
    for method_name in ["_fetch_face_config", "get_frigate_thumbnail", "get_frigate_snapshot"]:
        source = inspect.getsource(getattr(PresenceModule, method_name))
        assert "aiohttp.ClientSession()" not in source, (
            f"{method_name} should not create new aiohttp.ClientSession — use self._http_session"
        )
