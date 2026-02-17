"""Tests for discovery lifecycle merge logic (active/stale/archived)."""

import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


# ------------------------------------------------------------------
# _merge_with_lifecycle
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_entities_get_lifecycle_metadata(module, mock_hub):
    """First-time discovered entities get full lifecycle metadata."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "sensor.temperature": {"state": "22.5", "domain": "sensor"},
        "light.living_room": {"state": "on", "domain": "light"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    for entity_id in ["sensor.temperature", "light.living_room"]:
        assert entity_id in merged
        lc = merged[entity_id]["_lifecycle"]
        assert lc["status"] == "active"
        assert lc["first_discovered"] == now.isoformat()
        assert lc["last_seen_in_discovery"] == now.isoformat()
        assert lc["stale_since"] is None
        assert lc["archived_at"] is None


@pytest.mark.asyncio
async def test_existing_entities_preserve_first_discovered(module, mock_hub):
    """Re-discovered entities keep their original first_discovered timestamp."""
    original_time = "2026-01-01T00:00:00"
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.temperature": {
                    "state": "20.0",
                    "domain": "sensor",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": original_time,
                        "last_seen_in_discovery": "2026-02-16T12:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "sensor.temperature": {"state": "22.5", "domain": "sensor"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    lc = merged["sensor.temperature"]["_lifecycle"]
    assert lc["first_discovered"] == original_time  # preserved
    assert lc["last_seen_in_discovery"] == now.isoformat()  # updated
    assert lc["status"] == "active"
    assert lc["stale_since"] is None


@pytest.mark.asyncio
async def test_existing_entity_fields_updated(module, mock_hub):
    """Re-discovered entities get their non-lifecycle fields updated."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.temperature": {
                    "state": "20.0",
                    "domain": "sensor",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T12:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "sensor.temperature": {"state": "22.5", "domain": "sensor", "unit": "°C"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    assert merged["sensor.temperature"]["state"] == "22.5"
    assert merged["sensor.temperature"]["unit"] == "°C"


@pytest.mark.asyncio
async def test_missing_entities_marked_stale(module, mock_hub):
    """Entities in old cache but not in new discovery become stale."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.temperature": {
                    "state": "22.5",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T12:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
                "sensor.humidity": {
                    "state": "55",
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-01-15T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T12:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    # Only temperature in new discovery — humidity is missing
    new_items = {
        "sensor.temperature": {"state": "23.0"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    # Temperature stays active
    assert merged["sensor.temperature"]["_lifecycle"]["status"] == "active"

    # Humidity becomes stale
    lc = merged["sensor.humidity"]["_lifecycle"]
    assert lc["status"] == "stale"
    assert lc["stale_since"] == now.isoformat()
    assert lc["first_discovered"] == "2026-01-15T00:00:00"  # preserved


@pytest.mark.asyncio
async def test_stale_entities_restored_to_active(module, mock_hub):
    """Stale entities restored to active when rediscovered."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.temperature": {
                    "state": "20.0",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-10T00:00:00",
                        "stale_since": "2026-02-15T00:00:00",
                        "archived_at": None,
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "sensor.temperature": {"state": "22.5"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    lc = merged["sensor.temperature"]["_lifecycle"]
    assert lc["status"] == "active"
    assert lc["first_discovered"] == "2026-01-01T00:00:00"  # preserved
    assert lc["last_seen_in_discovery"] == now.isoformat()
    assert lc["stale_since"] is None  # cleared
    assert lc["archived_at"] is None


@pytest.mark.asyncio
async def test_archived_entities_restored_to_active(module, mock_hub):
    """Archived entities restored to active on rediscovery, first_discovered preserved."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.temperature": {
                    "state": "20.0",
                    "_lifecycle": {
                        "status": "archived",
                        "first_discovered": "2025-12-01T00:00:00",
                        "last_seen_in_discovery": "2026-01-01T00:00:00",
                        "stale_since": "2026-01-05T00:00:00",
                        "archived_at": "2026-01-08T00:00:00",
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "sensor.temperature": {"state": "22.5"},
    }

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    lc = merged["sensor.temperature"]["_lifecycle"]
    assert lc["status"] == "active"
    assert lc["first_discovered"] == "2025-12-01T00:00:00"  # preserved
    assert lc["last_seen_in_discovery"] == now.isoformat()
    assert lc["stale_since"] is None
    assert lc["archived_at"] is None


@pytest.mark.asyncio
async def test_already_stale_keeps_original_stale_since(module, mock_hub):
    """Already-stale entities don't get stale_since re-stamped."""
    original_stale = "2026-02-10T00:00:00"
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.removed": {
                    "state": "unavailable",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-09T00:00:00",
                        "stale_since": original_stale,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {}  # Nothing rediscovered

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    lc = merged["sensor.removed"]["_lifecycle"]
    assert lc["status"] == "stale"
    assert lc["stale_since"] == original_stale  # NOT re-stamped


@pytest.mark.asyncio
async def test_already_archived_keeps_status(module, mock_hub):
    """Already-archived entities keep archived status when not rediscovered."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.gone": {
                    "state": "unavailable",
                    "_lifecycle": {
                        "status": "archived",
                        "first_discovered": "2025-12-01T00:00:00",
                        "last_seen_in_discovery": "2025-12-15T00:00:00",
                        "stale_since": "2025-12-20T00:00:00",
                        "archived_at": "2025-12-23T00:00:00",
                    },
                },
            }
        }
    )

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {}

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    lc = merged["sensor.gone"]["_lifecycle"]
    assert lc["status"] == "archived"
    assert lc["archived_at"] == "2025-12-23T00:00:00"  # unchanged


@pytest.mark.asyncio
async def test_merge_applies_to_devices(module, mock_hub):
    """Lifecycle merge works for devices, not just entities."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "device_001": {"name": "Living Room Light", "manufacturer": "Philips"},
    }

    merged = await module._merge_with_lifecycle("devices", new_items, now)

    assert "device_001" in merged
    lc = merged["device_001"]["_lifecycle"]
    assert lc["status"] == "active"
    assert lc["first_discovered"] == now.isoformat()


@pytest.mark.asyncio
async def test_merge_applies_to_areas(module, mock_hub):
    """Lifecycle merge works for areas."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {
        "area_living_room": {"name": "Living Room"},
    }

    merged = await module._merge_with_lifecycle("areas", new_items, now)

    assert "area_living_room" in merged
    lc = merged["area_living_room"]["_lifecycle"]
    assert lc["status"] == "active"
    assert lc["first_discovered"] == now.isoformat()


@pytest.mark.asyncio
async def test_merge_with_no_existing_cache(module, mock_hub):
    """When get_cache returns None, all items are new."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {"a": {"val": 1}, "b": {"val": 2}}

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    assert len(merged) == 2
    for key in ["a", "b"]:
        assert merged[key]["_lifecycle"]["status"] == "active"


@pytest.mark.asyncio
async def test_merge_with_empty_data_cache(module, mock_hub):
    """When get_cache returns entry with empty data, all items are new."""
    mock_hub.get_cache = AsyncMock(return_value={"data": {}})

    now = datetime(2026, 2, 17, 12, 0, 0)
    new_items = {"a": {"val": 1}}

    merged = await module._merge_with_lifecycle("entities", new_items, now)

    assert len(merged) == 1
    assert merged["a"]["_lifecycle"]["status"] == "active"


# ------------------------------------------------------------------
# _archive_expired_entities
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_stale_past_ttl(module, mock_hub):
    """Stale entities past TTL get archived."""
    stale_since = (datetime(2026, 2, 17, 12, 0, 0) - timedelta(hours=100)).isoformat()
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.gone": {
                    "state": "unavailable",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-10T00:00:00",
                        "stale_since": stale_since,
                        "archived_at": None,
                    },
                },
                "sensor.recent_stale": {
                    "state": "unavailable",
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T00:00:00",
                        "stale_since": datetime(2026, 2, 17, 11, 0, 0).isoformat(),
                        "archived_at": None,
                    },
                },
            }
        }
    )

    mock_hub.cache.get_config_value = AsyncMock(return_value="72")

    await module._archive_expired_entities("entities")

    # Should have written cache — one entity archived
    mock_hub.set_cache.assert_called_once()
    call_args = mock_hub.set_cache.call_args[0]
    assert call_args[0] == "entities"
    data = call_args[1]

    assert data["sensor.gone"]["_lifecycle"]["status"] == "archived"
    assert data["sensor.gone"]["_lifecycle"]["archived_at"] is not None

    # Recent stale should still be stale
    assert data["sensor.recent_stale"]["_lifecycle"]["status"] == "stale"


@pytest.mark.asyncio
async def test_archive_noop_when_nothing_expired(module, mock_hub):
    """No cache write when nothing needs archiving."""
    recent_stale = datetime(2026, 2, 17, 11, 0, 0).isoformat()
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.recent_stale": {
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T00:00:00",
                        "stale_since": recent_stale,
                        "archived_at": None,
                    },
                },
                "sensor.active": {
                    "_lifecycle": {
                        "status": "active",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-17T00:00:00",
                        "stale_since": None,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    mock_hub.cache.get_config_value = AsyncMock(return_value="72")

    await module._archive_expired_entities("entities")

    # No cache write
    mock_hub.set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_archive_no_cache_entry(module, mock_hub):
    """No-op when cache entry doesn't exist."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    await module._archive_expired_entities("entities")

    mock_hub.set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_archive_uses_config_ttl(module, mock_hub):
    """Archive respects custom TTL from config."""
    # 10-hour TTL, entity stale for 11 hours
    stale_since = (datetime(2026, 2, 17, 12, 0, 0) - timedelta(hours=11)).isoformat()
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "sensor.gone": {
                    "_lifecycle": {
                        "status": "stale",
                        "first_discovered": "2026-01-01T00:00:00",
                        "last_seen_in_discovery": "2026-02-16T00:00:00",
                        "stale_since": stale_since,
                        "archived_at": None,
                    },
                },
            }
        }
    )

    mock_hub.cache.get_config_value = AsyncMock(return_value="10")

    await module._archive_expired_entities("entities")

    mock_hub.set_cache.assert_called_once()
    data = mock_hub.set_cache.call_args[0][1]
    assert data["sensor.gone"]["_lifecycle"]["status"] == "archived"


# ------------------------------------------------------------------
# _store_discovery_results integration
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_calls_merge_for_entities_devices_areas(module, mock_hub):
    """_store_discovery_results uses lifecycle merge for entities/devices/areas."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    capabilities_data = {
        "entities": {"sensor.temp": {"state": "22"}},
        "devices": {"dev_1": {"name": "Lamp"}},
        "areas": {"area_1": {"name": "Kitchen"}},
        "capabilities": {},
        "entity_count": 1,
        "device_count": 1,
        "area_count": 1,
        "timestamp": "2026-02-17T12:00:00",
        "ha_version": "2024.1.0",
    }

    await module._store_discovery_results(capabilities_data)

    # Check all three got lifecycle metadata
    cache_calls = {call[0][0]: call[0][1] for call in mock_hub.set_cache.call_args_list}

    for key in ["entities", "devices", "areas"]:
        assert key in cache_calls, f"{key} not in cache calls"
        data = cache_calls[key]
        for item_id, item_data in data.items():
            assert "_lifecycle" in item_data, f"No lifecycle in {key}/{item_id}"
            assert item_data["_lifecycle"]["status"] == "active"


@pytest.mark.asyncio
async def test_store_preserves_capabilities_organic_merge(module, mock_hub):
    """Capabilities merge logic (organic preservation) stays unchanged."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "seed_cap": {"entities": ["sensor.a"], "source": "seed"},
                "organic_cap": {"entities": ["sensor.b"], "source": "organic", "usefulness": 80},
            }
        }
    )

    capabilities_data = {
        "entities": {},
        "devices": {},
        "areas": {},
        "capabilities": {
            "seed_cap": {"entities": ["sensor.a_updated"]},
            "new_cap": {"entities": ["sensor.c"]},
        },
    }

    await module._store_discovery_results(capabilities_data)

    cap_calls = [c for c in mock_hub.set_cache.call_args_list if c[0][0] == "capabilities"]
    assert len(cap_calls) == 1
    caps = cap_calls[0][0][1]

    assert "seed_cap" in caps
    assert "new_cap" in caps
    assert "organic_cap" in caps
    assert caps["organic_cap"]["source"] == "organic"


# ============================================================================
# Archive Scheduling
# ============================================================================


class TestArchiveScheduling:
    """Archive check should run periodically via initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_schedules_archive_check(self, module, mock_hub):
        """initialize() should schedule periodic archive expiry checks."""
        module.run_discovery = AsyncMock()
        await module.initialize()

        task_ids = [call.kwargs.get("task_id", "") for call in mock_hub.schedule_task.call_args_list]
        assert "discovery_archive_check" in task_ids


# ============================================================================
# Discovery Events
# ============================================================================


class TestDiscoveryEvents:
    """Discovery should publish events for consumers."""

    @pytest.mark.asyncio
    async def test_publishes_discovery_complete(self, module, mock_hub):
        """_store_discovery_results should publish discovery_complete event."""
        results = {
            "entities": {"light.x": {"area_id": "room"}},
            "devices": {},
            "areas": {},
            "capabilities": {},
        }
        await module._store_discovery_results(results)

        event_types = [c[0][0] for c in mock_hub.publish.call_args_list]
        assert "discovery_complete" in event_types
