"""Tests that seed discovery preserves organic capabilities in the cache."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aria.modules.discovery import DiscoveryModule


@pytest.fixture
def mock_hub():
    hub = MagicMock()
    hub.set_cache = AsyncMock()
    hub.get_cache = AsyncMock(return_value=None)
    hub.schedule_task = AsyncMock()
    hub.cache = MagicMock()
    hub.cache.get_config_value = AsyncMock(return_value="")
    return hub


@pytest.mark.asyncio
async def test_store_preserves_organic_capabilities(mock_hub):
    """Seed discovery should not overwrite organic capabilities."""
    # Existing cache has seed + organic capabilities
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "power_monitoring": {"entities": ["sensor.power"], "source": "seed"},
                "home_pressure_0": {
                    "entities": ["sensor.pressure_1", "sensor.pressure_2"],
                    "source": "organic",
                    "usefulness": 72,
                    "status": "promoted",
                },
                "motion_cluster_3": {
                    "entities": ["binary_sensor.motion_1"],
                    "source": "organic",
                    "usefulness": 45,
                    "status": "candidate",
                },
            }
        }
    )

    module = DiscoveryModule(mock_hub, "http://localhost:8123", "test-token")

    # Seed discovery returns only seed capabilities (no organic ones)
    seed_results = {
        "entities": {},
        "devices": {},
        "areas": {},
        "capabilities": {
            "power_monitoring": {"entities": ["sensor.power"]},
            "lighting": {"entities": ["light.lamp1"]},
        },
    }

    await module._store_discovery_results(seed_results)

    # Find the set_cache call for capabilities
    cap_calls = [call for call in mock_hub.set_cache.call_args_list if call[0][0] == "capabilities"]
    assert len(cap_calls) == 1

    stored_caps = cap_calls[0][0][1]

    # Seed capabilities present
    assert "power_monitoring" in stored_caps
    assert "lighting" in stored_caps

    # Organic capabilities preserved
    assert "home_pressure_0" in stored_caps
    assert stored_caps["home_pressure_0"]["source"] == "organic"
    assert stored_caps["home_pressure_0"]["usefulness"] == 72

    assert "motion_cluster_3" in stored_caps
    assert stored_caps["motion_cluster_3"]["source"] == "organic"


@pytest.mark.asyncio
async def test_store_no_existing_cache(mock_hub):
    """When no existing cache, seed capabilities are written as-is."""
    mock_hub.get_cache = AsyncMock(return_value=None)

    module = DiscoveryModule(mock_hub, "http://localhost:8123", "test-token")

    seed_results = {
        "entities": {},
        "devices": {},
        "areas": {},
        "capabilities": {
            "power_monitoring": {"entities": ["sensor.power"]},
        },
    }

    await module._store_discovery_results(seed_results)

    cap_calls = [call for call in mock_hub.set_cache.call_args_list if call[0][0] == "capabilities"]
    assert len(cap_calls) == 1
    stored_caps = cap_calls[0][0][1]
    assert "power_monitoring" in stored_caps
    assert len(stored_caps) == 1


@pytest.mark.asyncio
async def test_seed_cap_overrides_organic_with_same_name(mock_hub):
    """If seed and organic share a name, seed wins (it's the source of truth)."""
    mock_hub.get_cache = AsyncMock(
        return_value={
            "data": {
                "power_monitoring": {
                    "entities": ["sensor.old"],
                    "source": "organic",
                    "usefulness": 50,
                },
            }
        }
    )

    module = DiscoveryModule(mock_hub, "http://localhost:8123", "test-token")

    seed_results = {
        "entities": {},
        "devices": {},
        "areas": {},
        "capabilities": {
            "power_monitoring": {"entities": ["sensor.power_new"]},
        },
    }

    await module._store_discovery_results(seed_results)

    cap_calls = [call for call in mock_hub.set_cache.call_args_list if call[0][0] == "capabilities"]
    stored_caps = cap_calls[0][0][1]
    # Seed version wins
    assert stored_caps["power_monitoring"]["entities"] == ["sensor.power_new"]
    assert stored_caps["power_monitoring"].get("source") != "organic"
