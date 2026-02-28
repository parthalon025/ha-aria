"""Failing test for #305: presence aware/naive datetime mix causes TypeError.

_handle_frigate_event stores signals with datetime.now(UTC) (aware).
_flush_presence_state uses datetime.now() (naive).
_get_active_signals then does aware_ts >= naive_cutoff → TypeError.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from aria.modules.presence import PresenceModule


class _MockCacheManager:
    async def get_config_value(self, key: str, fallback: Any = None) -> Any:
        return fallback

    async def get_included_entity_ids(self):
        return set()

    async def get_all_curation(self):
        return []


class _MockHub:
    def __init__(self):
        self._cache: dict[str, Any] = {}
        self.cache = _MockCacheManager()

    def is_running(self) -> bool:
        return False

    def get_module(self, name: str):
        return None

    async def get_cache(self, category: str):
        return self._cache.get(category)

    async def set_cache(self, category: str, data: Any, metadata: dict | None = None):
        self._cache[category] = {"data": data}

    async def schedule_task(self, *args, **kwargs):
        pass

    async def publish(self, *args, **kwargs):
        pass

    def subscribe(self, *args, **kwargs):
        pass

    def unsubscribe(self, *args, **kwargs):
        pass


@pytest.fixture
def presence():
    hub = _MockHub()
    p = PresenceModule(hub=hub, ha_url="http://localhost:8123", ha_token="test")
    return p


@pytest.mark.asyncio
async def test_flush_does_not_raise_on_aware_signal_closes_305(presence):
    """Flush must not raise TypeError when room signals contain aware datetimes.

    _handle_frigate_event stores signals with datetime.now(UTC) (aware).
    If _flush_presence_state uses naive datetime.now(), comparison raises TypeError.
    """
    # Simulate what _handle_frigate_event does: store an aware timestamp signal
    aware_ts = datetime.now(tz=UTC)
    presence._room_signals["living_room"].append(("camera_person", 0.9, "test person", aware_ts))

    # Patch hub.get_cache to return None for unifi_client_state so cross-validation is skipped
    presence.hub._cache.clear()

    # This must NOT raise TypeError: can't compare offset-naive and offset-aware datetimes
    try:
        await presence._flush_presence_state()
    except TypeError as e:
        pytest.fail(f"_flush_presence_state raised TypeError with aware signal: {e}")
