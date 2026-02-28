"""Hub validation — module init, cache, API, WebSocket, event bus."""

import pytest
from fastapi.testclient import TestClient

import aria.hub.api as _api_module
from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub
from tests.synthetic.pipeline import PipelineRunner
from tests.synthetic.simulator import HouseholdSimulator

_TEST_API_KEY = "test-aria-key"


@pytest.fixture(scope="module")
def hub_with_data(tmp_path_factory):
    """Hub seeded with engine pipeline output."""
    tmp = tmp_path_factory.mktemp("val_hub")
    cache_path = str(tmp / "hub.db")

    sim = HouseholdSimulator(scenario="stable_couple", days=21, seed=42)
    snapshots = sim.generate()
    runner = PipelineRunner(snapshots, data_dir=tmp / "engine")
    result = runner.run_full()

    return {
        "cache_path": cache_path,
        "engine_dir": tmp / "engine",
        "runner": runner,
        "result": result,
        "snapshots": snapshots,
    }


class TestHubInitialization:
    @pytest.mark.asyncio
    async def test_hub_initializes(self, tmp_path):
        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        assert hub.is_running()
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_hub_cache_read_write(self, tmp_path):
        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        version = await hub.set_cache("test_category", {"key": "value"})
        assert version >= 1
        data = await hub.get_cache("test_category")
        assert data is not None
        # get_cache returns wrapper: {category, data, version, last_updated, metadata}
        assert data["data"]["key"] == "value"
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_hub_module_registration(self, tmp_path):
        from aria.hub.core import Module

        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        module = Module("test_module", hub)
        hub.register_module(module)
        assert "test_module" in hub.modules
        await hub.shutdown()


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self, tmp_path):
        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        received = []

        async def handler(data):
            received.append(data)

        hub.subscribe("test_event", handler)
        await hub.publish("test_event", {"msg": "hello"})
        assert len(received) == 1
        assert received[0]["msg"] == "hello"
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_cache_update_fires_event(self, tmp_path):
        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        events = []

        async def handler(data):
            events.append(data)

        hub.subscribe("cache_updated", handler)
        await hub.set_cache("intelligence", {"test": True})
        assert len(events) >= 1
        assert events[0]["category"] == "intelligence"
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_sequential_events_in_order(self, tmp_path):
        cache_path = str(tmp_path / "test_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        order = []

        async def handler(data):
            order.append(data.get("seq"))

        hub.subscribe("test_seq", handler)
        for i in range(5):
            await hub.publish("test_seq", {"seq": i})
        assert order == [0, 1, 2, 3, 4]
        await hub.shutdown()


class TestCachePopulation:
    @pytest.mark.asyncio
    async def test_intelligence_cache_writable(self, hub_with_data):
        cache_path = hub_with_data["cache_path"]
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        result = hub_with_data["result"]
        intelligence_data = {
            "predictions": result["predictions"],
            "baselines": result["baselines"],
            "scores": result["scores"],
        }
        version = await hub.set_cache("intelligence", intelligence_data)
        assert version >= 1
        loaded = await hub.get_cache("intelligence")
        assert loaded is not None
        # get_cache returns wrapper with "data" key containing actual payload
        assert "predictions" in loaded["data"]
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_categories_populated(self, hub_with_data):
        cache_path = str(hub_with_data["engine_dir"] / "multi_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        categories = {
            "intelligence": {"predictions": hub_with_data["result"]["predictions"]},
            "capabilities": {"items": ["power", "occupancy", "weather"]},
            "entities": {"count": 46, "domains": ["light", "sensor", "switch"]},
        }
        for cat, data in categories.items():
            await hub.set_cache(cat, data)
        for cat in categories:
            loaded = await hub.get_cache(cat)
            assert loaded is not None, f"Cache category {cat} should be populated"
        await hub.shutdown()


class TestAPIEndpoints:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, tmp_path):
        cache_path = str(tmp_path / "api_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        app = create_api(hub)
        client = TestClient(app)
        # Health endpoint is at /health (not /api/health)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_cache_endpoint_returns_data(self, tmp_path):
        cache_path = str(tmp_path / "api_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        await hub.set_cache("intelligence", {"test_key": "test_value"})
        app = create_api(hub)
        original = _api_module._ARIA_API_KEY
        _api_module._ARIA_API_KEY = _TEST_API_KEY
        try:
            client = TestClient(app, headers={"X-API-Key": _TEST_API_KEY})
            response = client.get("/api/cache/intelligence")
            assert response.status_code == 200
        finally:
            _api_module._ARIA_API_KEY = original
        await hub.shutdown()

    @pytest.mark.asyncio
    async def test_cache_missing_returns_404(self, tmp_path):
        cache_path = str(tmp_path / "api_hub.db")
        hub = IntelligenceHub(cache_path)
        await hub.initialize()
        app = create_api(hub)
        original = _api_module._ARIA_API_KEY
        _api_module._ARIA_API_KEY = _TEST_API_KEY
        try:
            client = TestClient(app, headers={"X-API-Key": _TEST_API_KEY})
            # API returns 404 for nonexistent cache categories
            response = client.get("/api/cache/nonexistent")
            assert response.status_code == 404
        finally:
            _api_module._ARIA_API_KEY = original
        await hub.shutdown()
