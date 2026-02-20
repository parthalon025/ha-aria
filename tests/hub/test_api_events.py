"""Tests for EventStore HTTP API endpoints."""

import asyncio

import pytest

from aria.shared.event_store import EventStore


@pytest.fixture
def event_store_hub(api_hub, tmp_path):
    """api_hub with a real EventStore attached."""
    store = EventStore(str(tmp_path / "events.db"))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(store.initialize())
    loop.run_until_complete(
        store.insert_event("2026-02-20T10:00:00", "light.bedroom", "light", "off", "on", area_id="bedroom")
    )
    loop.run_until_complete(
        store.insert_event("2026-02-20T10:05:00", "light.kitchen", "light", "off", "on", area_id="kitchen")
    )
    api_hub.event_store = store
    yield api_hub
    loop.run_until_complete(store.close())


def test_get_state_events(event_store_hub, api_client):
    """GET /api/state-events returns events from EventStore."""
    response = api_client.get("/api/state-events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 2
    assert data["count"] == 2


def test_get_state_events_filter_by_area(event_store_hub, api_client):
    """GET /api/state-events?area_id= filters by area."""
    response = api_client.get("/api/state-events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59&area_id=bedroom")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["entity_id"] == "light.bedroom"


def test_get_state_events_filter_by_entity(event_store_hub, api_client):
    """GET /api/state-events?entity_id= filters by entity."""
    response = api_client.get(
        "/api/state-events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59&entity_id=light.kitchen"
    )
    assert response.status_code == 200
    assert len(response.json()["events"]) == 1


def test_get_state_events_filter_by_domain(event_store_hub, api_client):
    """GET /api/state-events?domain= filters by domain."""
    response = api_client.get("/api/state-events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59&domain=light")
    assert response.status_code == 200
    assert len(response.json()["events"]) == 2


def test_get_state_events_stats(event_store_hub, api_client):
    """GET /api/state-events/stats returns total count."""
    response = api_client.get("/api/state-events/stats")
    assert response.status_code == 200
    assert response.json()["total_events"] == 2


def test_get_state_events_no_store(api_hub, api_client):
    """GET /api/state-events returns 503 when EventStore not available."""
    api_hub.event_store = None
    response = api_client.get("/api/state-events?start=2026-02-20T00:00:00&end=2026-02-20T23:59:59")
    assert response.status_code == 503
