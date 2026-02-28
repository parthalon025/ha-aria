"""Tests for /api/faces/* endpoints."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aria.faces.store import FaceEmbeddingStore
from aria.hub.api import create_api
from aria.hub.core import IntelligenceHub


@pytest.fixture
def faces_store(tmp_path):
    s = FaceEmbeddingStore(str(tmp_path / "faces.db"))
    s.initialize()
    return s


@pytest.fixture
def api_hub(faces_store):
    mock_hub = MagicMock(spec=IntelligenceHub)
    mock_hub.cache = MagicMock()
    mock_hub.modules = {}
    mock_hub.module_status = {}
    mock_hub.subscribers = {}
    mock_hub.subscribe = MagicMock()
    mock_hub._request_count = 0
    mock_hub._audit_logger = None
    mock_hub.set_cache = AsyncMock()
    mock_hub.get_uptime_seconds = MagicMock(return_value=0)
    mock_hub.faces_store = faces_store
    return mock_hub


_TEST_API_KEY = "test-aria-key"


@pytest.fixture
def api_client(api_hub):
    import aria.hub.api as _api_module

    original_key = _api_module._ARIA_API_KEY
    _api_module._ARIA_API_KEY = _TEST_API_KEY
    try:
        app = create_api(api_hub)
        yield TestClient(app, headers={"X-API-Key": _TEST_API_KEY})
    finally:
        _api_module._ARIA_API_KEY = original_key


class TestFacesQueueAPI:
    def test_get_queue_empty(self, api_client):
        response = api_client.get("/api/faces/queue")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["depth"] == 0

    def test_label_queued_face(self, api_client, api_hub):
        """POST /api/faces/label marks item reviewed and saves embedding."""
        vec = np.random.rand(512).astype(np.float32)
        qid = api_hub.faces_store.add_to_review_queue("evt-001", "/tmp/face.jpg", vec, [], 0.8)
        response = api_client.post(
            "/api/faces/label",
            json={
                "queue_id": qid,
                "person_name": "justin",
            },
        )
        assert response.status_code == 200
        assert api_hub.faces_store.get_queue_depth() == 0
        embeddings = api_hub.faces_store.get_embeddings_for_person("justin")
        assert len(embeddings) == 1


class TestFacesPeopleAPI:
    def test_get_people_empty(self, api_client):
        response = api_client.get("/api/faces/people")
        assert response.status_code == 200
        assert response.json()["people"] == []

    def test_get_people_with_data(self, api_client, api_hub):
        for i in range(3):
            api_hub.faces_store.add_embedding(
                "justin", np.random.rand(512).astype(np.float32), f"evt-{i}", f"/tmp/x{i}.jpg", 0.9, "bootstrap", True
            )
        response = api_client.get("/api/faces/people")
        assert response.status_code == 200
        data = response.json()
        assert len(data["people"]) == 1
        assert data["people"][0]["person_name"] == "justin"
        assert data["people"][0]["count"] == 3


class TestFacesStatsAPI:
    def test_get_stats(self, api_client):
        response = api_client.get("/api/faces/stats")
        assert response.status_code == 200
        data = response.json()
        assert "queue_depth" in data
        assert "known_people" in data
        assert "auto_label_rate" in data
        assert isinstance(data["auto_label_rate"], float)
        # Fields added in audit round — verify presence so they don't regress silently
        assert "last_face_processed_at" in data
        assert "face_pipeline_errors" in data

    def test_auto_label_rate_zero_when_empty(self, api_client):
        """Rate is 0.0 when there are no events."""
        response = api_client.get("/api/faces/stats")
        assert response.json()["auto_label_rate"] == 0.0

    def test_image_endpoint_404_for_missing(self, api_client):
        """Returns 404 for non-existent queue item."""
        response = api_client.get("/api/faces/image/999")
        assert response.status_code == 404


class TestDismissQueueItemAPI:
    def test_dismiss_success(self, api_client, api_hub):
        """POST /api/faces/queue/{id}/dismiss returns dismissed id for existing pending item."""
        vec = np.random.rand(512).astype(np.float32)
        qid = api_hub.faces_store.add_to_review_queue("evt-dismiss", "/tmp/face.jpg", vec, [], 0.5)
        response = api_client.post(f"/api/faces/queue/{qid}/dismiss")
        assert response.status_code == 200
        assert response.json() == {"dismissed": qid}
        # Queue depth should now be 0 (item dismissed)
        assert api_hub.faces_store.get_queue_depth() == 0

    def test_dismiss_404_not_found(self, api_client):
        """POST /api/faces/queue/{id}/dismiss returns 404 for nonexistent item."""
        response = api_client.post("/api/faces/queue/99999/dismiss")
        assert response.status_code == 404


class TestPersonSamplesAPI:
    def test_get_samples_known_person(self, api_client, api_hub):
        """GET /api/faces/people/{name}/samples returns samples for a known person."""
        for i in range(2):
            api_hub.faces_store.add_embedding(
                "alice",
                np.random.rand(512).astype(np.float32),
                f"evt-alice-{i}",
                f"/tmp/alice{i}.jpg",
                0.9,
                "bootstrap",
                True,
            )
        response = api_client.get("/api/faces/people/alice/samples")
        assert response.status_code == 200
        data = response.json()
        assert data["person_name"] == "alice"
        assert len(data["samples"]) == 2
        # embedding blob must be stripped
        for s in data["samples"]:
            assert "embedding" not in s

    def test_get_samples_unknown_person_empty(self, api_client):
        """GET /api/faces/people/{name}/samples returns empty list for unknown person."""
        response = api_client.get("/api/faces/people/nobody/samples")
        assert response.status_code == 200
        data = response.json()
        assert data["person_name"] == "nobody"
        assert data["samples"] == []


class TestDeletePersonSampleAPI:
    def test_delete_sample_success(self, api_client, api_hub):
        """DELETE /api/faces/people/{name}/samples/{id} returns deleted id."""
        eid = api_hub.faces_store.add_embedding(
            "bob",
            np.random.rand(512).astype(np.float32),
            "evt-bob-1",
            "/tmp/bob.jpg",
            0.9,
            "bootstrap",
            True,
        )
        response = api_client.delete(f"/api/faces/people/bob/samples/{eid}")
        assert response.status_code == 200
        assert response.json() == {"deleted": eid}
        # Verify gone
        assert api_hub.faces_store.get_embeddings_for_person("bob") == []

    def test_delete_sample_404_not_found(self, api_client):
        """DELETE /api/faces/people/{name}/samples/{id} returns 404 for nonexistent sample."""
        response = api_client.delete("/api/faces/people/bob/samples/99999")
        assert response.status_code == 404


class TestRenamePersonAPI:
    def test_rename_success(self, api_client, api_hub):
        """POST /api/faces/people/{name}/rename returns renamed count and removes old name."""
        for i in range(2):
            api_hub.faces_store.add_embedding(
                "carol",
                np.random.rand(512).astype(np.float32),
                f"evt-carol-{i}",
                f"/tmp/carol{i}.jpg",
                0.9,
                "bootstrap",
                True,
            )
        response = api_client.post("/api/faces/people/carol/rename", json={"new_name": "caroline"})
        assert response.status_code == 200
        data = response.json()
        assert data["renamed"] == 2
        assert data["old_name"] == "carol"
        assert data["new_name"] == "caroline"
        # Old name should be gone
        people = [p["person_name"] for p in api_hub.faces_store.get_known_people()]
        assert "carol" not in people
        assert "caroline" in people


class TestDeletePersonAPI:
    def test_delete_person_success(self, api_client, api_hub):
        """DELETE /api/faces/people/{name} returns deleted count and person no longer listed."""
        for i in range(3):
            api_hub.faces_store.add_embedding(
                "dave",
                np.random.rand(512).astype(np.float32),
                f"evt-dave-{i}",
                f"/tmp/dave{i}.jpg",
                0.9,
                "bootstrap",
                True,
            )
        response = api_client.delete("/api/faces/people/dave")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == 3
        assert data["person_name"] == "dave"
        # Person should be gone
        people = [p["person_name"] for p in api_hub.faces_store.get_known_people()]
        assert "dave" not in people


class TestQueuePaginationAPI:
    def test_queue_limit_respected(self, api_client, api_hub):
        """GET /api/faces/queue?limit=3 returns at most 3 items from a larger queue."""
        for i in range(6):
            api_hub.faces_store.add_to_review_queue(
                f"evt-pg-{i}", f"/tmp/pg{i}.jpg", np.random.rand(512).astype(np.float32), [], 0.5
            )
        response = api_client.get("/api/faces/queue?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        # depth reflects total pending, not just the page
        assert data["depth"] == 6

    def test_label_face_beyond_queue_limit(self, api_client, api_hub):
        """POST /api/faces/label works for items that would be beyond the old limit=1000.

        This regression test ensures the direct-by-id lookup (get_pending_queue_item)
        is used rather than a limit-based scan. We add 3 items and label the last one.
        """
        ids = []
        for i in range(3):
            qid = api_hub.faces_store.add_to_review_queue(
                f"evt-deep-{i}",
                f"/tmp/deep{i}.jpg",
                np.random.rand(512).astype(np.float32),
                [],
                0.6,
            )
            ids.append(qid)
        # Label the last item (highest ID) — with the old limit=N scan this would fail
        # if N < number of items; with the direct lookup it always works
        response = api_client.post("/api/faces/label", json={"queue_id": ids[-1], "person_name": "frank"})
        assert response.status_code == 200
        assert api_hub.faces_store.get_embeddings_for_person("frank") != []


class TestDeleteSamplePersonConstraintAPI:
    def test_delete_sample_wrong_person_returns_404(self, api_client, api_hub):
        """DELETE /api/faces/people/alice/samples/{bob_sample_id} returns 404."""
        bob_id = api_hub.faces_store.add_embedding(
            "bob",
            np.random.rand(512).astype(np.float32),
            "evt-bob-x",
            "/tmp/bob_x.jpg",
            0.9,
            "bootstrap",
            True,
        )
        # Try to delete bob's sample via alice's URL
        response = api_client.delete(f"/api/faces/people/alice/samples/{bob_id}")
        assert response.status_code == 404
        # Bob's sample must still exist
        assert api_hub.faces_store.get_embeddings_for_person("bob") != []


class TestExportAPI:
    def test_export_includes_people_and_queue(self, api_client, api_hub):
        """GET /api/faces/export includes labeled people and pending queue items."""
        # Add a labeled embedding
        api_hub.faces_store.add_embedding(
            "frank",
            np.random.rand(512).astype(np.float32),
            "evt-frank-1",
            "/tmp/frank.jpg",
            0.9,
            "manual",
            True,
        )
        # Add a pending queue item with an image path (no disk file needed for export)
        vec = np.random.rand(512).astype(np.float32)
        api_hub.faces_store.add_to_review_queue("evt-q-1", "/tmp/unknown.jpg", vec, [], 0.6)

        response = api_client.get("/api/faces/export")
        assert response.status_code == 200
        data = response.json()
        assert "exported_at" in data
        assert any(p["person_name"] == "frank" for p in data["people"])
        assert "pending_queue" in data
        assert "pending_queue_count" in data
        assert data["pending_queue_count"] >= 1
        # embedding blobs must not appear in queue items
        for q in data["pending_queue"]:
            assert "embedding" not in q

    def test_export_empty_store(self, api_client):
        """GET /api/faces/export with no data returns empty people and queue."""
        response = api_client.get("/api/faces/export")
        assert response.status_code == 200
        data = response.json()
        assert data["people"] == []
        assert data["pending_queue"] == []
        assert data["pending_queue_count"] == 0


class TestEmbeddingImageAPI:
    def test_embedding_image_404_nonexistent(self, api_client):
        """GET /api/faces/embedding-image/{id} returns 404 for nonexistent embedding."""
        response = api_client.get("/api/faces/embedding-image/99999")
        assert response.status_code == 404

    def test_embedding_image_404_no_image_path(self, api_client, api_hub):
        """GET /api/faces/embedding-image/{id} returns 404 when image_path is missing on disk."""
        eid = api_hub.faces_store.add_embedding(
            "eve",
            np.random.rand(512).astype(np.float32),
            "evt-eve-1",
            # Path that does not exist on disk
            "/nonexistent/path/eve.jpg",
            0.9,
            "bootstrap",
            True,
        )
        response = api_client.get(f"/api/faces/embedding-image/{eid}")
        assert response.status_code == 404
