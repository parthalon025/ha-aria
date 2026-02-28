"""Tests for FacePipeline — live event processing."""

import pytest

pytest.importorskip("cv2", reason="faces optional deps ([faces] extra) not installed")

from unittest.mock import MagicMock

import numpy as np
import pytest

from aria.faces.pipeline import FacePipeline
from aria.faces.store import FaceEmbeddingStore


@pytest.fixture
def store(tmp_path):
    s = FaceEmbeddingStore(str(tmp_path / "faces.db"))
    s.initialize()
    # Seed with one known person — 10 verified embeddings (unique event_id per row)
    vec = np.ones(512, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    for i in range(10):
        s.add_embedding(
            person_name="justin",
            embedding=vec.copy(),
            event_id=f"evt_{i}",
            image_path=f"/tmp/x_{i}.jpg",
            confidence=0.95,
            source="bootstrap",
            verified=True,
        )
    return s


@pytest.fixture
def pipeline(store):
    return FacePipeline(store=store, frigate_url="http://localhost:5000")


class TestFacePipelineMatching:
    def test_high_confidence_returns_auto_label(self, pipeline):
        """Above threshold returns auto-label result."""
        vec = np.ones(512, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        mock_extractor = MagicMock()
        mock_extractor.extract_embedding.return_value = vec
        mock_extractor.find_best_match.return_value = [{"person_name": "justin", "confidence": 0.92}]
        pipeline.extractor = mock_extractor

        result = pipeline.process_embedding(vec, event_id="evt-auto")
        assert result["action"] == "auto_label"
        assert result["person_name"] == "justin"
        assert result["confidence"] >= 0.50

    def test_low_confidence_queues_for_review(self, pipeline):
        """Below threshold adds to review queue."""
        vec = np.ones(512, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        mock_extractor = MagicMock()
        mock_extractor.find_best_match.return_value = [{"person_name": "justin", "confidence": 0.30}]
        pipeline.extractor = mock_extractor

        result = pipeline.process_embedding(vec, event_id="evt-queue", image_path="/tmp/f.jpg")
        assert result["action"] == "queued"
        assert pipeline.store.get_queue_depth() == 1

    def test_no_match_queues_as_unknown(self, pipeline):
        """Empty match list queues as unknown."""
        vec = np.random.rand(512).astype(np.float32)
        mock_extractor = MagicMock()
        mock_extractor.find_best_match.return_value = []
        pipeline.extractor = mock_extractor

        result = pipeline.process_embedding(vec, event_id="evt-unknown", image_path="/tmp/f.jpg")
        assert result["action"] == "queued"

    def test_threshold_uses_only_verified_count(self, pipeline):
        """Threshold count uses only verified embeddings — not auto-labels."""
        # Get the count with 10 verified embeddings (seeded in fixture)
        named = pipeline.store.get_all_named_embeddings()
        verified_count = sum(1 for e in named if e.get("verified") and e["person_name"] == "justin")
        threshold = pipeline.store.get_threshold_for_person("justin", labeled_count=verified_count)
        # 10 verified embeddings → threshold = max(0.50, 0.85 - 0.005*10) = 0.80
        assert abs(threshold - 0.80) < 0.001
