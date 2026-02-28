"""Tests for FaceExtractor — InsightFace wrapper."""

import pytest

pytest.importorskip("cv2", reason="faces optional deps ([faces] extra) not installed")

from unittest.mock import MagicMock, patch

import numpy as np

from aria.faces.extractor import FaceExtractor


def _make_face(embedding=None, bbox=None):
    """Return a mock InsightFace Face object."""
    face = MagicMock()
    face.embedding = embedding if embedding is not None else np.random.rand(512).astype(np.float32)
    face.bbox = bbox if bbox is not None else [0.0, 0.0, 100.0, 100.0]
    return face


def _mock_app(faces_return=None, get_side_effect=None):
    """Return a mock FaceAnalysis app."""
    mock = MagicMock()
    if get_side_effect is not None:
        mock.get.side_effect = get_side_effect
    else:
        mock.get.return_value = faces_return if faces_return is not None else []
    return mock


class TestFaceExtractorEmbedding:
    def test_returns_normalized_512d_array(self):
        """extract_embedding returns L2-normalized 512-d float32 array."""
        extractor = FaceExtractor()
        raw = np.random.rand(512).astype(np.float32)
        face = _make_face(embedding=raw)

        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[face])),
            patch("aria.faces.extractor.cv2.imread", return_value=MagicMock()),
        ):
            result = extractor.extract_embedding("/tmp/fake.jpg")

        assert result is not None
        assert result.shape == (512,)
        assert result.dtype == np.float32
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5  # L2 normalized

    def test_returns_none_on_no_face(self):
        """Returns None when InsightFace finds no face."""
        extractor = FaceExtractor()
        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[])),
            patch("aria.faces.extractor.cv2.imread", return_value=MagicMock()),
        ):
            result = extractor.extract_embedding("/tmp/fake.jpg")
        assert result is None

    def test_returns_none_on_unreadable_image(self):
        """Returns None when cv2.imread fails (returns None)."""
        extractor = FaceExtractor()
        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app()),
            patch("aria.faces.extractor.cv2.imread", return_value=None),
        ):
            result = extractor.extract_embedding("/tmp/missing.jpg")
        assert result is None

    def test_returns_none_on_exception(self):
        """Returns None (not raise) on unexpected errors."""
        extractor = FaceExtractor()
        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app(get_side_effect=Exception("GPU OOM"))),
            patch("aria.faces.extractor.cv2.imread", return_value=MagicMock()),
        ):
            result = extractor.extract_embedding("/tmp/fake.jpg")
        assert result is None

    def test_returns_none_when_app_unavailable(self):
        """Returns None when InsightFace is not installed."""
        extractor = FaceExtractor()
        with patch("aria.faces.extractor._get_app", return_value=None):
            result = extractor.extract_embedding("/tmp/fake.jpg")
        assert result is None

    def test_picks_largest_face(self):
        """When multiple faces are detected, returns embedding of the largest."""
        extractor = FaceExtractor()
        small_embed = np.zeros(512, dtype=np.float32)
        small_embed[0] = 1.0
        large_embed = np.zeros(512, dtype=np.float32)
        large_embed[1] = 1.0

        small_face = _make_face(embedding=small_embed, bbox=[0.0, 0.0, 10.0, 10.0])  # area=100
        large_face = _make_face(embedding=large_embed, bbox=[0.0, 0.0, 100.0, 100.0])  # area=10000

        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[small_face, large_face])),
            patch("aria.faces.extractor.cv2.imread", return_value=MagicMock()),
        ):
            result = extractor.extract_embedding("/tmp/fake.jpg")

        # Result should be large_embed (normalized)
        assert result is not None
        assert result[1] > result[0]

    def test_cosine_similarity_identical(self):
        """cosine_similarity returns 1.0 for identical unit vectors."""
        extractor = FaceExtractor()
        v = np.random.rand(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert abs(extractor.cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """cosine_similarity returns 0.0 for orthogonal vectors."""
        extractor = FaceExtractor()
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert abs(extractor.cosine_similarity(a, b)) < 1e-5

    def test_find_best_match_sorts_by_confidence(self):
        """find_best_match returns sorted candidates above min threshold."""
        extractor = FaceExtractor()
        query = np.ones(512, dtype=np.float32)
        query /= np.linalg.norm(query)

        named = [
            {"person_name": "justin", "embedding": query.copy()},  # perfect match
            {"person_name": "carter", "embedding": -query.copy()},  # opposite
        ]
        candidates = extractor.find_best_match(query, named, min_threshold=-1.0)
        assert candidates[0]["person_name"] == "justin"
        assert candidates[0]["confidence"] > 0.99
        assert candidates[1]["confidence"] < candidates[0]["confidence"]

    def test_returns_none_on_near_zero_embedding(self):
        """Returns None when embedding norm is near-zero (corrupted/quantized model output)."""
        extractor = FaceExtractor()
        near_zero_embedding = np.full(512, 1e-10, dtype=np.float32)
        face = _make_face(embedding=near_zero_embedding)

        with (
            patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[face])),
            patch("aria.faces.extractor.cv2.imread", return_value=MagicMock()),
        ):
            result = extractor.extract_embedding("/tmp/fake.jpg")
        assert result is None

    def test_get_app_cached_none_returns_none(self, monkeypatch):
        """_get_app returns None on repeated calls after init failure without retrying."""
        import aria.faces.extractor as ext_mod

        monkeypatch.setattr(ext_mod, "_app_import_attempted", True)
        monkeypatch.setattr(ext_mod, "_app", None)
        result = ext_mod._get_app()
        assert result is None
