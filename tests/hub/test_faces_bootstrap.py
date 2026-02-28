"""Tests for BootstrapPipeline — batch clip extraction + DBSCAN clustering."""

import numpy as np
import pytest

pytest.importorskip("cv2", reason="faces optional deps ([faces] extra) not installed")

from aria.faces.bootstrap import BootstrapPipeline
from aria.faces.store import FaceEmbeddingStore


@pytest.fixture
def store(tmp_path):
    s = FaceEmbeddingStore(str(tmp_path / "faces.db"))
    s.initialize()
    return s


@pytest.fixture
def pipeline(store):
    return BootstrapPipeline(
        clips_dir="/tmp/fake_clips",
        store=store,
    )


class TestBootstrapPipeline:
    def test_scan_returns_jpg_paths(self, pipeline, tmp_path):
        """scan_clips returns all .jpg files in clips dir."""
        clips = tmp_path / "clips"
        clips.mkdir()
        (clips / "backyard-001.jpg").touch()
        (clips / "backyard-002.jpg").touch()
        (clips / "backyard-001-clean.png").touch()  # exclude PNGs
        pipeline.clips_dir = str(clips)
        paths = pipeline.scan_clips()
        assert len(paths) == 2
        assert all(p.endswith(".jpg") for p in paths)

    def test_cluster_embeddings_groups_similar(self, pipeline):
        """DBSCAN clusters near-identical embeddings into same cluster."""
        base = np.random.rand(512).astype(np.float32)
        base /= np.linalg.norm(base)
        # Two tight clusters
        cluster_a = [base + np.random.rand(512).astype(np.float32) * 0.01 for _ in range(5)]
        cluster_b = [-base + np.random.rand(512).astype(np.float32) * 0.01 for _ in range(5)]
        embeddings = cluster_a + cluster_b
        for i, e in enumerate(embeddings):
            embeddings[i] = e / np.linalg.norm(e)

        labels = pipeline.cluster_embeddings(embeddings)
        assert len(set(labels)) == 2  # exactly 2 clusters (no noise with tight groups)

    def test_cluster_returns_noise_label(self, pipeline):
        """DBSCAN returns -1 for outlier embeddings."""
        # Orthogonal unit vectors — cosine distance = 1.0 between every pair,
        # far exceeding eps=0.4, so every point is isolated noise.
        embeddings = []
        for i in range(20):
            v = np.zeros(512, dtype=np.float32)
            v[i] = 1.0  # standard basis vector — orthogonal to all others
            embeddings.append(v)
        labels = pipeline.cluster_embeddings(embeddings)
        # All points are mutually orthogonal (dist=1.0 > eps=0.4) — all noise
        assert -1 in labels

    def test_build_clusters_dict(self, pipeline):
        """build_clusters groups image paths by cluster label."""
        paths = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        labels = [0, 0, 1, -1]  # -1 = noise
        clusters = pipeline.build_clusters(paths, labels)
        assert len(clusters["cluster_0"]) == 2
        assert len(clusters["cluster_1"]) == 1
        assert "unknown" in clusters
        assert len(clusters["unknown"]) == 1
