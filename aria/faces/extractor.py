"""InsightFace wrapper for face detection and embedding extraction."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton — populated on first successful call to _get_app().
# Lazy init keeps ARIA startup fast and avoids model download at import time.
_app = None
_app_import_attempted = False


def _get_app():
    """Return FaceAnalysis app, initializing on first call. Returns None if unavailable."""
    global _app, _app_import_attempted
    if _app_import_attempted:
        return _app
    _app_import_attempted = True
    try:
        from insightface.app import FaceAnalysis  # noqa: PLC0415

        logger.info(
            "FaceExtractor: loading InsightFace buffalo_l"
            " (first run downloads ~300MB to ~/.insightface/models/buffalo_l/)"
        )
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _app = app
        logger.info("FaceExtractor: InsightFace buffalo_l ready")
    except (ImportError, Exception):
        logger.error("FaceExtractor: insightface not available — face recognition disabled")
        _app = None
    return _app


class FaceExtractor:
    """Extract 512-d ArcFace embeddings from image files using InsightFace buffalo_l."""

    def extract_embedding(self, image_path: str) -> np.ndarray | None:
        """Return 512-d float32 L2-normalized ArcFace embedding, or None if no face detected.

        When multiple faces are present, returns the embedding for the largest face
        by bounding-box area (most likely the primary subject in surveillance footage).
        """
        import cv2  # noqa: PLC0415

        app = _get_app()
        if app is None:
            logger.error("FaceExtractor: insightface not installed")
            return None
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.debug("FaceExtractor: could not read image %s", image_path)
                return None
            faces = app.get(img)
            if not faces:
                return None
            # Largest bounding-box area = most prominent face in frame
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            vec = face.embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return None
            return vec / norm  # L2 normalize for cosine similarity
        except Exception:
            logger.exception("FaceExtractor: unexpected error on %s", image_path)
            return None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors.

        For unit vectors, cosine similarity reduces to the dot product — O(n), no sqrt.
        """
        return float(np.dot(a, b))

    def find_best_match(
        self,
        query: np.ndarray,
        named_embeddings: list[dict],
        min_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Compare query against named embeddings, return sorted candidates.

        Groups multiple embeddings per person and averages similarity scores,
        which is more robust to outlier embeddings than taking the maximum.
        """
        scores: dict[str, list[float]] = {}
        for entry in named_embeddings:
            name = entry["person_name"]
            sim = self.cosine_similarity(query, entry["embedding"])
            scores.setdefault(name, []).append(sim)

        candidates = [
            {"person_name": name, "confidence": float(np.mean(sims))}
            for name, sims in scores.items()
            if float(np.mean(sims)) >= min_threshold
        ]
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)
