# InsightFace Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace DeepFace (TensorFlow) with InsightFace (ONNX Runtime GPU) in `FaceExtractor`, migrating the 31 existing labeled embeddings automatically.

**Architecture:** Single-file swap in `aria/faces/extractor.py` keeping the exact same public interface. A one-time migration script re-embeds all verified labeled images through the new backend. pyproject.toml `faces`/`faces-gpu` extras are replaced with a new `faces` extra using InsightFace deps.

**Tech Stack:** insightface>=0.7.3, onnxruntime-gpu>=1.17.0, opencv-python-headless>=4.8.0

---

### Task 1: Install dependencies

**Files:**
- Modify: `pyproject.toml` (extras `faces`, `faces-gpu`)

**Step 1: Check for onnxruntime conflict**

```bash
cd ~/Documents/projects/ha-aria
.venv/bin/pip show onnxruntime 2>/dev/null && echo "CONFLICT — must remove first" || echo "No conflict"
```

If conflict exists: `.venv/bin/pip uninstall onnxruntime -y`

**Step 2: Install packages**

```bash
.venv/bin/pip install insightface>=0.7.3 onnxruntime-gpu>=1.17.0 opencv-python-headless>=4.8.0
```

Expected: all three install successfully. InsightFace models are NOT downloaded at this stage.

**Step 3: Verify GPU provider is available**

```bash
.venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Expected output contains: `'CUDAExecutionProvider'`

**Step 4: Update pyproject.toml extras**

Replace the `faces` and `faces-gpu` extras with:

```toml
# Face recognition — install with: pip install -e ".[faces]"
# CUDA support is via onnxruntime-gpu; no separate GPU extra needed
faces = [
    "insightface>=0.7.3",
    "onnxruntime-gpu>=1.17.0",
    "opencv-python-headless>=4.8.0",
]
```

Remove the old `faces-gpu` extra entirely.

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: replace deepface/tensorflow with insightface+onnxruntime-gpu in faces extra"
```

---

### Task 2: Rewrite extractor.py

**Files:**
- Modify: `aria/faces/extractor.py` (full rewrite, same public interface)

**Step 1: Write the failing tests first**

Replace `tests/hub/test_faces_extractor.py` entirely with:

```python
"""Tests for FaceExtractor — InsightFace wrapper."""

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

        with patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[face])), \
             patch("cv2.imread", return_value=MagicMock()):
            result = extractor.extract_embedding("/tmp/fake.jpg")

        assert result is not None
        assert result.shape == (512,)
        assert result.dtype == np.float32
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5  # L2 normalized

    def test_returns_none_on_no_face(self):
        """Returns None when InsightFace finds no face."""
        extractor = FaceExtractor()
        with patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[])), \
             patch("cv2.imread", return_value=MagicMock()):
            result = extractor.extract_embedding("/tmp/fake.jpg")
        assert result is None

    def test_returns_none_on_unreadable_image(self):
        """Returns None when cv2.imread fails (returns None)."""
        extractor = FaceExtractor()
        with patch("aria.faces.extractor._get_app", return_value=_mock_app()), \
             patch("cv2.imread", return_value=None):
            result = extractor.extract_embedding("/tmp/missing.jpg")
        assert result is None

    def test_returns_none_on_exception(self):
        """Returns None (not raise) on unexpected errors."""
        extractor = FaceExtractor()
        with patch("aria.faces.extractor._get_app", return_value=_mock_app(get_side_effect=Exception("GPU OOM"))), \
             patch("cv2.imread", return_value=MagicMock()):
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

        small_face = _make_face(embedding=small_embed, bbox=[0.0, 0.0, 10.0, 10.0])   # area=100
        large_face = _make_face(embedding=large_embed, bbox=[0.0, 0.0, 100.0, 100.0]) # area=10000

        with patch("aria.faces.extractor._get_app", return_value=_mock_app(faces_return=[small_face, large_face])), \
             patch("cv2.imread", return_value=MagicMock()):
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
            {"person_name": "justin", "embedding": query.copy()},   # perfect match
            {"person_name": "carter", "embedding": -query.copy()},  # opposite
        ]
        candidates = extractor.find_best_match(query, named, min_threshold=-1.0)
        assert candidates[0]["person_name"] == "justin"
        assert candidates[0]["confidence"] > 0.99
        assert candidates[1]["confidence"] < candidates[0]["confidence"]
```

**Step 2: Run tests to confirm they fail (patching wrong target)**

```bash
cd ~/Documents/projects/ha-aria
.venv/bin/python -m pytest tests/hub/test_faces_extractor.py -v
```

Expected: most tests FAIL — `_get_app` doesn't exist yet.

**Step 3: Rewrite extractor.py**

Replace `aria/faces/extractor.py` entirely:

```python
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
```

**Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/hub/test_faces_extractor.py -v
```

Expected: all 9 tests PASS.

**Step 5: Run full faces test suite to check no regressions**

```bash
.venv/bin/python -m pytest tests/hub/test_faces_extractor.py tests/hub/test_faces_pipeline.py -v
```

Expected: all pass. `test_faces_pipeline.py` mocks `extractor.extract_embedding` directly so needs no changes.

**Step 6: Commit**

```bash
git add aria/faces/extractor.py tests/hub/test_faces_extractor.py
git commit -m "feat: replace DeepFace with InsightFace buffalo_l in FaceExtractor"
```

---

### Task 3: Write and run the migration script

**Files:**
- Create: `scripts/migrate_face_embeddings.py`

**Step 1: Write the script**

Create `scripts/migrate_face_embeddings.py`:

```python
#!/usr/bin/env python3
"""One-time migration: re-embed labeled faces using InsightFace after DeepFace swap.

Run once after installing InsightFace. Deletes all existing embeddings and
re-processes every verified+labeled image through the new backend.

Usage:
    cd ~/Documents/projects/ha-aria
    .venv/bin/python scripts/migrate_face_embeddings.py
"""

import logging
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    db_path = Path.home() / "ha-logs/intelligence/cache/faces.db"
    if not db_path.exists():
        logger.error("faces.db not found at %s", db_path)
        return 1

    # Check InsightFace is available before touching the DB
    try:
        from aria.faces.extractor import FaceExtractor  # noqa: PLC0415
    except ImportError as e:
        logger.error("Cannot import FaceExtractor: %s", e)
        return 1

    # Load all verified labeled records BEFORE clearing
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute("""
            SELECT DISTINCT person_name, image_path
            FROM face_embeddings
            WHERE verified = 1
              AND person_name IS NOT NULL
              AND image_path IS NOT NULL
        """).fetchall()

    logger.info("Found %d verified labeled images to re-embed", len(rows))
    if not rows:
        logger.warning("No labeled images found — nothing to migrate")
        return 0

    # Initialize extractor (triggers buffalo_l download on first run — may take 30-60s)
    logger.info("Initializing InsightFace extractor...")
    extractor = FaceExtractor()
    # Force init by calling extract_embedding on a dummy path (returns None, but loads model)
    from aria.faces.extractor import _get_app  # noqa: PLC0415
    if _get_app() is None:
        logger.error("InsightFace failed to initialize — check installation")
        return 1
    logger.info("InsightFace ready")

    # Clear ALL existing embeddings
    with closing(sqlite3.connect(str(db_path))) as conn:
        deleted = conn.execute("DELETE FROM face_embeddings").rowcount
        conn.commit()
    logger.info("Cleared %d existing embeddings", deleted)

    # Re-embed each labeled image
    inserted = 0
    skipped_missing = 0
    skipped_no_face = 0

    with closing(sqlite3.connect(str(db_path))) as conn:
        for person_name, image_path in rows:
            path = Path(image_path)
            if not path.exists():
                logger.warning("  SKIP (file missing): %s → %s", person_name, image_path)
                skipped_missing += 1
                continue

            embedding = extractor.extract_embedding(str(path))
            if embedding is None:
                logger.warning("  SKIP (no face detected): %s → %s", person_name, image_path)
                skipped_no_face += 1
                continue

            blob = embedding.astype(np.float32).tobytes()
            conn.execute(
                """
                INSERT INTO face_embeddings
                    (person_name, embedding, event_id, image_path, confidence, source, verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_name,
                    blob,
                    f"migration_{path.stem}",
                    str(path),
                    1.0,
                    "migration",
                    1,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()
            logger.info("  OK: %s → %s", person_name, path.name)
            inserted += 1

    logger.info(
        "\nMigration complete: %d inserted, %d skipped (missing), %d skipped (no face)",
        inserted, skipped_missing, skipped_no_face,
    )
    if skipped_no_face:
        logger.warning(
            "%d images had no detectable face — these people will need to be re-labeled "
            "from new Frigate events or a different source image.", skipped_no_face,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Run it**

```bash
cd ~/Documents/projects/ha-aria
.venv/bin/python scripts/migrate_face_embeddings.py
```

Expected output (approximate):
```
Found 16 verified labeled images to re-embed
Initializing InsightFace extractor...
FaceExtractor: loading InsightFace buffalo_l (first run downloads ~300MB...)
FaceExtractor: InsightFace buffalo_l ready
Cleared 128 existing embeddings
  OK: Justin → <event_id>.jpg
  OK: Lisa → <event_id>.jpg
  ...
Migration complete: 14 inserted, 0 skipped (missing), 2 skipped (no face)
```

**Note on skipped images:** Some surveillance snapshots may not have a detectable face (side profile, too small, blurry). These people will be re-labeled naturally as new Frigate events arrive.

**Step 3: Verify database state**

```bash
.venv/bin/python -c "
import sqlite3
from contextlib import closing
from pathlib import Path
db = Path.home() / 'ha-logs/intelligence/cache/faces.db'
with closing(sqlite3.connect(db)) as conn:
    total = conn.execute('SELECT COUNT(*) FROM face_embeddings').fetchone()[0]
    named = conn.execute('SELECT person_name, COUNT(*) FROM face_embeddings GROUP BY person_name').fetchall()
    print(f'Total: {total}')
    for name, count in named:
        print(f'  {name}: {count} embeddings')
"
```

Expected: named people present with at least 1 embedding each.

**Step 4: Commit**

```bash
git add scripts/migrate_face_embeddings.py
git commit -m "chore: add one-time face embedding migration script (DeepFace → InsightFace)"
```

---

### Task 4: Verify end-to-end

**Step 1: Restart aria-hub**

```bash
systemctl --user restart aria-hub
journalctl --user -u aria-hub -f
```

Watch for: `FaceExtractor: InsightFace buffalo_l ready` — confirms model loaded from cache (fast this time, no download).

**Step 2: Check VRAM usage**

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
```

Expected: InsightFace uses ~300–500 MiB, leaving room for Ollama.

**Step 3: Verify API returns known people**

```bash
curl -s http://127.0.0.1:8001/api/faces/people | python3 -m json.tool
```

Expected: all migrated people appear in the list with embedding counts.

**Step 4: Check face stats**

```bash
curl -s http://127.0.0.1:8001/api/faces/stats | python3 -m json.tool
```

Expected: `known_people > 0`, `face_pipeline_errors = 0`.

**Step 5: Run full faces test suite one final time**

```bash
cd ~/Documents/projects/ha-aria
.venv/bin/python -m pytest tests/hub/test_faces_extractor.py tests/hub/test_faces_pipeline.py tests/hub/test_presence.py -v
```

Expected: all pass.

**Step 6: Commit design doc + cleanup**

```bash
git add docs/plans/2026-02-27-insightface-migration-design.md docs/plans/2026-02-27-insightface-migration.md
git commit -m "docs: add InsightFace migration design and plan"
```
