# InsightFace Migration Design

**Date:** 2026-02-27
**Status:** Approved
**Scope:** `aria/faces/extractor.py` + deps + migration script + tests

## Problem

`FaceExtractor` wraps DeepFace (FaceNet512 + RetinaFace) via TensorFlow. On the GTX 1650 Mobile
(4GB VRAM), TF preallocates GPU memory at import and holds it — competing with Ollama for the
same 4GB pool. Per-frame latency is 300–600ms; cold start is 8–15s.

## Solution

Replace DeepFace with InsightFace (`buffalo_l` model pack) via ONNX Runtime GPU.

| Metric | DeepFace (before) | InsightFace buffalo_l (after) |
|--------|------------------|------------------------------|
| Per-frame latency | 300–600ms | ~40–80ms |
| VRAM footprint | ~1–2GB (TF grab) | ~300–500MB (surgical) |
| Accuracy (LFW) | 99.65% (FaceNet512) | 99.83% (ArcFace-R100) |
| Cold start | 8–15s | ~1–2s |
| Ollama coexistence | Poor | Good |

## Architecture

No architectural change — the `FaceExtractor` interface is unchanged:
- `extract_embedding(image_path: str) -> np.ndarray | None` — 512-d L2-normalized ArcFace embedding
- `find_best_match(query, named_embeddings, min_threshold) -> list[dict]` — pure numpy, no backend dep
- `cosine_similarity(a, b) -> float` — unchanged

InsightFace returns Face objects with `.embedding` (512-d, not normalized). When multiple
faces are detected, take the largest by bounding-box area (most likely the primary subject).

Lazy model load on first call — same pattern as the existing `_get_deepface()` sentinel.

## Embedding Incompatibility

DeepFace FaceNet512 and InsightFace ArcFace produce 512-d vectors in different embedding spaces.
Existing embeddings in `face_embeddings` are incompatible and must be discarded.

**Migration approach:** Auto re-embed. All 31 named embeddings are `source='manual'`
and `verified=True` with existing image files on disk. After swapping the extractor, a
one-time migration script re-processes each image through InsightFace and re-inserts with
the original person label. Training set is fully preserved.

## Files Changed

| File | Change |
|------|--------|
| `aria/faces/extractor.py` | Replace DeepFace with InsightFace FaceAnalysis(buffalo_l) |
| `pyproject.toml` | Add insightface, onnxruntime-gpu, opencv-python-headless; remove deepface |
| `scripts/migrate_face_embeddings.py` | One-time migration: clear + re-embed labeled images |
| `tests/hub/test_faces_extractor.py` | Update mocks from DeepFace to InsightFace |

## Dependencies

```
insightface>=0.7.3
onnxruntime-gpu>=1.17.0
opencv-python-headless>=4.8.0
```

Remove: `deepface>=0.0.93`

Note: `onnxruntime-gpu` and `onnxruntime` cannot coexist. Check for conflicts before install.
InsightFace downloads `buffalo_l` model pack (~300MB) to `~/.insightface/models/buffalo_l/`
on first `app.prepare()` call.

## Migration Script Behavior

`scripts/migrate_face_embeddings.py`:
1. Load all distinct `(person_name, image_path)` pairs where `verified=1`
2. Initialize InsightFace extractor (triggers model download if needed)
3. For each image: extract embedding → insert with original person_name, verified=True, source='migration'
4. Clear ALL rows from `face_embeddings` where `source != 'migration'`
5. Print summary: N images processed, N embeddings inserted, N skipped (no face detected)

Run once, then delete the script. Not a recurring job.

## Test Updates

`test_faces_extractor.py` currently patches `aria.faces.extractor._get_deepface`.
After migration, patch the InsightFace app singleton instead:
- `extract_embedding` tests: mock `app.get()` return value (list of Face-like objects)
- No-face case: mock returns `[]`
- Exception case: mock raises `Exception`
- `cosine_similarity` and `find_best_match` tests: no changes (pure numpy)

## Verification

After migration:
1. Run test suite: `pytest tests/hub/test_faces_extractor.py tests/hub/test_faces_pipeline.py -v`
2. Run migration script — confirm N embeddings re-inserted
3. Vertical trace: `aria serve` → trigger bootstrap or submit a snapshot manually → confirm
   queue receives items → label one → confirm `auto_label` fires on re-encounter
4. Check `nvidia-smi` VRAM: InsightFace should show ~300–500MB, not 1–2GB

## Risks

- **Model download at first run**: `buffalo_l` is ~300MB. Will block first inference call
  for ~30–60s on first startup. Add log message so operator knows it's downloading.
- **No faces detected in existing images**: Some labeled images may not yield a face with
  InsightFace (different detector sensitivity). Script prints skip count; manual re-label needed.
- **onnxruntime conflict**: If `onnxruntime` (CPU-only) is installed alongside `onnxruntime-gpu`,
  one must be removed first. Script should check and warn.
