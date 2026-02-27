"""FastAPI routes for face recognition pipeline."""

import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _BootstrapProgress:
    """Thread-safe bootstrap progress tracker, attached to hub at runtime."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = False
        self.processed = 0
        self.total = 0
        self.started_at: str | None = None
        self.last_ran: str | None = None

    def start(self, total: int) -> None:
        with self._lock:
            self.running = True
            self.processed = 0
            self.total = total
            self.started_at = datetime.now(UTC).isoformat()

    def update(self, processed: int) -> None:
        with self._lock:
            self.processed = processed

    def finish(self) -> None:
        with self._lock:
            self.running = False
            self.last_ran = datetime.now(UTC).isoformat()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.running,
                "processed": self.processed,
                "total": self.total,
                "started_at": self.started_at,
                "last_ran": self.last_ran,
            }


class LabelRequest(BaseModel):
    queue_id: int
    person_name: str


def _register_face_routes(router: APIRouter, hub: Any) -> None:  # noqa: C901, PLR0915
    """Register /api/faces/* endpoints on the given router."""

    def _store():
        store = getattr(hub, "faces_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Face store not initialized")
        return store

    @router.get("/api/faces/queue")
    async def get_face_queue(limit: int = 20):
        """Return pending review queue, highest priority first."""
        try:
            store = _store()
            items = store.get_review_queue(limit=limit)
            # Strip embedding blob from API response (not serializable, not needed by UI)
            for item in items:
                item.pop("embedding", None)
            return {"items": items, "depth": store.get_queue_depth()}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error fetching face queue")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/faces/label")
    async def label_face(req: LabelRequest):
        """Label a queued face and save embedding to training store."""
        try:
            store = _store()
            # Direct lookup by primary key — avoids LIMIT-based scan that returns 404
            # for valid items beyond the first N rows (happens after large bootstrap runs)
            item = store.get_pending_queue_item(req.queue_id)
            if item is None:
                raise HTTPException(status_code=404, detail="Queue item not found or already reviewed")

            # Capture embedding before marking reviewed
            embedding = item["embedding"]

            # mark_reviewed returns True only if this request won the race (rowcount > 0).
            # AND reviewed_at IS NULL in the UPDATE means only the first concurrent
            # request succeeds. Skip add_embedding on False to prevent duplicate
            # training records from concurrent labels of the same queue item.
            won_race = store.mark_reviewed(req.queue_id, person_name=req.person_name)

            if won_race:
                try:
                    store.add_embedding(
                        person_name=req.person_name,
                        embedding=embedding,
                        event_id=item["event_id"],
                        image_path=item["image_path"],
                        confidence=1.0,
                        source="manual",
                        verified=True,
                    )
                except Exception as emb_exc:
                    exc_str = str(emb_exc).lower()
                    if "unique" in exc_str or "duplicate" in exc_str or "constraint" in exc_str:
                        logger.warning(
                            "label_face: duplicate embedding insert for queue_id=%d, person=%s — skipping",
                            req.queue_id,
                            req.person_name,
                        )
                    else:
                        logger.error(
                            "label_face: add_embedding failed for queue_id=%d, person=%s: %s",
                            req.queue_id,
                            req.person_name,
                            emb_exc,
                        )
                        raise HTTPException(status_code=500, detail="Failed to store embedding") from emb_exc
            else:
                logger.info("label_face: queue_id=%d already reviewed by concurrent request, skipping", req.queue_id)

            return {"status": "ok", "person_name": req.person_name}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error labeling face")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/people")
    async def get_known_people():
        """Return all known people with embedding counts."""
        try:
            store = _store()
            return {"people": store.get_known_people()}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error fetching known people")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/stats")
    async def get_face_stats():
        """Return queue depth, known people count, auto-label rate, and pipeline health."""
        try:
            store = _store()
            return {
                "queue_depth": store.get_queue_depth(),
                "known_people": len(store.get_known_people()),
                "auto_label_rate": store.get_auto_label_rate(),
                "last_face_processed_at": getattr(hub, "_face_last_processed", None),
                "face_pipeline_errors": getattr(hub, "_face_pipeline_errors", 0),
            }
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error fetching face stats")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/image/{queue_id}")
    async def get_face_image(queue_id: int):
        """Serve face image for a review queue item."""
        from fastapi.responses import FileResponse

        try:
            store = _store()
            image_path = store.get_queue_image_path(queue_id)
            if not image_path:
                raise HTTPException(status_code=404, detail="Queue item not found")
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise HTTPException(status_code=404, detail="Image not available")
            return FileResponse(str(path), media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error serving face image for queue_id=%d", queue_id)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/embedding-image/{embedding_id}")
    async def get_embedding_image(embedding_id: int):
        """Serve face image for an embedding record (for person gallery)."""
        from fastapi.responses import FileResponse

        try:
            store = _store()
            record = store.get_embeddings_by_id(embedding_id)
            if not record:
                raise HTTPException(status_code=404, detail="Embedding not found")
            image_path = record.get("image_path")
            if not image_path:
                raise HTTPException(status_code=404, detail="Image not available")
            path = Path(image_path)
            if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg"}:
                raise HTTPException(status_code=404, detail="Image not available")
            return FileResponse(str(path), media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error serving embedding image for id=%d", embedding_id)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/bootstrap/status")
    async def get_bootstrap_status():
        """Return current bootstrap progress (running, processed, total, last_ran)."""
        progress: _BootstrapProgress | None = getattr(hub, "_bootstrap_progress", None)
        if progress is None:
            return {"running": False, "processed": 0, "total": 0, "started_at": None, "last_ran": None}
        return progress.snapshot()

    @router.post("/api/faces/bootstrap")
    async def trigger_bootstrap():
        """Kick off bootstrap batch extraction as a background task."""
        import asyncio

        from aria.faces.bootstrap import BootstrapPipeline

        try:
            store = _store()
            clips_dir = os.environ.get("FRIGATE_CLIPS_DIR", str(Path.home() / "frigate/media/clips"))

            # Idempotent: create progress tracker once, reuse across runs
            if not hasattr(hub, "_bootstrap_progress"):
                hub._bootstrap_progress = _BootstrapProgress()
            progress: _BootstrapProgress = hub._bootstrap_progress

            # Guard against concurrent bootstrap runs
            if progress.running:
                return {"status": "already_running", "processed": progress.processed, "total": progress.total}

            # Lock the progress tracker BEFORE purge to close the TOCTOU window.
            # Without this, two concurrent requests that both pass the running check
            # could both call purge + create tasks before either sets running=True.
            # pipeline.run() will call progress.start(actual_total) to set the real total.
            progress.start(0)

            # Purge old reviewed items — only when a fresh bootstrap actually starts
            try:
                store.purge_reviewed_items(older_than_days=7)
            except Exception:
                logger.warning("Failed to purge old queue items before bootstrap")

            async def _run():
                try:
                    pipeline = BootstrapPipeline(clips_dir=clips_dir, store=store)
                    await asyncio.to_thread(pipeline.run, progress)
                finally:
                    progress.finish()

            from aria.shared.utils import log_task_exception

            task = asyncio.create_task(_run())
            task.add_done_callback(log_task_exception)
            return {"status": "started", "clips_dir": clips_dir}
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("bootstrap trigger failed: %s", exc)
            progress_tracker: _BootstrapProgress | None = getattr(hub, "_bootstrap_progress", None)
            if progress_tracker is not None:
                progress_tracker.finish()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/api/faces/deploy")
    async def deploy_to_frigate():
        """Copy best N images per person to Frigate faces dir."""
        import shutil
        from pathlib import Path

        try:
            store = _store()
            frigate_faces = (
                Path(os.environ.get("FRIGATE_CLIPS_DIR", str(Path.home() / "frigate/media/clips"))) / "faces"
            )
            people = store.get_known_people()
            deployed = []
            for person in people:
                name = person["person_name"]
                embeddings = store.get_embeddings_for_person(name)
                top = sorted(
                    [e for e in embeddings if e.get("verified") and e.get("image_path")],
                    key=lambda x: x.get("confidence", 0),
                    reverse=True,
                )[:10]
                dest_dir = frigate_faces / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for i, entry in enumerate(top):
                    src = Path(entry["image_path"])
                    if src.exists():
                        shutil.copy2(src, dest_dir / f"{i:03d}.jpg")
                deployed.append({"person": name, "images": len(top)})
            return {"deployed": deployed}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error deploying to Frigate")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/faces/queue/clear")
    async def clear_review_queue():
        """Mark all pending queue items as reviewed (dismissed) without labeling."""
        try:
            store = _store()
            cleared = store.dismiss_queue()
            logger.info("Cleared %d pending queue items", cleared)
            return {"cleared": cleared}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error clearing queue")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/faces/queue/{queue_id}/dismiss")
    async def dismiss_queue_item(queue_id: int):
        """Dismiss a single pending queue item without labeling."""
        try:
            store = _store()
            found = store.dismiss_single(queue_id)
            if not found:
                raise HTTPException(status_code=404, detail="Queue item not found or already reviewed")
            return {"dismissed": queue_id}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error dismissing queue item %d", queue_id)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/people/{name}/samples")
    async def get_person_samples(name: str):
        """Return all embeddings (samples) for a person, without the embedding blob."""
        try:
            store = _store()
            embeddings = store.get_embeddings_for_person(name)
            # Strip raw embedding blob — not serializable and not needed by UI
            samples = [{k: v for k, v in e.items() if k != "embedding"} for e in embeddings]
            return {"person_name": name, "samples": samples}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error fetching samples for person %s", name)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.delete("/api/faces/people/{name}/samples/{sample_id}")
    async def delete_person_sample(name: str, sample_id: int):
        """Delete a single embedding sample for a person.

        Enforces person_name constraint — sample_id must belong to {name}.
        Returns 404 if not found or belongs to a different person.
        """
        try:
            store = _store()
            deleted = store.delete_embedding_for_person(sample_id, name)
            if not deleted:
                raise HTTPException(status_code=404, detail="Sample not found or does not belong to this person")
            return {"deleted": sample_id}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error deleting sample %d for person %s", sample_id, name)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    class _RenameRequest(BaseModel):
        new_name: str

    @router.post("/api/faces/people/{name}/rename")
    async def rename_person(name: str, req: _RenameRequest):
        """Rename all embeddings from one person name to another."""
        try:
            store = _store()
            count = store.rename_person(name, req.new_name)
            return {"renamed": count, "old_name": name, "new_name": req.new_name}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error renaming person %s -> %s", name, req.new_name)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.delete("/api/faces/people/{name}")
    async def delete_person(name: str):
        """Delete all embeddings for a person."""
        try:
            store = _store()
            count = store.delete_person(name)
            return {"deleted": count, "person_name": name}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error deleting person %s", name)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/faces/export")
    async def export_face_data():
        """Export all labeled people, sample metadata, and pending queue items as JSON."""
        try:
            store = _store()
            people = store.get_known_people()
            export = []
            for person in people:
                name = person["person_name"]
                embeddings = store.get_embeddings_for_person(name)
                samples = [
                    {
                        "id": e["id"],
                        "confidence": e.get("confidence"),
                        "source": e.get("source"),
                        "verified": bool(e.get("verified")),
                        "image_path": e.get("image_path"),
                        "created_at": e.get("created_at"),
                    }
                    for e in embeddings
                    if e.get("image_path")
                ]
                export.append({"person_name": name, "sample_count": len(samples), "samples": samples})

            # Include pending review queue items (unlabeled faces awaiting review)
            queue_items = store.get_review_queue(limit=10000)
            pending_queue = [
                {
                    "id": q["id"],
                    "event_id": q.get("event_id"),
                    "image_path": q.get("image_path"),
                    "priority": q.get("priority"),
                    "top_candidates": q.get("top_candidates"),
                    "camera": q.get("camera"),
                    "created_at": q.get("created_at"),
                }
                for q in queue_items
                if q.get("image_path")
            ]

            return {
                "exported_at": datetime.now(UTC).isoformat(),
                "people": export,
                "pending_queue": pending_queue,
                "pending_queue_count": len(pending_queue),
            }
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error exporting face data")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/faces/restart-frigate")
    async def restart_frigate():
        """Restart the Frigate Docker container to reload the face library."""
        import asyncio
        import subprocess

        def _restart():
            result = subprocess.run(
                ["/usr/bin/docker", "restart", "frigate"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()

        try:
            returncode, stdout, stderr = await asyncio.to_thread(_restart)
            if returncode != 0:
                logger.error("Frigate restart failed (rc=%d): %s", returncode, stderr)
                raise HTTPException(status_code=500, detail=f"Frigate restart failed: {stderr}")
            logger.info("Frigate restarted successfully")
            return {"status": "restarted", "container": stdout or "frigate"}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error restarting Frigate")
            raise HTTPException(status_code=500, detail="Internal server error") from None
