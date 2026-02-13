"""FastAPI routes for Intelligence Hub REST API."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List, Set
import logging
import json
from datetime import datetime

from aria.hub.core import IntelligenceHub


logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSockets."""
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


def create_api(hub: IntelligenceHub) -> FastAPI:
    """Create FastAPI application with hub routes.

    Args:
        hub: IntelligenceHub instance

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="HA Intelligence Hub API",
        description="REST API for Home Assistant Intelligence Hub",
        version="0.1.0"
    )

    ws_manager = WebSocketManager()

    # Subscribe to hub events for WebSocket broadcasting
    async def broadcast_cache_update(data: Dict[str, Any]):
        await ws_manager.broadcast({
            "type": "cache_updated",
            "data": data
        })

    hub.subscribe("cache_updated", broadcast_cache_update)

    # Mount SPA dashboard (serves index.html for all unmatched /ui/ paths)
    spa_dist = Path(__file__).parent.parent / "dashboard" / "spa" / "dist"
    app.mount("/ui", StaticFiles(directory=str(spa_dist), html=True), name="spa")

    # Health check
    @app.get("/")
    async def root():
        """API root - health check."""
        return {"status": "ok", "service": "HA Intelligence Hub"}

    @app.get("/health")
    async def health():
        """Detailed health check."""
        try:
            health_data = await hub.health_check()
            return JSONResponse(content=health_data)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": str(e)}
            )

    # Cache endpoints
    @app.get("/api/cache")
    async def list_cache_categories():
        """List all cache categories."""
        try:
            categories = await hub.cache.list_categories()
            return {"categories": categories}
        except Exception as e:
            logger.error(f"Error listing cache categories: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/cache/{category}")
    async def get_cache(category: str):
        """Get cache data by category."""
        try:
            data = await hub.get_cache(category)
            if data is None:
                raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
            return data
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting cache '{category}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/cache/{category}")
    async def set_cache(category: str, payload: Dict[str, Any]):
        """Set cache data for category.

        Request body:
        {
            "data": {...},
            "metadata": {...}  // optional
        }
        """
        try:
            data = payload.get("data")
            if data is None:
                raise HTTPException(status_code=400, detail="Missing 'data' field in request body")

            metadata = payload.get("metadata")
            version = await hub.set_cache(category, data, metadata)

            return {
                "status": "ok",
                "category": category,
                "version": version
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error setting cache '{category}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/cache/{category}")
    async def delete_cache(category: str):
        """Delete cache category."""
        try:
            deleted = await hub.cache.delete(category)
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

            return {"status": "ok", "category": category, "deleted": True}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting cache '{category}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Events endpoints
    @app.get("/api/events")
    async def get_events(
        event_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100
    ):
        """Get recent events from event log."""
        try:
            events = await hub.cache.get_events(
                event_type=event_type,
                category=category,
                limit=limit
            )
            return {"events": events, "count": len(events)}
        except Exception as e:
            logger.error(f"Error getting events: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Module management endpoints
    @app.get("/api/modules")
    async def list_modules():
        """List all registered modules."""
        try:
            modules = [
                {
                    "module_id": module_id,
                    "registered": True
                }
                for module_id in hub.modules.keys()
            ]
            return {"modules": modules, "count": len(modules)}
        except Exception as e:
            logger.error(f"Error listing modules: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/modules/{module_id}")
    async def get_module(module_id: str):
        """Get module information."""
        try:
            module = await hub.get_module(module_id)
            if module is None:
                raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

            return {
                "module_id": module.module_id,
                "registered": True
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting module '{module_id}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Shadow engine endpoints
    @app.get("/api/shadow/predictions")
    async def get_shadow_predictions(limit: int = 50, offset: int = 0):
        """Get recent predictions with outcomes."""
        try:
            predictions = await hub.cache.get_recent_predictions(limit=limit, offset=offset)
            return {"predictions": predictions, "count": len(predictions)}
        except Exception as e:
            logger.error(f"Error getting shadow predictions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/shadow/accuracy")
    async def get_shadow_accuracy():
        """Get shadow engine accuracy metrics."""
        try:
            stats = await hub.cache.get_accuracy_stats()
            pipeline = await hub.cache.get_pipeline_state()

            return {
                "overall_accuracy": stats.get("overall_accuracy", 0),
                "predictions_total": stats.get("total_resolved", 0),
                "predictions_correct": stats.get("per_outcome", {}).get("correct", 0),
                "predictions_disagreement": stats.get("per_outcome", {}).get("disagreement", 0),
                "predictions_nothing": stats.get("per_outcome", {}).get("nothing", 0),
                "by_type": stats.get("per_outcome", {}),
                "daily_trend": stats.get("daily_trend", []),
                "stage": pipeline.get("current_stage", "shadow") if pipeline else "shadow",
            }
        except Exception as e:
            logger.error(f"Error getting shadow accuracy: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/shadow/disagreements")
    async def get_shadow_disagreements(limit: int = 20):
        """Get top disagreements sorted by confidence (most informative first)."""
        try:
            predictions = await hub.cache.get_recent_predictions(
                limit=200, outcome_filter="disagreement"
            )
            # Sort by confidence descending (highest confidence wrong = most informative)
            sorted_preds = sorted(
                predictions,
                key=lambda p: p.get("confidence", 0),
                reverse=True
            )[:limit]
            return {"disagreements": sorted_preds, "count": len(sorted_preds)}
        except Exception as e:
            logger.error(f"Error getting shadow disagreements: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/pipeline")
    async def get_pipeline_status():
        """Get current pipeline stage and gate progress."""
        try:
            pipeline = await hub.cache.get_pipeline_state()
            if pipeline is None:
                return {
                    "current_stage": "shadow",
                    "gates": {},
                    "message": "Pipeline state not yet initialized"
                }
            return pipeline
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    PIPELINE_STAGES = ['backtest', 'shadow', 'suggest', 'autonomous']
    PIPELINE_GATES = {
        'backtest': {'field': 'backtest_accuracy', 'threshold': 0.40, 'label': 'backtest accuracy'},
        'shadow': {'field': 'shadow_accuracy_7d', 'threshold': 0.50, 'label': '7-day shadow accuracy'},
        'suggest': {'field': 'suggest_approval_rate_14d', 'threshold': 0.70, 'label': '14-day approval rate'},
    }

    @app.post("/api/pipeline/advance")
    async def pipeline_advance():
        """Advance pipeline to next stage (with gate validation)."""
        try:
            pipeline = await hub.cache.get_pipeline_state()
            if pipeline is None:
                raise HTTPException(status_code=400, detail="Pipeline state not initialized")

            current = pipeline["current_stage"]
            idx = PIPELINE_STAGES.index(current)

            if idx >= len(PIPELINE_STAGES) - 1:
                raise HTTPException(status_code=400, detail="Already at final stage")

            # Check gate requirement for current stage
            gate = PIPELINE_GATES.get(current)
            if gate:
                current_value = pipeline.get(gate['field']) or 0
                if current_value < gate['threshold']:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Gate not met",
                            "gate": gate['label'],
                            "required": gate['threshold'],
                            "current": current_value,
                        }
                    )

            next_stage = PIPELINE_STAGES[idx + 1]
            now = datetime.now().isoformat()
            await hub.cache.update_pipeline_state(
                current_stage=next_stage,
                stage_entered_at=now,
            )

            updated = await hub.cache.get_pipeline_state()
            await hub.publish("pipeline_updated", updated)
            return updated

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error advancing pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/pipeline/retreat")
    async def pipeline_retreat():
        """Retreat pipeline to previous stage (no gates)."""
        try:
            pipeline = await hub.cache.get_pipeline_state()
            if pipeline is None:
                raise HTTPException(status_code=400, detail="Pipeline state not initialized")

            current = pipeline["current_stage"]
            idx = PIPELINE_STAGES.index(current)

            if idx <= 0:
                raise HTTPException(status_code=400, detail="Already at first stage")

            prev_stage = PIPELINE_STAGES[idx - 1]
            now = datetime.now().isoformat()
            await hub.cache.update_pipeline_state(
                current_stage=prev_stage,
                stage_entered_at=now,
            )

            updated = await hub.cache.get_pipeline_state()
            await hub.publish("pipeline_updated", updated)
            return updated

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retreating pipeline: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Config endpoints
    @app.get("/api/config")
    async def get_all_config():
        """Get all configuration parameters."""
        try:
            configs = await hub.cache.get_all_config()
            return {"configs": configs}
        except Exception as e:
            logger.error(f"Error getting all config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/config/reset/{key:path}")
    async def reset_config(key: str):
        """Reset a configuration parameter to its default value."""
        try:
            config = await hub.cache.reset_config(key)
            return config
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error resetting config '{key}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/config/{key:path}")
    async def get_config(key: str):
        """Get a single configuration parameter."""
        try:
            config = await hub.cache.get_config(key)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
            return config
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting config '{key}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/config/{key:path}")
    async def put_config(key: str, request: Request):
        """Update a configuration parameter value."""
        try:
            body = await request.json()
            value = body.get("value")
            changed_by = body.get("changed_by", "user")
            config = await hub.cache.set_config(key, value, changed_by=changed_by)
            await ws_manager.broadcast({"type": "config_updated", "data": {"key": key}})
            return config
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error updating config '{key}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/config-history")
    async def get_config_history(key: Optional[str] = None, limit: int = 50):
        """Get configuration change history."""
        try:
            history = await hub.cache.get_config_history(key=key, limit=limit)
            return {"history": history, "count": len(history)}
        except Exception as e:
            logger.error(f"Error getting config history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Curation endpoints
    @app.get("/api/curation")
    async def get_all_curation():
        """Get all entity curation classifications."""
        try:
            curations = await hub.cache.get_all_curation()
            return {"curations": curations}
        except Exception as e:
            logger.error(f"Error getting all curation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/curation/summary")
    async def get_curation_summary():
        """Get curation tier/status counts summary."""
        try:
            summary = await hub.cache.get_curation_summary()
            return summary
        except Exception as e:
            logger.error(f"Error getting curation summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/api/curation/{entity_id:path}")
    async def put_curation(entity_id: str, request: Request):
        """Override a single entity's curation classification."""
        try:
            body = await request.json()
            status = body.get("status")
            decided_by = body.get("decided_by", "user")
            result = await hub.cache.upsert_curation(
                entity_id, status=status, decided_by=decided_by, human_override=True
            )
            await hub.publish("curation_updated", {"entity_id": entity_id, "status": status})
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error updating curation for '{entity_id}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/curation/bulk")
    async def bulk_update_curation(request: Request):
        """Bulk approve/reject entity curations."""
        try:
            body = await request.json()
            entity_ids = body.get("entity_ids", [])
            status = body.get("status")
            decided_by = body.get("decided_by", "user")
            count = await hub.cache.bulk_update_curation(
                entity_ids, status=status, decided_by=decided_by
            )
            await hub.publish("curation_updated", {"count": count, "status": status})
            return {"updated": count}
        except Exception as e:
            logger.error(f"Error bulk updating curation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time events."""
        await ws_manager.connect(websocket)

        try:
            # Send initial connection message
            await websocket.send_json({
                "type": "connected",
                "message": "Connected to HA Intelligence Hub"
            })

            # Keep connection alive
            while True:
                try:
                    # Receive messages from client (ping/pong, subscriptions, etc.)
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    # Handle client messages
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        logger.debug(f"Received WebSocket message: {message}")

                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break

        finally:
            ws_manager.disconnect(websocket)

    return app
