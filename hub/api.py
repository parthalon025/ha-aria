"""FastAPI routes for Intelligence Hub REST API."""

from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, Any, List, Set
import logging
import json

from hub.core import IntelligenceHub
from dashboard.routes import create_dashboard_router


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

    # Mount dashboard static files
    dashboard_static = Path(__file__).parent.parent / "dashboard" / "static"
    app.mount("/ui/static", StaticFiles(directory=str(dashboard_static)), name="dashboard_static")

    # Mount dashboard router
    dashboard_router = create_dashboard_router(hub)
    app.include_router(dashboard_router)

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
