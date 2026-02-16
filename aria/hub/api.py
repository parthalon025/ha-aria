"""FastAPI routes for ARIA Hub REST API."""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Security,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aria.hub.core import IntelligenceHub

logger = logging.getLogger(__name__)

# --- Optional API key authentication ---
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_ARIA_API_KEY = os.environ.get("ARIA_API_KEY")


async def verify_api_key(key: str = Security(_api_key_header)):
    """Verify API key if ARIA_API_KEY is configured, otherwise allow all."""
    if _ARIA_API_KEY and key != _ARIA_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# --- Pydantic request models ---
class ConfigUpdate(BaseModel):
    value: Any
    changed_by: str = "user"


class CurationUpdate(BaseModel):
    status: str
    decided_by: str = "user"


class BulkCurationUpdate(BaseModel):
    entity_ids: list[str]
    status: str
    decided_by: str = "user"


class WebSocketManager:
    """Manages WebSocket connections and broadcasts."""

    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]):
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


PIPELINE_STAGES = ["backtest", "shadow", "suggest", "autonomous"]
PIPELINE_GATES = {
    "backtest": {"field": "backtest_accuracy", "threshold": 0.40, "label": "backtest accuracy"},
    "shadow": {"field": "shadow_accuracy_7d", "threshold": 0.50, "label": "7-day shadow accuracy"},
    "suggest": {"field": "suggest_approval_rate_14d", "threshold": 0.70, "label": "14-day approval rate"},
}


def _register_cache_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register cache CRUD endpoints on the router."""

    @router.get("/api/cache")
    async def list_cache_categories():
        """List all cache categories."""
        try:
            categories = await hub.cache.list_categories()
            return {"categories": categories}
        except Exception:
            logger.exception("Error listing cache categories")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/cache/{category}")
    async def get_cache(category: str):
        """Get cache data by category."""
        try:
            data = await hub.get_cache(category)
            if data is None:
                raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
            return data
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error getting cache '%s'", category)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/cache/{category}")
    async def set_cache(category: str, payload: dict[str, Any]):
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

            return {"status": "ok", "category": category, "version": version}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error setting cache '%s'", category)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.delete("/api/cache/{category}")
    async def delete_cache(category: str):
        """Delete cache category."""
        try:
            deleted = await hub.cache.delete(category)
            if not deleted:
                raise HTTPException(status_code=404, detail=f"Category '{category}' not found")

            return {"status": "ok", "category": category, "deleted": True}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error deleting cache '%s'", category)
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_utility_routes(router: APIRouter, hub: IntelligenceHub, ws_manager: WebSocketManager) -> None:
    """Register utility endpoints (version, cache keys, metrics, events, modules)."""
    from aria import __version__

    @router.get("/api/version")
    async def get_version():
        """Return package version and runtime info."""
        return {
            "version": __version__,
            "package": "ha-aria",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

    @router.get("/api/cache/keys")
    async def list_cache_keys():
        """List all cached categories with last-updated timestamps."""
        try:
            categories = await hub.cache.list_categories()
            keys = []
            for cat in categories:
                entry = await hub.cache.get(cat)
                keys.append(
                    {
                        "category": cat,
                        "last_updated": entry.get("last_updated") if entry else None,
                        "version": entry.get("version") if entry else None,
                    }
                )
            return {"keys": keys, "count": len(keys)}
        except Exception:
            logger.exception("Error listing cache keys")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/metrics")
    async def get_metrics():
        """Return basic operational metrics."""
        return {
            "cache_categories": len(await hub.cache.list_categories()),
            "uptime_seconds": round(hub.get_uptime_seconds()),
            "requests_total": hub._request_count,
            "websocket_clients": len(ws_manager.active_connections),
        }

    # Events endpoints
    @router.get("/api/events")
    async def get_events(
        event_type: str | None = None,
        category: str | None = None,
        limit: int = Query(default=100, le=1000),
    ):
        """Get recent events from event log."""
        try:
            events = await hub.cache.get_events(event_type=event_type, category=category, limit=limit)
            return {"events": events, "count": len(events)}
        except Exception:
            logger.exception("Error getting events")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    # Module management endpoints
    @router.get("/api/modules")
    async def list_modules():
        """List all registered modules."""
        try:
            modules = [{"module_id": module_id, "registered": True} for module_id in hub.modules]
            return {"modules": modules, "count": len(modules)}
        except Exception:
            logger.exception("Error listing modules")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/modules/{module_id}")
    async def get_module(module_id: str):
        """Get module information."""
        try:
            module = await hub.get_module(module_id)
            if module is None:
                raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

            return {"module_id": module.module_id, "registered": True}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error getting module '%s'", module_id)
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_ml_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register ML-related endpoints (drift, features, models, anomalies, SHAP, pipeline)."""

    @router.get("/api/ml/drift")
    async def get_ml_drift():
        """Get drift detection status per metric."""
        try:
            intel = await hub.cache.get("intelligence")
            if not intel:
                return {"metrics": {}, "needs_retrain": False, "method": "none", "days_analyzed": 0}
            drift = intel.get("drift_status", {})
            return {
                "needs_retrain": drift.get("needs_retrain", False),
                "reason": drift.get("reason", "no data"),
                "drifted_metrics": drift.get("drifted_metrics", []),
                "rolling_mae": drift.get("rolling_mae", {}),
                "current_mae": drift.get("current_mae", {}),
                "threshold": drift.get("threshold", {}),
                "page_hinkley": drift.get("page_hinkley", {}),
                "adwin": drift.get("adwin", {}),
                "days_analyzed": drift.get("days_analyzed", 0),
            }
        except Exception:
            logger.exception("Error getting ML drift status")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/ml/features")
    async def get_ml_features():
        """Get mRMR feature selection results."""
        try:
            intel = await hub.cache.get("intelligence")
            if not intel:
                return {"selected": [], "total": 0, "method": "none"}
            features = intel.get("feature_selection", {})
            return {
                "selected": features.get("selected_features", []),
                "total": features.get("total_features", 0),
                "method": features.get("method", "mrmr"),
                "max_features": features.get("max_features", 30),
                "last_computed": features.get("last_computed"),
            }
        except Exception:
            logger.exception("Error getting ML features")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/ml/models")
    async def get_ml_models():
        """Get model health: reference comparison, incremental status, forecaster."""
        try:
            intel = await hub.cache.get("intelligence")
            if not intel:
                return {"reference": None, "incremental": None, "forecaster": None, "ml_models": None}
            return {
                "reference": intel.get("reference_model", None),
                "incremental": intel.get("incremental_training", None),
                "forecaster": intel.get("forecaster_backend", None),
                "ml_models": intel.get("ml_models", None),
            }
        except Exception:
            logger.exception("Error getting ML models")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/ml/anomalies")
    async def get_ml_anomalies():
        """Get recent anomaly detection results."""
        try:
            intel = await hub.cache.get("intelligence")
            if not intel:
                return {"anomalies": [], "autoencoder": {"enabled": False}, "isolation_forest": {}}
            return {
                "anomalies": intel.get("anomaly_alerts", []),
                "autoencoder": intel.get("autoencoder_status", {"enabled": False}),
                "isolation_forest": intel.get("isolation_forest_status", {}),
            }
        except Exception:
            logger.exception("Error getting ML anomalies")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/ml/shap")
    async def get_ml_shap():
        """Get SHAP feature attributions for latest predictions."""
        try:
            intel = await hub.cache.get("intelligence")
            if not intel:
                return {"attributions": [], "available": False}
            shap_data = intel.get("shap_attributions", {})
            return {
                "available": bool(shap_data),
                "attributions": shap_data.get("attributions", []),
                "model_type": shap_data.get("model_type"),
                "computed_at": shap_data.get("computed_at"),
            }
        except Exception:
            logger.exception("Error getting SHAP attributions")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/ml/pipeline")
    async def get_ml_pipeline():
        """Aggregate ML pipeline state for the Pipeline Overview UI."""
        try:
            intel = await hub.cache.get("intelligence") or {}
            training_meta = await hub.cache.get("ml_training_metadata")
            training_data = (training_meta or {}).get("data", training_meta or {})
            presence = await hub.cache.get("presence")
            presence_data = (presence or {}).get("data", presence or {})
            feedback_health = await hub.cache.get("feedback_health")
            fb_data = (feedback_health or {}).get("data", feedback_health or {})

            snapshot_log = intel.get("snapshot_log", {})
            feature_sel = intel.get("feature_selection", {})
            ml_models = intel.get("ml_models", {})

            return {
                "data_collection": {
                    "entity_count": intel.get("entity_count", 0),
                    "snapshot_count_intraday": snapshot_log.get("intraday_count", 0),
                    "snapshot_count_daily": snapshot_log.get("daily_count", 0),
                    "last_snapshot": snapshot_log.get("last_snapshot"),
                    "health_guard": intel.get("health_guard", {}).get("status", "unknown"),
                    "presence_connected": bool(presence_data.get("connected", False)),
                },
                "feature_engineering": {
                    "total_features": feature_sel.get("total_features", 0),
                    "selected_features": feature_sel.get("selected_features", []),
                    "method": feature_sel.get("method", "none"),
                },
                "model_training": {
                    "last_trained": training_data.get("last_trained"),
                    "model_types": ["GradientBoosting", "RandomForest", "LightGBM"],
                    "targets": training_data.get("targets_trained", []),
                    "validation_split": "80/20 chronological",
                    "total_snapshots_used": training_data.get("num_snapshots", 0),
                },
                "predictions": {
                    "scores": ml_models.get("scores", {}),
                },
                "feedback_loop": {
                    "ml_feedback_caps": (fb_data.get("ml_feedback", {}) or {}).get("capabilities_updated", 0),
                    "shadow_feedback_caps": (fb_data.get("shadow_feedback", {}) or {}).get("capabilities_updated", 0),
                    "drift_flagged": len(intel.get("drift_status", {}).get("drifted_metrics", [])),
                    "last_feedback_write": training_data.get("last_trained"),
                },
            }
        except Exception:
            logger.exception("Error getting ML pipeline state")
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_discovery_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register organic discovery and capability registry endpoints."""

    # --- Organic discovery endpoints ---

    @router.get("/api/capabilities/candidates")
    async def get_capability_candidates():
        """Return only candidate capabilities."""
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            return {}
        return {name: cap for name, cap in cached["data"].items() if cap.get("status") == "candidate"}

    @router.get("/api/capabilities/history")
    async def get_discovery_history():
        """Return discovery run history."""
        cached = await hub.cache.get("discovery_history")
        if not cached or not cached.get("data"):
            return []
        return cached["data"]

    @router.put("/api/capabilities/{capability_name}/promote")
    async def promote_capability(capability_name: str):
        """Manually promote a candidate capability."""
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            raise HTTPException(status_code=404, detail="Capabilities not found")
        caps = cached["data"]
        if capability_name not in caps:
            raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")
        caps[capability_name]["status"] = "promoted"
        caps[capability_name]["promoted_at"] = datetime.utcnow().strftime("%Y-%m-%d")
        await hub.cache.set("capabilities", caps)
        return {"capability": capability_name, "status": "promoted"}

    @router.put("/api/capabilities/{capability_name}/archive")
    async def archive_capability(capability_name: str):
        """Manually archive a capability."""
        cached = await hub.cache.get("capabilities")
        if not cached or not cached.get("data"):
            raise HTTPException(status_code=404, detail="Capabilities not found")
        caps = cached["data"]
        if capability_name not in caps:
            raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")
        caps[capability_name]["status"] = "archived"
        await hub.cache.set("capabilities", caps)
        return {"capability": capability_name, "status": "archived"}

    @router.get("/api/settings/discovery")
    async def get_discovery_settings():
        """Get current discovery settings."""
        module = hub.modules.get("organic_discovery")
        if not module:
            return {"error": "Organic discovery module not loaded"}
        return module.settings

    @router.put("/api/settings/discovery")
    async def update_discovery_settings(body: dict = Body(...)):  # noqa: B008
        """Update discovery settings."""
        module = hub.modules.get("organic_discovery")
        if not module:
            raise HTTPException(status_code=404, detail="Organic discovery module not loaded")
        await module.update_settings(body)
        return {"status": "updated", "settings": module.settings}

    @router.post("/api/discovery/run")
    async def trigger_discovery_run():
        """Trigger an on-demand discovery run."""
        module = hub.modules.get("organic_discovery")
        if not module:
            raise HTTPException(status_code=404, detail="Organic discovery module not loaded")
        import asyncio

        asyncio.create_task(module.run_discovery())
        return {"status": "started"}

    @router.get("/api/discovery/status")
    async def get_discovery_status():
        """Get discovery module status."""
        module = hub.modules.get("organic_discovery")
        if not module:
            return {"loaded": False}
        history = module.history
        return {
            "loaded": True,
            "last_run": history[-1]["timestamp"] if history else None,
            "total_runs": len(history),
            "settings": module.settings,
        }

    # Capability prediction toggle
    @router.put("/api/capabilities/{capability_name}/can-predict")
    async def toggle_can_predict(capability_name: str, body: dict = Body(...)):  # noqa: B008
        """Toggle can_predict flag for a capability."""
        try:
            cached = await hub.cache.get("capabilities")
            if not cached or not cached.get("data"):
                raise HTTPException(status_code=404, detail="Capabilities not yet discovered")
            caps = cached["data"]
            if capability_name not in caps:
                raise HTTPException(status_code=404, detail=f"Unknown capability: {capability_name}")
            value = body.get("can_predict")
            if not isinstance(value, bool):
                raise HTTPException(status_code=400, detail="can_predict must be a boolean")
            caps[capability_name]["can_predict"] = value
            await hub.cache.set("capabilities", caps)
            return {"capability": capability_name, "can_predict": value}
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error toggling can_predict")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    # --- Capability Registry API ---
    @router.get("/api/capabilities/registry")
    async def list_capability_registry(
        layer: str | None = None,
        status: str | None = None,
    ):
        """List all declared capabilities from the code registry."""
        from dataclasses import asdict

        from aria.capabilities import CapabilityRegistry

        registry = CapabilityRegistry()
        registry.collect_from_modules()
        caps = registry.list_all()
        if layer:
            caps = [c for c in caps if c.layer == layer]
        if status:
            caps = [c for c in caps if c.status == status]
        return {
            "capabilities": [asdict(c) for c in caps],
            "total": len(caps),
            "by_layer": {lyr: len([c for c in caps if c.layer == lyr]) for lyr in sorted({c.layer for c in caps})},
            "by_status": {st: len([c for c in caps if c.status == st]) for st in sorted({c.status for c in caps})},
        }

    @router.get("/api/capabilities/registry/graph")
    async def capabilities_registry_graph():
        """Dependency graph of all registered capabilities."""
        from aria.capabilities import CapabilityRegistry

        registry = CapabilityRegistry()
        registry.collect_from_modules()
        graph = registry.dependency_graph()
        nodes = [
            {"id": cap.id, "name": cap.name, "layer": cap.layer, "status": cap.status} for cap in registry.list_all()
        ]
        edges = []
        for cap_id, deps in graph.items():
            for dep_id in deps:
                edges.append({"from": dep_id, "to": cap_id})
        return {"nodes": nodes, "edges": edges}

    @router.get("/api/capabilities/registry/health")
    async def capabilities_registry_health():
        """Runtime health per capability — checks module status from hub."""
        from aria.capabilities import CapabilityRegistry

        registry = CapabilityRegistry()
        registry.collect_from_modules()
        module_status = {}
        if hasattr(hub, "module_status"):
            module_status = dict(hub.module_status)
        return registry.health(module_status)

    @router.get("/api/capabilities/registry/{capability_id}")
    async def get_capability_registry(capability_id: str):
        """Get a single capability by ID from the code registry."""
        from dataclasses import asdict

        from aria.capabilities import CapabilityRegistry

        registry = CapabilityRegistry()
        registry.collect_from_modules()
        cap = registry.get(capability_id)
        if not cap:
            raise HTTPException(status_code=404, detail=f"Capability '{capability_id}' not found")
        return asdict(cap)


def _register_feedback_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register automation feedback, feedback health, and activity labeling endpoints."""

    # Automation suggestion feedback endpoints
    @router.post("/api/automations/feedback")
    async def record_automation_feedback(body: dict = Body(...)):  # noqa: B008
        """Record user feedback on an automation suggestion."""
        suggestion_id = body.get("suggestion_id")
        capability_source = body.get("capability_source")
        user_action = body.get("user_action")

        if not suggestion_id or not capability_source or not user_action:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing required fields: suggestion_id, capability_source, user_action"},
            )

        allowed_actions = ("accepted", "rejected", "modified", "ignored")
        if user_action not in allowed_actions:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid user_action. Must be one of: {', '.join(allowed_actions)}"},
            )

        # Read existing feedback data or create empty structure
        cached = await hub.get_cache("automation_feedback")
        feedback_data = cached["data"] if cached and cached.get("data") else {"suggestions": {}, "per_capability": {}}

        # Store feedback entry
        feedback_data["suggestions"][suggestion_id] = {
            "capability_source": capability_source,
            "user_action": user_action,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Update per-capability counters
        cap_stats = feedback_data["per_capability"].get(
            capability_source,
            {
                "suggested": 0,
                "accepted": 0,
                "rejected": 0,
                "acceptance_rate": 0.0,
            },
        )
        cap_stats["suggested"] = cap_stats.get("suggested", 0) + 1
        if user_action == "accepted":
            cap_stats["accepted"] = cap_stats.get("accepted", 0) + 1
        elif user_action == "rejected":
            cap_stats["rejected"] = cap_stats.get("rejected", 0) + 1

        total = cap_stats["suggested"]
        cap_stats["acceptance_rate"] = round(cap_stats["accepted"] / total, 3) if total > 0 else 0.0
        feedback_data["per_capability"][capability_source] = cap_stats

        await hub.set_cache("automation_feedback", feedback_data, {"source": "user_feedback"})

        return {"status": "recorded", "suggestion_id": suggestion_id}

    @router.get("/api/automations/feedback")
    async def get_automation_feedback():
        """Get automation suggestion feedback history."""
        cached = await hub.get_cache("automation_feedback")
        if cached and cached.get("data"):
            return cached["data"]
        return {"suggestions": {}, "per_capability": {}}

    # Feedback health endpoint
    @router.get("/api/capabilities/feedback/health")
    async def get_feedback_health():
        """Summarize health of all feedback channels."""
        entry = await hub.get_cache("capabilities")
        caps = entry.get("data", {}) if entry else {}
        ml_count = sum(1 for c in caps.values() if isinstance(c, dict) and c.get("ml_accuracy"))
        shadow_count = sum(1 for c in caps.values() if isinstance(c, dict) and c.get("shadow_accuracy"))
        drift_count = sum(1 for c in caps.values() if isinstance(c, dict) and c.get("drift_flagged"))

        labels_entry = await hub.get_cache("activity_labels")
        label_stats = labels_entry.get("data", {}).get("label_stats", {}) if labels_entry else {}

        feedback_entry = await hub.get_cache("automation_feedback")
        suggestion_stats = feedback_entry.get("data", {}).get("per_capability", {}) if feedback_entry else {}

        return {
            "capabilities_total": len([c for c in caps.values() if isinstance(c, dict)]),
            "capabilities_with_ml_feedback": ml_count,
            "capabilities_with_shadow_feedback": shadow_count,
            "capabilities_drift_flagged": drift_count,
            "activity_labels": label_stats.get("total_labels", 0),
            "activity_classifier_ready": label_stats.get("classifier_ready", False),
            "automation_feedback_count": sum(v.get("suggested", 0) for v in suggestion_stats.values()),
        }

    # Activity labeling endpoints
    @router.get("/api/activity/current")
    async def get_current_activity():
        """Get current predicted activity."""
        entry = await hub.get_cache("activity_labels")
        if not entry or not entry.get("data"):
            return {"predicted": "unknown", "confidence": 0, "method": "none"}
        return entry["data"].get("current_activity", {"predicted": "unknown", "confidence": 0, "method": "none"})

    @router.post("/api/activity/label")
    async def post_activity_label(body: dict = Body(...)):  # noqa: B008
        """Record a user-confirmed or corrected activity label."""
        actual = body.get("actual_activity", "")
        if not actual:
            return JSONResponse(
                status_code=400,
                content={"error": "actual_activity required"},
            )
        entry = await hub.get_cache("activity_labels")
        current = entry["data"].get("current_activity", {}) if entry and entry.get("data") else {}
        predicted = current.get("predicted", "unknown")
        context = current.get("sensor_context", {})
        source = "confirmed" if actual == predicted else "corrected"
        labeler = hub.modules.get("activity_labeler")
        if labeler:
            stats = await labeler.record_label(predicted, actual, context, source)
            return {"status": "recorded", "predicted": predicted, "actual": actual, "source": source, "stats": stats}
        return {"status": "recorded", "predicted": predicted, "actual": actual, "source": source}

    @router.get("/api/activity/labels")
    async def get_activity_labels(limit: int = 50):
        """Get activity label history."""
        entry = await hub.get_cache("activity_labels")
        if not entry or not entry.get("data"):
            return {"labels": [], "label_stats": {}}
        data = entry["data"]
        return {"labels": data.get("labels", [])[-limit:], "label_stats": data.get("label_stats", {})}

    @router.get("/api/activity/stats")
    async def get_activity_stats():
        """Get activity labeling statistics."""
        entry = await hub.get_cache("activity_labels")
        if not entry or not entry.get("data"):
            return {"total_labels": 0, "classifier_ready": False}
        return entry["data"].get("label_stats", {})


def _register_shadow_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register shadow engine endpoints."""

    @router.get("/api/shadow/predictions")
    async def get_shadow_predictions(
        limit: int = Query(default=50, le=1000),
        offset: int = 0,
    ):
        """Get recent predictions with outcomes."""
        try:
            predictions = await hub.cache.get_recent_predictions(limit=limit, offset=offset)
            return {"predictions": predictions, "count": len(predictions)}
        except Exception:
            logger.exception("Error getting shadow predictions")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/shadow/accuracy")
    async def get_shadow_accuracy():
        """Get shadow engine accuracy metrics."""
        try:
            stats = await hub.cache.get_accuracy_stats()
            pipeline = await hub.cache.get_pipeline_state()

            result = {
                "overall_accuracy": stats.get("overall_accuracy", 0),
                "predictions_total": stats.get("total_resolved", 0),
                "predictions_correct": stats.get("per_outcome", {}).get("correct", 0),
                "predictions_disagreement": stats.get("per_outcome", {}).get("disagreement", 0),
                "predictions_nothing": stats.get("per_outcome", {}).get("nothing", 0),
                "by_type": stats.get("per_outcome", {}),
                "daily_trend": stats.get("daily_trend", []),
                "stage": pipeline.get("current_stage", "shadow") if pipeline else "shadow",
            }

            # Include Thompson Sampling stats if shadow engine is loaded
            shadow_mod = hub.modules.get("shadow_engine")
            if shadow_mod and hasattr(shadow_mod, "get_thompson_stats"):
                raw_stats = shadow_mod.get_thompson_stats()
                thompson = shadow_mod._thompson if hasattr(shadow_mod, "_thompson") else None
                # Wrap bucket stats with sampler config for frontend
                result["thompson_sampling"] = {
                    "buckets": {
                        k: {
                            "alpha": v["alpha"],
                            "beta": v["beta"],
                            "explore_rate": round(1.0 - v["mean"], 3),
                            "samples": v["trials"],
                        }
                        for k, v in raw_stats.items()
                    },
                    "strategy": "thompson",
                    "discount_factor": thompson.discount_factor if thompson else 0.95,
                    "window_size": thompson.window_size if thompson else 100,
                }

            return result
        except Exception:
            logger.exception("Error getting shadow accuracy")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/shadow/disagreements")
    async def get_shadow_disagreements(limit: int = Query(default=20, le=1000)):
        """Get top disagreements sorted by confidence (most informative first)."""
        try:
            predictions = await hub.cache.get_recent_predictions(limit=200, outcome_filter="disagreement")
            # Sort by confidence descending (highest confidence wrong = most informative)
            sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)[:limit]
            return {"disagreements": sorted_preds, "count": len(sorted_preds)}
        except Exception:
            logger.exception("Error getting shadow disagreements")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/shadow/propagation")
    async def get_shadow_propagation():
        """Get correction propagation stats from shadow engine."""
        try:
            shadow_mod = hub.modules.get("shadow_engine")
            if not shadow_mod or not hasattr(shadow_mod, "propagator"):
                return {"enabled": False, "stats": {}}
            prop = shadow_mod.propagator
            if prop is None:
                return {"enabled": False, "stats": {}}
            return {
                "enabled": True,
                "stats": {
                    "replay_buffer_size": len(prop._replay_buffer) if hasattr(prop, "_replay_buffer") else 0,
                    "replay_buffer_capacity": prop.buffer_size if hasattr(prop, "buffer_size") else 0,
                    "cell_observations": len(prop._cell_observations) if hasattr(prop, "_cell_observations") else 0,
                    "bandwidth": getattr(prop, "bandwidth", 1.0),
                },
            }
        except Exception:
            logger.exception("Error getting shadow propagation stats")
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_pipeline_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register pipeline status, advance, and retreat endpoints."""

    @router.get("/api/pipeline")
    async def get_pipeline_status():
        """Get current pipeline stage, gate progress, and multi-metric health.

        Multi-metric gates augment single-accuracy thresholds with:
        - coverage: fraction of contexts where predictions were attempted
        - confidence_calibration: how well confidence scores match actual accuracy
        - trend_direction: whether accuracy is improving, stable, or degrading
        """
        try:
            pipeline = await hub.cache.get_pipeline_state()
            if pipeline is None:
                return {
                    "current_stage": "shadow",
                    "gates": {},
                    "stage_health": {},
                    "message": "Pipeline state not yet initialized",
                }

            # Bridge: auto-populate backtest_accuracy from shadow accuracy
            # when in backtest stage. The backtest gate was designed for a holdout
            # evaluation pipeline that was never built — shadow accuracy is the
            # best available proxy for prediction quality during backtest stage.
            if pipeline.get("current_stage") == "backtest" and pipeline.get("backtest_accuracy") is None:
                try:
                    bridge_stats = await hub.cache.get_accuracy_stats()
                    shadow_acc = bridge_stats.get("overall_accuracy", 0) / 100.0
                    if shadow_acc > 0:
                        await hub.cache.update_pipeline_state(backtest_accuracy=round(shadow_acc, 4))
                        pipeline["backtest_accuracy"] = round(shadow_acc, 4)
                except Exception as e:
                    logger.warning("Failed to bridge backtest_accuracy from shadow stats: %s", e)

            # Compute multi-metric stage health from accuracy stats
            try:
                accuracy_stats = await hub.cache.get_accuracy_stats()
            except Exception:
                accuracy_stats = {}
            stage_health = _compute_stage_health(accuracy_stats)

            pipeline["stage_health"] = stage_health
            pipeline["gates"] = PIPELINE_GATES
            pipeline["stages"] = PIPELINE_STAGES
            return pipeline
        except Exception:
            logger.exception("Error getting pipeline status")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/pipeline/advance")
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
                current_value = pipeline.get(gate["field"]) or 0
                if current_value < gate["threshold"]:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Gate not met",
                            "gate": gate["label"],
                            "required": gate["threshold"],
                            "current": current_value,
                        },
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
        except Exception:
            logger.exception("Error advancing pipeline")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/pipeline/retreat")
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
        except Exception:
            logger.exception("Error retreating pipeline")
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _compute_stage_health(stats: dict) -> dict:
    """Compute multi-metric stage health from accuracy stats."""
    stage_health: dict[str, Any] = {}
    try:
        total = stats.get("total_resolved", 0)
        total_attempted = stats.get("total_attempted", total)

        # Coverage: predictions attempted / events eligible
        coverage = (total / max(total_attempted, 1)) if total > 0 else 0.0
        stage_health["coverage"] = round(coverage, 3)

        # Confidence calibration: compare mean confidence vs actual accuracy
        overall_acc = stats.get("overall_accuracy", 0) / 100.0  # normalize to 0-1
        # Calibration error = |mean_confidence - actual_accuracy|
        # Perfect calibration: predictions with 0.7 confidence are right 70% of the time
        mean_conf = stats.get("mean_confidence", overall_acc)
        calibration_error = abs(mean_conf - overall_acc)
        stage_health["confidence_calibration"] = round(1.0 - calibration_error, 3)

        # Trend direction from daily_trend
        trend = stats.get("daily_trend", [])
        if len(trend) >= 3:
            recent = [d.get("accuracy", 0) for d in trend[-3:]]
            earlier = [d.get("accuracy", 0) for d in trend[-6:-3]] if len(trend) >= 6 else recent
            recent_avg = sum(recent) / len(recent)
            earlier_avg = sum(earlier) / len(earlier)
            delta = recent_avg - earlier_avg
            if delta > 2:
                stage_health["trend_direction"] = "improving"
            elif delta < -2:
                stage_health["trend_direction"] = "degrading"
            else:
                stage_health["trend_direction"] = "stable"
            stage_health["trend_delta"] = round(delta, 2)
        else:
            stage_health["trend_direction"] = "insufficient_data"
            stage_health["trend_delta"] = 0.0

        stage_health["predictions_resolved"] = total
    except Exception as e:
        logger.warning("Failed to compute stage health metrics: %s", e)
        stage_health["error"] = "Could not compute stage health"

    return stage_health


def _register_config_routes(router: APIRouter, hub: IntelligenceHub, ws_manager: WebSocketManager) -> None:
    """Register config CRUD and history endpoints."""

    @router.get("/api/config")
    async def get_all_config():
        """Get all configuration parameters."""
        try:
            configs = await hub.cache.get_all_config()
            return {"configs": configs}
        except Exception:
            logger.exception("Error getting all config")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/config/reset/{key:path}")
    async def reset_config(key: str):
        """Reset a configuration parameter to its default value."""
        try:
            config = await hub.cache.reset_config(key)
            return config
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception("Error resetting config '%s'", key)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/config/{key:path}")
    async def get_config(key: str):
        """Get a single configuration parameter."""
        try:
            config = await hub.cache.get_config(key)
            if config is None:
                raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
            return config
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error getting config '%s'", key)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.put("/api/config/{key:path}")
    async def put_config(key: str, body: ConfigUpdate):
        """Update a configuration parameter value."""
        try:
            config = await hub.cache.set_config(key, body.value, changed_by=body.changed_by)
            await ws_manager.broadcast({"type": "config_updated", "data": {"key": key}})
            return config
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception("Error updating config '%s'", key)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/config-history")
    async def get_config_history(
        key: str | None = None,
        limit: int = Query(default=50, le=1000),
    ):
        """Get configuration change history."""
        try:
            history = await hub.cache.get_config_history(key=key, limit=limit)
            return {"history": history, "count": len(history)}
        except Exception:
            logger.exception("Error getting config history")
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_curation_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register curation CRUD and bulk update endpoints."""

    @router.get("/api/curation")
    async def get_all_curation():
        """Get all entity curation classifications."""
        try:
            curations = await hub.cache.get_all_curation()
            return {"curations": curations}
        except Exception:
            logger.exception("Error getting all curation")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.get("/api/curation/summary")
    async def get_curation_summary():
        """Get curation tier/status counts summary."""
        try:
            summary = await hub.cache.get_curation_summary()
            return summary
        except Exception:
            logger.exception("Error getting curation summary")
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.put("/api/curation/{entity_id:path}")
    async def put_curation(entity_id: str, body: CurationUpdate):
        """Override a single entity's curation classification."""
        try:
            result = await hub.cache.upsert_curation(
                entity_id, status=body.status, decided_by=body.decided_by, human_override=True
            )
            await hub.publish("curation_updated", {"entity_id": entity_id, "status": body.status})
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception:
            logger.exception("Error updating curation for '%s'", entity_id)
            raise HTTPException(status_code=500, detail="Internal server error") from None

    @router.post("/api/curation/bulk")
    async def bulk_update_curation(body: BulkCurationUpdate):
        """Bulk approve/reject entity curations."""
        try:
            count = await hub.cache.bulk_update_curation(
                body.entity_ids, status=body.status, decided_by=body.decided_by
            )
            await hub.publish("curation_updated", {"count": count, "status": body.status})
            return {"updated": count}
        except Exception:
            logger.exception("Error bulk updating curation")
            raise HTTPException(status_code=500, detail="Internal server error") from None


def _register_frigate_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register Frigate proxy endpoints for thumbnails and snapshots."""

    @router.get("/api/frigate/thumbnail/{event_id}")
    async def frigate_thumbnail(event_id: str):
        """Proxy a Frigate event thumbnail for the dashboard."""
        presence_mod = hub.modules.get("presence")
        if not presence_mod or not hasattr(presence_mod, "get_frigate_thumbnail"):
            raise HTTPException(status_code=503, detail="Presence module not loaded")
        data = await presence_mod.get_frigate_thumbnail(event_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        from fastapi.responses import Response

        return Response(content=data, media_type="image/jpeg")

    @router.get("/api/frigate/snapshot/{event_id}")
    async def frigate_snapshot(event_id: str):
        """Proxy a Frigate event snapshot for the dashboard."""
        presence_mod = hub.modules.get("presence")
        if not presence_mod or not hasattr(presence_mod, "get_frigate_snapshot"):
            raise HTTPException(status_code=503, detail="Presence module not loaded")
        data = await presence_mod.get_frigate_snapshot(event_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        from fastapi.responses import Response

        return Response(content=data, media_type="image/jpeg")


def create_api(hub: IntelligenceHub) -> FastAPI:
    """Create FastAPI application with hub routes.

    Args:
        hub: IntelligenceHub instance

    Returns:
        FastAPI application
    """
    from aria import __version__

    app = FastAPI(
        title="ARIA",
        description="REST API for ARIA — Adaptive Residence Intelligence Architecture",
        version=__version__,
    )

    ws_manager = WebSocketManager()

    # --- Request timing middleware ---
    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        hub._request_count += 1
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        if elapsed > 1.0:
            logger.warning(f"{request.method} {request.url.path} took {elapsed:.2f}s")
        else:
            logger.debug(f"{request.method} {request.url.path} took {elapsed:.3f}s")
        return response

    # Subscribe to hub events for WebSocket broadcasting
    async def broadcast_cache_update(data: dict[str, Any]):
        await ws_manager.broadcast({"type": "cache_updated", "data": data})

    hub.subscribe("cache_updated", broadcast_cache_update)

    # Mount SPA dashboard (serves index.html for all unmatched /ui/ paths)
    spa_dist = Path(__file__).parent.parent / "dashboard" / "spa" / "dist"
    app.mount("/ui", StaticFiles(directory=str(spa_dist), html=True), name="spa")

    # Authenticated router — all /api/* routes require API key when configured
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    # Health check (unauthenticated — useful for uptime monitors)
    @app.get("/")
    async def root():
        """API root - health check."""
        return {"status": "ok", "service": "ARIA"}

    @app.get("/health")
    async def health():
        """Detailed health check with module status and uptime."""
        try:
            health_data = await hub.health_check()
            return JSONResponse(content=health_data)
        except Exception:
            logger.exception("Health check failed")
            return JSONResponse(status_code=500, content={"status": "error", "error": "Health check failed"})

    # Register all route groups
    _register_utility_routes(router, hub, ws_manager)
    _register_cache_routes(router, hub)
    _register_ml_routes(router, hub)
    _register_discovery_routes(router, hub)
    _register_feedback_routes(router, hub)
    _register_shadow_routes(router, hub)
    _register_pipeline_routes(router, hub)
    _register_config_routes(router, hub, ws_manager)
    _register_curation_routes(router, hub)
    _register_frigate_routes(router, hub)

    # WebSocket endpoint (auth handled inline — FastAPI dependency injection
    # doesn't apply to websocket routes on the main app)
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time events."""
        # Check API key if configured
        if _ARIA_API_KEY:
            token = websocket.query_params.get("token")
            if token != _ARIA_API_KEY:
                await websocket.close(code=4003)
                return

        await ws_manager.connect(websocket)

        try:
            # Send initial connection message
            await websocket.send_json({"type": "connected", "message": "Connected to ARIA"})

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
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break

        finally:
            ws_manager.disconnect(websocket)

    # Include authenticated router
    app.include_router(router)

    return app
