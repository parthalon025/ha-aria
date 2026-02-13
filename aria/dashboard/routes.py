# DEPRECATED: Replaced by SPA in dashboard/spa/ (mounted at /ui via StaticFiles in hub/api.py)
# Kept for reference until SPA is fully validated. Safe to delete after validation.
"""Dashboard routes for HA Intelligence Hub web UI (DEPRECATED)."""

import logging
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aria.hub.core import IntelligenceHub


logger = logging.getLogger(__name__)

# Setup templates
dashboard_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(dashboard_dir / "templates"))


def create_dashboard_router(hub: IntelligenceHub) -> APIRouter:
    """Create dashboard router with hub instance.

    Args:
        hub: IntelligenceHub instance

    Returns:
        FastAPI router for dashboard
    """
    router = APIRouter(prefix="/ui", tags=["dashboard"])

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page - hub status and module health."""
        try:
            health = await hub.health_check()
            categories = await hub.cache.list_categories()
            recent_events = await hub.cache.get_events(limit=10)

            return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    "health": health,
                    "categories": categories,
                    "recent_events": recent_events
                }
            )
        except Exception as e:
            logger.error(f"Error rendering home page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/discovery", response_class=HTMLResponse)
    async def discovery(request: Request):
        """Discovery page - entity and device browser."""
        try:
            # Get discovery data from cache
            entities_cache = await hub.get_cache("entities")
            devices_cache = await hub.get_cache("devices")
            areas_cache = await hub.get_cache("areas")
            capabilities_cache = await hub.get_cache("capabilities")

            entities = entities_cache["data"] if entities_cache else {}
            devices = devices_cache["data"] if devices_cache else {}
            areas = areas_cache["data"] if areas_cache else {}
            capabilities = capabilities_cache["data"] if capabilities_cache else {}

            # Pre-compute domain breakdown and slim entity data for frontend
            domain_counts = {}
            unavailable_count = 0
            area_entity_counts = {}
            slim_entities = {}
            for eid, edata in entities.items():
                domain = edata.get("domain", eid.split(".")[0])
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if edata.get("state") in ("unavailable", "unknown"):
                    unavailable_count += 1
                aid = edata.get("area_id")
                if aid:
                    area_entity_counts[aid] = area_entity_counts.get(aid, 0) + 1

                # Only pass fields the template actually uses (strip attributes blob)
                slim_entities[eid] = {
                    "state": edata.get("state", ""),
                    "friendly_name": edata.get("friendly_name", ""),
                    "domain": domain,
                    "device_class": edata.get("device_class"),
                    "area_id": aid,
                    "device_id": edata.get("device_id"),
                    "unit_of_measurement": edata.get("unit_of_measurement"),
                }

            # Slim device data — only fields the template uses
            slim_devices = {}
            for did, ddata in devices.items():
                slim_devices[did] = {
                    "name": ddata.get("name", ""),
                    "manufacturer": ddata.get("manufacturer"),
                    "model": ddata.get("model"),
                    "area_id": ddata.get("area_id"),
                }

            domain_breakdown = sorted(
                [{"domain": d, "count": c} for d, c in domain_counts.items()],
                key=lambda x: -x["count"]
            )

            return templates.TemplateResponse(
                "discovery.html",
                {
                    "request": request,
                    "entities": slim_entities,
                    "devices": slim_devices,
                    "areas": areas,
                    "capabilities": capabilities,
                    "entity_count": len(entities),
                    "device_count": len(devices),
                    "area_count": len(areas),
                    "capability_count": len(capabilities),
                    "unavailable_count": unavailable_count,
                    "domain_breakdown": domain_breakdown,
                    "area_entity_counts": area_entity_counts,
                }
            )
        except Exception as e:
            logger.error(f"Error rendering discovery page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/capabilities", response_class=HTMLResponse)
    async def capabilities(request: Request):
        """Capabilities page - capability list with entity counts."""
        try:
            # Get capabilities from cache
            capabilities_cache = await hub.get_cache("capabilities")

            capabilities = capabilities_cache["data"] if capabilities_cache else {}

            return templates.TemplateResponse(
                "capabilities.html",
                {
                    "request": request,
                    "capabilities": capabilities,
                    "capability_count": len(capabilities)
                }
            )
        except Exception as e:
            logger.error(f"Error rendering capabilities page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/predictions", response_class=HTMLResponse)
    async def predictions(request: Request):
        """Predictions page - ML predictions with confidence."""
        try:
            # Get predictions from cache
            predictions_cache = await hub.get_cache("ml_predictions")

            predictions = predictions_cache["data"] if predictions_cache else {}

            return templates.TemplateResponse(
                "predictions.html",
                {
                    "request": request,
                    "predictions": predictions
                }
            )
        except Exception as e:
            logger.error(f"Error rendering predictions page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/patterns", response_class=HTMLResponse)
    async def patterns(request: Request):
        """Patterns page - detected patterns with LLM descriptions."""
        try:
            # Get patterns from cache
            patterns_cache = await hub.get_cache("patterns")

            patterns = patterns_cache["data"] if patterns_cache else {}

            return templates.TemplateResponse(
                "patterns.html",
                {
                    "request": request,
                    "patterns": patterns
                }
            )
        except Exception as e:
            logger.error(f"Error rendering patterns page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/automations", response_class=HTMLResponse)
    async def automations(request: Request):
        """Automations page - automation suggestions with approve/reject."""
        try:
            # Get automation suggestions from cache
            automations_cache = await hub.get_cache("automation_suggestions")

            automations = automations_cache["data"] if automations_cache else {}

            return templates.TemplateResponse(
                "automations.html",
                {
                    "request": request,
                    "automations": automations
                }
            )
        except Exception as e:
            logger.error(f"Error rendering automations page: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return router


# Export for convenience (will be initialized with hub instance)
router = None
