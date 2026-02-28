"""Module source configuration routes.

Lets users toggle which data sources (signals, domains, entities) feed
each module.  Sources are stored as comma-separated config values using
the existing config system.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aria.hub.core import IntelligenceHub

logger = logging.getLogger(__name__)

# Valid modules and their config keys
MODULE_SOURCE_KEYS = {
    "presence": "presence.enabled_signals",
    "activity": "activity.enabled_domains",
    "anomaly": "anomaly.enabled_entities",
    "shadow": "shadow.enabled_capabilities",
    "discovery": "discovery.domain_filter",
}

MODULE_VALID_SOURCES = {
    "presence": {
        "camera_person",
        "camera_face",
        "motion",
        "light_interaction",
        "dimmer_press",
        "door",
        "media_active",
        "media_inactive",
        "device_tracker",
    },
    "activity": {"light", "switch", "binary_sensor", "media_player", "climate", "cover"},
    "anomaly": {"light", "binary_sensor", "climate", "media_player", "switch"},
    "shadow": {"light", "binary_sensor", "climate", "media_player"},
    "discovery": {"light", "switch", "binary_sensor", "media_player", "climate", "cover"},
}


class SourceUpdate(BaseModel):
    sources: list[str]


def _register_module_config_routes(router: APIRouter, hub: IntelligenceHub) -> None:
    """Register module source config endpoints on the router."""

    @router.get("/api/config/modules/{module}/sources")
    async def get_module_sources(module: str):
        """Get enabled data sources for a module."""
        if module not in MODULE_SOURCE_KEYS:
            raise HTTPException(404, f"Unknown module: {module}")
        config_key = MODULE_SOURCE_KEYS[module]
        config = await hub.cache.get_config(config_key)
        # get_config returns a dict with "value" key, or None
        raw = config.get("value", "") if config else ""
        sources = [s.strip() for s in (raw or "").split(",") if s.strip()]
        return {"module": module, "sources": sources, "config_key": config_key}

    @router.put("/api/config/modules/{module}/sources")
    async def put_module_sources(module: str, body: SourceUpdate):
        """Update enabled data sources for a module. At least one must remain."""
        if module not in MODULE_SOURCE_KEYS:
            raise HTTPException(404, f"Unknown module: {module}")
        if not body.sources:
            raise HTTPException(400, "At least one source must remain enabled")
        valid = MODULE_VALID_SOURCES.get(module, set())
        invalid = [s for s in body.sources if s not in valid]
        if invalid:
            raise HTTPException(400, f"Invalid sources: {invalid}")
        config_key = MODULE_SOURCE_KEYS[module]
        value = ",".join(body.sources)
        await hub.cache.set_config(config_key, value, changed_by="user")
        # Publish event bus notification so running modules react to live config changes
        # (Lesson #6 / #317 — config writes without publish() have no effect on running modules)
        await hub.publish("config_updated", {"module": module, "key": config_key, "value": value})
        logger.info("Module %s sources updated: %s", module, body.sources)
        return {"module": module, "sources": body.sources, "config_key": config_key}
