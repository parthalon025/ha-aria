# aria/shared/entity_graph.py
"""Centralized entity→device→area graph.

Single source of truth for resolving the HA three-tier hierarchy:
  entity → device → area

Replaces per-module resolution logic in discovery, presence, and
snapshot collector. Updated from discovery cache on cache_updated events.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EntityGraph:
    """In-memory entity→device→area graph built from discovery cache."""

    def __init__(self):
        self._entities: dict[str, dict[str, Any]] = {}
        self._devices: dict[str, dict[str, Any]] = {}
        self._areas: list[dict[str, Any]] = []
        self._area_index: dict[str, list[dict[str, Any]]] = {}  # area_id → [entities]

    def update(
        self,
        entities: dict[str, dict[str, Any]],
        devices: dict[str, dict[str, Any]],
        areas: list[dict[str, Any]],
    ) -> None:
        """Rebuild the graph from fresh discovery data."""
        self._entities = dict(entities)
        self._devices = dict(devices)
        self._areas = list(areas)
        self._rebuild_area_index()
        logger.debug(
            "EntityGraph updated: %d entities, %d devices, %d areas",
            len(entities),
            len(devices),
            len(areas),
        )

    def _rebuild_area_index(self) -> None:
        """Rebuild the area→entities reverse index."""
        self._area_index = {}
        for entity_id, entity in self._entities.items():
            area_id = self._resolve_area(entity)
            if area_id:
                self._area_index.setdefault(area_id, []).append({**entity, "entity_id": entity_id})

    def _resolve_area(self, entity: dict[str, Any]) -> str | None:
        """Resolve area for an entity: direct area_id, then device fallback."""
        # Entity-level area takes priority
        if entity.get("area_id") is not None:
            return entity["area_id"]
        # Fall back to device's area
        device_id = entity.get("device_id")
        if device_id and device_id in self._devices:
            return self._devices[device_id].get("area_id")
        return None

    def get_area(self, entity_id: str) -> str | None:
        """Get area_id for an entity (entity→device→area chain)."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        return self._resolve_area(entity)

    def get_device(self, entity_id: str) -> dict[str, Any] | None:
        """Get device info for an entity."""
        entity = self._entities.get(entity_id)
        if not entity:
            return None
        device_id = entity.get("device_id")
        return self._devices.get(device_id) if device_id else None

    def entities_in_area(self, area_id: str) -> list[dict[str, Any]]:
        """Get all entities in an area."""
        return list(self._area_index.get(area_id, []))

    def entities_by_domain(self, domain: str) -> list[dict[str, Any]]:
        """Get all entities of a specific domain."""
        return [{**e, "entity_id": eid} for eid, e in self._entities.items() if eid.startswith(f"{domain}.")]

    def has_entity(self, entity_id: str) -> bool:
        """Check whether an entity exists in the graph."""
        return entity_id in self._entities

    def all_areas(self) -> list[dict[str, Any]]:
        """Get all known areas."""
        return list(self._areas)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def device_count(self) -> int:
        return len(self._devices)

    @property
    def area_count(self) -> int:
        return len(self._areas)
