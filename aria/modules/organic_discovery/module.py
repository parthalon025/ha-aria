"""Organic Discovery Module — hub integration for automatic capability discovery.

Orchestrates the 5 core submodules (feature_vectors, clustering, seed_validation,
naming, scoring) and integrates with the ARIA hub lifecycle.
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List

from aria.hub.core import Module, IntelligenceHub
from aria.capabilities import Capability
from aria.modules.organic_discovery.feature_vectors import build_feature_matrix
from aria.modules.organic_discovery.clustering import cluster_entities
from aria.modules.organic_discovery.seed_validation import validate_seeds
from aria.modules.organic_discovery.naming import heuristic_name, heuristic_description
from aria.modules.organic_discovery.scoring import compute_usefulness, UsefulnessComponents
from aria.modules.organic_discovery.behavioral import cluster_behavioral

logger = logging.getLogger(__name__)


DEFAULT_SETTINGS: Dict[str, Any] = {
    "autonomy_mode": "suggest_and_wait",
    "naming_backend": "heuristic",
    "promote_threshold": 50,
    "archive_threshold": 10,
    "promote_streak_days": 7,
    "archive_streak_days": 14,
}


class OrganicDiscoveryModule(Module):
    """Discovers capabilities organically by clustering HA entities."""

    CAPABILITIES = [
        Capability(
            id="organic_discovery",
            name="Organic Capability Discovery",
            description="Two-layer HDBSCAN clustering to discover capabilities from entity attributes and temporal co-occurrence.",
            module="organic_discovery",
            layer="hub",
            config_keys=[],
            test_paths=[
                "tests/hub/test_organic_discovery_module.py",
                "tests/hub/test_organic_clustering.py",
                "tests/hub/test_organic_behavioral.py",
                "tests/hub/test_organic_feature_vectors.py",
                "tests/hub/test_organic_naming.py",
                "tests/hub/test_organic_scoring.py",
                "tests/hub/test_organic_seed_validation.py",
                "tests/hub/test_api_organic_discovery.py",
            ],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub):
        super().__init__("organic_discovery", hub)
        self.settings: Dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        """Load settings and history from cache, schedule periodic runs."""
        self.logger.info("Organic discovery module initializing...")

        # Load persisted settings
        settings_entry = await self.hub.get_cache("discovery_settings")
        if settings_entry and settings_entry.get("data"):
            loaded = settings_entry["data"]
            # Merge with defaults so new keys are always present
            merged = dict(DEFAULT_SETTINGS)
            merged.update(loaded)
            self.settings = merged
            self.logger.info(f"Loaded settings: autonomy_mode={self.settings['autonomy_mode']}")

        # Load persisted history
        history_entry = await self.hub.get_cache("discovery_history")
        if history_entry and history_entry.get("data"):
            self.history = history_entry["data"]
            self.logger.info(f"Loaded {len(self.history)} history records")

        # Schedule periodic discovery (every 6 hours)
        await self.hub.schedule_task(
            task_id="organic_discovery_periodic",
            coro=self._periodic_run,
            interval=timedelta(hours=6),
            run_immediately=False,
        )

        self.logger.info("Organic discovery module initialized")

    async def _periodic_run(self):
        """Wrapper for scheduled execution."""
        try:
            await self.run_discovery()
        except Exception as e:
            self.logger.error(f"Periodic organic discovery failed: {e}")

    async def on_event(self, event_type: str, data: Dict[str, Any]):
        """Handle hub events (no-op for now)."""
        pass

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    async def _name_cluster(self, cluster_info: dict) -> tuple[str, str]:
        """Name and describe a cluster using the configured backend."""
        backend = self.settings["naming_backend"]

        if backend == "ollama":
            from aria.modules.organic_discovery.naming import ollama_name, ollama_description
            name = await ollama_name(cluster_info)
            description = await ollama_description(cluster_info)
        else:
            # heuristic (default and fallback)
            name = heuristic_name(cluster_info)
            description = heuristic_description(cluster_info)

        return name, description

    # ------------------------------------------------------------------
    # Discovery pipeline
    # ------------------------------------------------------------------

    async def run_discovery(self) -> Dict[str, Any]:
        """Execute the full organic discovery pipeline.

        1. Read entities, devices, capabilities, activity from cache
        2. Build feature matrix
        3. Cluster with HDBSCAN
        4. Validate against seed capabilities
        5. Name clusters
        6. Score usefulness
        7. Merge organic + seed capabilities
        8. Apply autonomy rules
        9. Write to cache and record history

        Returns:
            Run summary dict.
        """
        self.logger.info("Running organic discovery...")

        # 1. Read from cache
        entities_raw = await self._get_cache_data("entities", default={})
        # Cache stores entities as {entity_id: dict} — convert to list for feature_vectors
        if isinstance(entities_raw, dict):
            entities = list(entities_raw.values())
        else:
            entities = entities_raw
        devices = await self._get_cache_data("devices", default={})
        seed_caps = await self._get_cache_data("capabilities", default={})
        activity_data = await self._get_cache_data("activity_summary", default={})

        entity_activity = activity_data.get("entity_activity", {})

        # Build activity rates lookup
        activity_rates: Dict[str, float] = {}
        for eid, info in entity_activity.items():
            if isinstance(info, dict):
                activity_rates[eid] = info.get("daily_avg_changes", 0.0)
            else:
                activity_rates[eid] = float(info) if info else 0.0

        # 2. Build feature matrix
        if entities:
            matrix, entity_ids, feature_names = build_feature_matrix(
                entities=entities,
                devices=devices,
                entity_registry={},
                activity_rates=activity_rates,
            )
        else:
            import numpy as np
            matrix = np.empty((0, 0))
            entity_ids = []

        # 3. Cluster (cluster_entities handles small-input edge cases internally)
        clusters = []
        if matrix.shape[0] > 0:
            try:
                clusters = cluster_entities(matrix, entity_ids)
            except Exception as e:
                self.logger.warning(f"Clustering failed: {e}")

        # 4. Validate against seeds
        seed_validation = {}
        if seed_caps and clusters:
            try:
                seed_validation = validate_seeds(seed_caps, clusters)
            except Exception as e:
                self.logger.warning(f"Seed validation failed: {e}")

        # 5-6. Name and score each cluster
        organic_caps: Dict[str, Dict[str, Any]] = {}
        total_entities = len(entity_ids) if entity_ids else 1

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            member_ids = cluster["entity_ids"]
            silhouette = cluster.get("silhouette", 0.0)

            # Build cluster_info for naming
            cluster_info = self._build_cluster_info(member_ids, entities, devices)
            cluster_info["entity_ids"] = member_ids

            # Name
            name, description = await self._name_cluster(cluster_info)

            # Ensure unique names
            if name in organic_caps or name in seed_caps:
                name = f"{name}_{cluster_id}"

            # Score
            avg_activity = 0.0
            if member_ids:
                rates = [activity_rates.get(eid, 0.0) for eid in member_ids]
                avg_activity = sum(rates) / len(rates)

            components = UsefulnessComponents(
                predictability=0.0,  # no ML model yet for organic clusters
                stability=self._compute_stability(name),
                entity_coverage=len(member_ids) / total_entities,
                activity=min(avg_activity / 50.0, 1.0),  # normalize: 50 changes/day = 1.0
                cohesion=max(silhouette, 0.0),
            )
            usefulness = compute_usefulness(components)

            today = str(date.today())
            organic_caps[name] = {
                "available": True,
                "entities": member_ids,
                "total_count": len(member_ids),
                "can_predict": False,
                "source": "organic",
                "usefulness": usefulness,
                "usefulness_components": components.to_dict(),
                "layer": self._classify_layer(cluster_info),
                "status": "candidate",
                "first_seen": today,
                "promoted_at": None,
                "naming_method": self.settings["naming_backend"],
                "description": description,
                "stability_streak": self._count_streak(name),
            }

        # Phase 2: Behavioral clustering
        run_start = str(date.today())
        logbook_entries = await self._load_logbook()
        if logbook_entries:
            behavioral_clusters = cluster_behavioral(logbook_entries)
            for cluster in behavioral_clusters:
                cluster_id = cluster["cluster_id"]
                member_ids = cluster["entity_ids"]

                cluster_info = self._build_cluster_info(member_ids, entities, devices)
                cluster_info["entity_ids"] = member_ids
                cluster_info["temporal_pattern"] = cluster.get("temporal_pattern", {})

                name, description = await self._name_cluster(cluster_info)

                # Avoid name collision with domain capabilities or seeds
                if name in organic_caps or name in seed_caps:
                    name = f"behavioral_{name}_{cluster_id}"

                # Score
                avg_activity = 0.0
                if member_ids:
                    rates = [activity_rates.get(eid, 0.0) for eid in member_ids]
                    avg_activity = sum(rates) / len(rates)

                silhouette = cluster.get("silhouette", 0.0)
                components = UsefulnessComponents(
                    predictability=0.0,
                    stability=self._compute_stability(name),
                    entity_coverage=len(member_ids) / total_entities,
                    activity=min(avg_activity / 50.0, 1.0),
                    cohesion=max(silhouette, 0.0),
                )
                usefulness = compute_usefulness(components)

                organic_caps[name] = {
                    "available": True,
                    "entities": member_ids,
                    "total_count": len(member_ids),
                    "can_predict": False,
                    "source": "organic",
                    "usefulness": usefulness,
                    "usefulness_components": components.to_dict(),
                    "layer": "behavioral",
                    "status": "candidate",
                    "first_seen": run_start,
                    "promoted_at": None,
                    "naming_method": self.settings["naming_backend"],
                    "description": description,
                    "stability_streak": self._count_streak(name),
                    "temporal_pattern": cluster.get("temporal_pattern", {}),
                }

        # 7. Merge: seeds always preserved
        merged_caps: Dict[str, Dict[str, Any]] = {}

        # Add seeds first with canonical fields
        for seed_name, seed_data in seed_caps.items():
            merged_caps[seed_name] = {
                "available": seed_data.get("available", True),
                "entities": seed_data.get("entities", []),
                "total_count": seed_data.get("total_count", len(seed_data.get("entities", []))),
                "can_predict": seed_data.get("can_predict", False),
                "source": "seed",
                "usefulness": seed_data.get("usefulness", 100),
                "usefulness_components": seed_data.get("usefulness_components", {}),
                "layer": seed_data.get("layer", "domain"),
                "status": "promoted",
                "first_seen": seed_data.get("first_seen", str(date.today())),
                "promoted_at": seed_data.get("promoted_at", str(date.today())),
                "naming_method": "seed",
                "description": seed_data.get("description", ""),
                "stability_streak": seed_data.get("stability_streak", 0),
            }

        # Add organic capabilities
        for cap_name, cap_data in organic_caps.items():
            if cap_name not in merged_caps:
                merged_caps[cap_name] = cap_data

        # 8. Apply autonomy rules
        self._apply_autonomy(merged_caps)

        # 9. Write to cache
        await self.hub.set_cache(
            "capabilities",
            merged_caps,
            {"count": len(merged_caps), "source": "organic_discovery"},
        )

        # Record history
        run_record = {
            "timestamp": str(date.today()),
            "clusters_found": len(clusters),
            "organic_caps": list(organic_caps.keys()),
            "seed_validation": seed_validation,
            "total_merged": len(merged_caps),
        }
        self.history.append(run_record)

        # Persist history (keep last 90 entries)
        self.history = self.history[-90:]
        await self.hub.set_cache("discovery_history", self.history)

        # Publish event
        await self.hub.publish("organic_discovery_complete", run_record)

        self.logger.info(
            f"Organic discovery complete: {len(clusters)} clusters, "
            f"{len(organic_caps)} organic caps, {len(merged_caps)} total"
        )

        return run_record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_logbook(self, days: int = 14) -> list[dict]:
        """Load recent logbook entries from ~/ha-logs/ JSON files."""
        import json
        from pathlib import Path

        log_dir = Path.home() / "ha-logs"
        entries = []
        today = date.today()

        for i in range(days):
            day = today - timedelta(days=i)
            log_file = log_dir / f"{day.isoformat()}.json"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        day_entries = json.load(f)
                    if isinstance(day_entries, list):
                        entries.extend(day_entries)
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.warning(f"Failed to read {log_file}: {e}")

        self.logger.info(f"Loaded {len(entries)} logbook entries from {days} days")
        return entries

    async def _get_cache_data(self, key: str, default: Any = None) -> Any:
        """Get data from cache, returning default if not found."""
        entry = await self.hub.get_cache(key)
        if entry and "data" in entry:
            return entry["data"]
        return default if default is not None else {}

    def _build_cluster_info(
        self,
        member_ids: List[str],
        entities: List[Dict[str, Any]],
        devices: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build metadata dict for naming from member entity IDs."""
        entity_lookup = {e.get("entity_id", ""): e for e in entities}

        domains: Dict[str, int] = {}
        areas: Dict[str, int] = {}
        device_classes: Dict[str, int] = {}

        for eid in member_ids:
            entity = entity_lookup.get(eid, {})

            domain = entity.get("domain", "")
            if domain:
                domains[domain] = domains.get(domain, 0) + 1

            dc = entity.get("device_class")
            if dc:
                device_classes[dc] = device_classes.get(dc, 0) + 1

            # Resolve area through device
            area = entity.get("area_id")
            if not area:
                device_id = entity.get("device_id")
                if device_id and device_id in devices:
                    area = devices[device_id].get("area_id")
            if area:
                areas[area] = areas.get(area, 0) + 1

        return {
            "domains": domains,
            "areas": areas,
            "device_classes": device_classes,
        }

    def _classify_layer(self, cluster_info: Dict[str, Any]) -> str:
        """Classify a cluster as 'domain' or 'behavioral' layer.

        Single-domain clusters are 'domain'. Multi-domain are 'behavioral'.
        """
        domains = cluster_info.get("domains", {})
        if len(domains) <= 1:
            return "domain"
        return "behavioral"

    def _compute_stability(self, cap_name: str) -> float:
        """Compute stability score (0-1) from history.

        Stability = fraction of recent runs where this capability appeared.
        """
        if not self.history:
            return 0.0

        recent = self.history[-14:]  # last 14 runs
        appeared = sum(
            1 for h in recent
            if cap_name in h.get("organic_caps", [])
        )
        return appeared / len(recent)

    def _count_streak(self, cap_name: str) -> int:
        """Count consecutive most-recent runs where this capability appeared."""
        streak = 0
        for h in reversed(self.history):
            if cap_name in h.get("organic_caps", []):
                streak += 1
            else:
                break
        return streak

    def _apply_autonomy(self, caps: Dict[str, Dict[str, Any]]) -> None:
        """Apply autonomy rules to update status of organic capabilities in-place."""
        mode = self.settings["autonomy_mode"]
        promote_threshold = self.settings["promote_threshold"]
        archive_threshold = self.settings["archive_threshold"]
        promote_streak = self.settings["promote_streak_days"]
        archive_streak = self.settings["archive_streak_days"]

        for name, cap in caps.items():
            if cap.get("source") == "seed":
                continue  # seeds are always promoted

            usefulness = cap.get("usefulness", 0)
            streak = cap.get("stability_streak", 0)

            if mode == "suggest_and_wait":
                # Never auto-promote — everything stays as candidate
                pass

            elif mode == "auto_promote":
                # Promote at >= threshold for >= streak_days
                if usefulness >= promote_threshold and streak >= promote_streak:
                    cap["status"] = "promoted"
                    cap["promoted_at"] = str(date.today())
                # Archive at <= archive_threshold for >= archive_streak_days
                elif usefulness <= archive_threshold and streak >= archive_streak:
                    cap["status"] = "archived"

            elif mode == "autonomous":
                # Promote at >= 30 (lower bar)
                if usefulness >= 30:
                    cap["status"] = "promoted"
                    cap["promoted_at"] = str(date.today())
                # Archive at <= archive_threshold
                elif usefulness <= archive_threshold:
                    cap["status"] = "archived"
