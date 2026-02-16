"""Organic Discovery Module — hub integration for automatic capability discovery.

Orchestrates the 5 core submodules (feature_vectors, clustering, seed_validation,
naming, scoring) and integrates with the ARIA hub lifecycle.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any

from aria.capabilities import Capability, CapabilityRegistry
from aria.hub.core import IntelligenceHub, Module
from aria.modules.organic_discovery.behavioral import cluster_behavioral
from aria.modules.organic_discovery.clustering import cluster_entities
from aria.modules.organic_discovery.feature_vectors import build_feature_matrix
from aria.modules.organic_discovery.naming import heuristic_description, heuristic_name
from aria.modules.organic_discovery.scoring import UsefulnessComponents, compute_usefulness
from aria.modules.organic_discovery.seed_validation import validate_seeds

logger = logging.getLogger(__name__)


DEFAULT_SETTINGS: dict[str, Any] = {
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
            description=(
                "Two-layer HDBSCAN clustering to discover capabilities"
                " from entity attributes and temporal co-occurrence."
            ),
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
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
        self.history: list[dict[str, Any]] = []

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
            self.logger.info(
                f"Loaded settings: autonomy_mode={self.settings['autonomy_mode']}, "
                f"naming_backend={self.settings['naming_backend']}"
            )

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

    async def update_settings(self, updates: dict) -> None:
        """Update discovery settings with validation and persist to cache."""
        VALID_BACKENDS = {"heuristic", "ollama"}
        VALID_MODES = {"suggest_and_wait", "auto_promote", "manual_only"}

        if "naming_backend" in updates and updates["naming_backend"] not in VALID_BACKENDS:
            raise ValueError(f"Invalid naming_backend: {updates['naming_backend']}. Must be one of {VALID_BACKENDS}")
        if "autonomy_mode" in updates and updates["autonomy_mode"] not in VALID_MODES:
            raise ValueError(f"Invalid autonomy_mode: {updates['autonomy_mode']}. Must be one of {VALID_MODES}")

        self.settings.update(updates)
        await self.hub.set_cache("discovery_settings", self.settings, {"source": "settings_update"})
        self.logger.info(f"Settings updated: {updates}")

    async def _periodic_run(self):
        """Wrapper for scheduled execution."""
        try:
            await self.run_discovery()
        except Exception as e:
            self.logger.error(f"Periodic organic discovery failed: {e}")

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events — drift_detected flags capabilities for re-discovery."""
        if event_type == "drift_detected":
            cap_name = data.get("capability", "")
            if not cap_name:
                return
            caps_entry = await self.hub.get_cache("capabilities")
            if not caps_entry or not caps_entry.get("data"):
                return
            caps = caps_entry["data"]
            if cap_name in caps:
                caps[cap_name]["drift_flagged"] = True
                caps[cap_name]["drift_detected_at"] = datetime.now().isoformat()
                caps[cap_name]["drift_severity"] = data.get("severity", 0.0)
                await self.hub.set_cache("capabilities", caps, {"source": "drift_flag"})
                self.logger.warning(f"Capability '{cap_name}' flagged for re-discovery due to drift")

    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    async def _name_cluster(self, cluster_info: dict) -> tuple[str, str]:
        """Name and describe a cluster using the configured backend."""
        backend = self.settings["naming_backend"]

        if backend == "ollama":
            from aria.modules.organic_discovery.naming import ollama_description, ollama_name

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

    async def run_discovery(self) -> dict[str, Any]:
        """Execute the full organic discovery pipeline.

        Returns:
            Run summary dict.
        """
        self.logger.info("Running organic discovery...")

        # 1. Read from cache
        entities, devices, seed_caps, activity_rates = await self._read_discovery_inputs()

        # 2-3. Build feature matrix and cluster
        clusters, entity_ids = self._build_and_cluster(entities, devices, activity_rates)

        # 4. Validate against seeds
        seed_validation = self._validate_against_seeds(seed_caps, clusters)

        # 5-6. Name and score domain clusters
        demand_signals = self._collect_demand_signals()
        total_entities = len(entity_ids) if entity_ids else 1
        organic_caps = await self._score_domain_clusters(
            clusters,
            entities,
            devices,
            seed_caps,
            activity_rates,
            demand_signals,
            total_entities,
        )

        # Phase 2: Behavioral clustering
        await self._score_behavioral_clusters(
            organic_caps,
            entities,
            devices,
            seed_caps,
            activity_rates,
            demand_signals,
            total_entities,
        )

        # 7-8. Merge and apply autonomy
        merged_caps = self._merge_capabilities(seed_caps, organic_caps)
        self._apply_autonomy(merged_caps)

        # 9. Persist and publish
        run_record = await self._persist_discovery_results(merged_caps, clusters, organic_caps, seed_validation)

        self.logger.info(
            f"Organic discovery complete: {len(clusters)} clusters, "
            f"{len(organic_caps)} organic caps, {len(merged_caps)} total"
        )
        return run_record

    async def _read_discovery_inputs(
        self,
    ) -> tuple[list, dict, dict[str, Any], dict[str, float]]:
        """Read entities, devices, seed capabilities, and activity rates from cache."""
        entities_raw = await self._get_cache_data("entities", default={})
        entities = list(entities_raw.values()) if isinstance(entities_raw, dict) else entities_raw
        devices = await self._get_cache_data("devices", default={})
        seed_caps = await self._get_cache_data("capabilities", default={})
        activity_data = await self._get_cache_data("activity_summary", default={})

        entity_activity = activity_data.get("entity_activity", {})
        activity_rates: dict[str, float] = {}
        for eid, info in entity_activity.items():
            if isinstance(info, dict):
                activity_rates[eid] = info.get("daily_avg_changes", 0.0)
            else:
                activity_rates[eid] = float(info) if info else 0.0

        return entities, devices, seed_caps, activity_rates

    def _build_and_cluster(
        self,
        entities: list,
        devices: dict,
        activity_rates: dict[str, float],
    ) -> tuple[list, list[str]]:
        """Build feature matrix and run HDBSCAN clustering."""
        if entities:
            matrix, entity_ids, _feature_names = build_feature_matrix(
                entities=entities,
                devices=devices,
                entity_registry={},
                activity_rates=activity_rates,
            )
        else:
            import numpy as np

            matrix = np.empty((0, 0))
            entity_ids = []

        clusters = []
        if matrix.shape[0] > 0:
            try:
                clusters = cluster_entities(matrix, entity_ids)
            except Exception as e:
                self.logger.warning(f"Clustering failed: {e}")

        return clusters, entity_ids

    def _validate_against_seeds(self, seed_caps: dict, clusters: list) -> dict:
        """Validate clusters against seed capabilities."""
        if not seed_caps or not clusters:
            return {}
        try:
            return validate_seeds(seed_caps, clusters)
        except Exception as e:
            self.logger.warning(f"Seed validation failed: {e}")
            return {}

    async def _score_domain_clusters(  # noqa: PLR0913 — pipeline context params
        self,
        clusters: list,
        entities: list,
        devices: dict,
        seed_caps: dict,
        activity_rates: dict[str, float],
        demand_signals: list,
        total_entities: int,
    ) -> dict[str, dict[str, Any]]:
        """Name and score domain-layer clusters into organic capabilities."""
        organic_caps: dict[str, dict[str, Any]] = {}
        today = str(date.today())

        for cluster in clusters:
            cap_entry = await self._build_capability_entry(
                cluster,
                entities,
                devices,
                seed_caps,
                activity_rates,
                demand_signals,
                total_entities,
                organic_caps,
                today,
            )
            if cap_entry:
                name, data = cap_entry
                organic_caps[name] = data

        return organic_caps

    async def _score_behavioral_clusters(  # noqa: PLR0913 — pipeline context params
        self,
        organic_caps: dict[str, dict[str, Any]],
        entities: list,
        devices: dict,
        seed_caps: dict,
        activity_rates: dict[str, float],
        demand_signals: list,
        total_entities: int,
    ) -> None:
        """Run behavioral clustering and add results to organic_caps in place."""
        logbook_entries = await self._load_logbook()
        if not logbook_entries:
            return

        behavioral_clusters = cluster_behavioral(logbook_entries)
        run_start = str(date.today())

        for cluster in behavioral_clusters:
            cluster_info_extra = {"temporal_pattern": cluster.get("temporal_pattern", {})}
            cap_entry = await self._build_capability_entry(
                cluster,
                entities,
                devices,
                seed_caps,
                activity_rates,
                demand_signals,
                total_entities,
                organic_caps,
                run_start,
                layer_override="behavioral",
                name_prefix="behavioral_",
                extra_info=cluster_info_extra,
                extra_fields={"temporal_pattern": cluster.get("temporal_pattern", {})},
            )
            if cap_entry:
                name, data = cap_entry
                organic_caps[name] = data

    async def _build_capability_entry(  # noqa: PLR0913 — cluster scoring needs full context
        self,
        cluster: dict,
        entities: list,
        devices: dict,
        seed_caps: dict,
        activity_rates: dict[str, float],
        demand_signals: list,
        total_entities: int,
        existing_caps: dict,
        first_seen: str,
        layer_override: str | None = None,
        name_prefix: str = "",
        extra_info: dict | None = None,
        extra_fields: dict | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Build a single capability entry from a cluster.

        Returns:
            (name, cap_data) tuple, or None on failure.
        """
        cluster_id = cluster["cluster_id"]
        member_ids = cluster["entity_ids"]
        silhouette = cluster.get("silhouette", 0.0)

        cluster_info = self._build_cluster_info(member_ids, entities, devices)
        cluster_info["entity_ids"] = member_ids
        if extra_info:
            cluster_info.update(extra_info)

        name, description = await self._name_cluster(cluster_info)

        # Ensure unique names
        if name in existing_caps or name in seed_caps:
            name = f"{name_prefix}{name}_{cluster_id}" if name_prefix else f"{name}_{cluster_id}"

        # Score
        avg_activity = 0.0
        if member_ids:
            rates = [activity_rates.get(eid, 0.0) for eid in member_ids]
            avg_activity = sum(rates) / len(rates)

        components = UsefulnessComponents(
            predictability=self._compute_predictability(name, seed_caps),
            stability=self._compute_stability(name),
            entity_coverage=len(member_ids) / total_entities,
            activity=min(avg_activity / 50.0, 1.0),
            cohesion=max(silhouette, 0.0),
        )
        usefulness = compute_usefulness(components)

        # Demand alignment bonus
        entity_lookup = {e.get("entity_id", ""): e for e in entities}
        cluster_entity_data = [entity_lookup[eid] for eid in member_ids if eid in entity_lookup]
        demand_bonus = self._compute_demand_alignment(cluster_entity_data, demand_signals)
        usefulness = int(min(usefulness + demand_bonus * 100, 100))

        layer = layer_override or self._classify_layer(cluster_info)
        cap_data: dict[str, Any] = {
            "available": True,
            "entities": member_ids,
            "total_count": len(member_ids),
            "can_predict": False,
            "source": "organic",
            "usefulness": usefulness,
            "usefulness_components": components.to_dict(),
            "layer": layer,
            "status": "candidate",
            "first_seen": first_seen,
            "promoted_at": None,
            "naming_method": self.settings["naming_backend"],
            "description": description,
            "stability_streak": self._count_streak(name),
        }
        if extra_fields:
            cap_data.update(extra_fields)

        return name, cap_data

    @staticmethod
    def _merge_capabilities(
        seed_caps: dict[str, Any],
        organic_caps: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Merge seed and organic capabilities, seeds always preserved."""
        merged: dict[str, dict[str, Any]] = {}
        for seed_name, seed_data in seed_caps.items():
            merged[seed_name] = {
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
        for cap_name, cap_data in organic_caps.items():
            if cap_name not in merged:
                merged[cap_name] = cap_data
        return merged

    async def _persist_discovery_results(
        self,
        merged_caps: dict,
        clusters: list,
        organic_caps: dict,
        seed_validation: dict,
    ) -> dict[str, Any]:
        """Write merged capabilities to cache, record history, publish event."""
        await self.hub.set_cache(
            "capabilities",
            merged_caps,
            {"count": len(merged_caps), "source": "organic_discovery"},
        )

        run_record = {
            "timestamp": str(date.today()),
            "clusters_found": len(clusters),
            "organic_caps": list(organic_caps.keys()),
            "seed_validation": seed_validation,
            "total_merged": len(merged_caps),
        }
        self.history.append(run_record)
        self.history = self.history[-90:]
        await self.hub.set_cache("discovery_history", self.history)
        await self.hub.publish("organic_discovery_complete", run_record)
        return run_record

    # ------------------------------------------------------------------
    # Demand alignment
    # ------------------------------------------------------------------

    def _compute_demand_alignment(self, cluster_entities: list, demands: list) -> float:
        """Score how well a cluster aligns with consumer demand signals (0.0-0.2 bonus)."""
        if not demands:
            return 0.0
        best_score = 0.0
        domains = {e.get("domain", "") for e in cluster_entities if isinstance(e, dict)}
        device_classes = {e.get("device_class", "") for e in cluster_entities if isinstance(e, dict)}
        count = len(cluster_entities)
        for demand in demands:
            domain_match = bool(set(demand.entity_domains) & domains) if demand.entity_domains else True
            class_match = bool(set(demand.device_classes) & device_classes) if demand.device_classes else True
            size_match = count >= demand.min_entities
            if domain_match and class_match and size_match:
                best_score = max(best_score, 0.2)
            elif domain_match and class_match:
                best_score = max(best_score, 0.1)
            elif domain_match:
                best_score = max(best_score, 0.05)
        return best_score

    def _collect_demand_signals(self) -> list:
        """Collect all demand signals from registered capabilities."""
        try:
            registry = CapabilityRegistry()
            registry.collect_from_modules()
            signals = []
            for cap in registry._caps.values():
                signals.extend(cap.demand_signals)
            return signals
        except Exception:
            return []

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
        member_ids: list[str],
        entities: list[dict[str, Any]],
        devices: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build metadata dict for naming from member entity IDs."""
        entity_lookup = {e.get("entity_id", ""): e for e in entities}

        domains: dict[str, int] = {}
        areas: dict[str, int] = {}
        device_classes: dict[str, int] = {}

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

    def _classify_layer(self, cluster_info: dict[str, Any]) -> str:
        """Classify a cluster as 'domain' or 'behavioral' layer.

        Single-domain clusters are 'domain'. Multi-domain are 'behavioral'.
        """
        domains = cluster_info.get("domains", {})
        if len(domains) <= 1:
            return "domain"
        return "behavioral"

    def _compute_predictability(self, cap_name: str, existing_caps: dict) -> float:
        """Compute predictability from ML + shadow feedback signals.

        Blends ML accuracy (mean_r2, weight 0.7) with shadow accuracy
        (hit_rate, weight 0.3) from the capabilities cache.
        """
        caps = existing_caps.get("data", existing_caps) if isinstance(existing_caps, dict) else {}
        existing = caps.get(cap_name, {})
        ml_r2 = existing.get("ml_accuracy", {}).get("mean_r2", 0.0)
        shadow_hr = existing.get("shadow_accuracy", {}).get("hit_rate", 0.0)
        if ml_r2 + shadow_hr == 0:
            return 0.0
        return ml_r2 * 0.7 + shadow_hr * 0.3

    def _compute_stability(self, cap_name: str) -> float:
        """Compute stability score (0-1) from history.

        Stability = fraction of recent runs where this capability appeared.
        """
        if not self.history:
            return 0.0

        recent = self.history[-14:]  # last 14 runs
        appeared = sum(1 for h in recent if cap_name in h.get("organic_caps", []))
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

    def _apply_autonomy(self, caps: dict[str, dict[str, Any]]) -> None:
        """Apply autonomy rules to update status of organic capabilities in-place."""
        mode = self.settings["autonomy_mode"]
        promote_threshold = self.settings["promote_threshold"]
        archive_threshold = self.settings["archive_threshold"]
        promote_streak = self.settings["promote_streak_days"]
        archive_streak = self.settings["archive_streak_days"]

        for _name, cap in caps.items():
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
