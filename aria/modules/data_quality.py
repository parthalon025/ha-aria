"""Data Quality Module - Entity classification pipeline.

Reads discovery cache, computes per-entity metrics, classifies into
three tiers (auto-exclude, edge cases, default include), and writes
results to the entity_curation table. Runs on startup and daily.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from aria.hub.core import Module, IntelligenceHub
from aria.hub.constants import CACHE_ENTITIES, CACHE_ACTIVITY_LOG
from aria.capabilities import Capability

logger = logging.getLogger(__name__)

# Config keys with fallback defaults
CONFIG_AUTO_EXCLUDE_DOMAINS = "curation.auto_exclude_domains"
DEFAULT_AUTO_EXCLUDE_DOMAINS = (
    "update,tts,stt,scene,button,number,select,"
    "input_boolean,input_number,input_select,input_text,input_datetime,"
    "counter,script,zone,sun,weather,conversation,event,automation,camera,image,remote"
)

CONFIG_NOISE_EVENT_THRESHOLD = "curation.noise_event_threshold"
DEFAULT_NOISE_EVENT_THRESHOLD = 1000

CONFIG_STALE_DAYS_THRESHOLD = "curation.stale_days_threshold"
DEFAULT_STALE_DAYS_THRESHOLD = 30

CONFIG_UNAVAILABLE_GRACE_HOURS = "curation.unavailable_grace_hours"
DEFAULT_UNAVAILABLE_GRACE_HOURS = 0

CONFIG_VEHICLE_PATTERNS = "curation.vehicle_patterns"
DEFAULT_VEHICLE_PATTERNS = "tesla,luda,tessy,vehicle,car_"

# Presence-tracking domains always included at tier 2
PRESENCE_DOMAINS = {"person", "device_tracker"}

# Re-classification interval
RECLASSIFY_INTERVAL = timedelta(hours=24)


class DataQualityModule(Module):
    """Entity classification pipeline: metrics → tiers → curation table."""

    CAPABILITIES = [
        Capability(
            id="data_quality",
            name="Data Quality",
            description="Entity classification pipeline — auto-exclude, edge cases, default include.",
            module="data_quality",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_data_quality.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub):
        super().__init__("data_quality", hub)

    async def initialize(self):
        """Run initial classification and schedule daily re-run."""
        self.logger.info("Data quality module initializing...")
        await self.run_classification()
        await self.hub.schedule_task(
            task_id="data_quality_reclassify",
            coro=self.run_classification,
            interval=RECLASSIFY_INTERVAL,
            run_immediately=False,
        )
        self.logger.info("Data quality module initialized")

    async def run_classification(self):
        """Main pipeline: read cache → compute metrics → classify → upsert."""
        self.logger.info("Starting entity classification...")

        # Read entity data
        entities_entry = await self.hub.get_cache(CACHE_ENTITIES)
        entities_data = {}
        if entities_entry and entities_entry.get("data"):
            entities_data = entities_entry["data"]

        if not entities_data:
            self.logger.warning("No entity data in cache — skipping classification")
            return

        # Read activity data
        activity_entry = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        activity_windows = []
        if activity_entry and activity_entry.get("data"):
            activity_windows = activity_entry["data"].get("windows", [])

        # Read config thresholds
        auto_exclude_str = await self.hub.cache.get_config_value(
            CONFIG_AUTO_EXCLUDE_DOMAINS, DEFAULT_AUTO_EXCLUDE_DOMAINS
        )
        auto_exclude_domains = {d.strip() for d in auto_exclude_str.split(",")}

        noise_threshold = await self.hub.cache.get_config_value(
            CONFIG_NOISE_EVENT_THRESHOLD, DEFAULT_NOISE_EVENT_THRESHOLD
        )

        stale_days = await self.hub.cache.get_config_value(CONFIG_STALE_DAYS_THRESHOLD, DEFAULT_STALE_DAYS_THRESHOLD)

        vehicle_patterns_str = await self.hub.cache.get_config_value(CONFIG_VEHICLE_PATTERNS, DEFAULT_VEHICLE_PATTERNS)
        vehicle_patterns = [p.strip().lower() for p in vehicle_patterns_str.split(",")]

        unavailable_grace_hours = await self.hub.cache.get_config_value(
            CONFIG_UNAVAILABLE_GRACE_HOURS, DEFAULT_UNAVAILABLE_GRACE_HOURS
        )

        config_thresholds = {
            "auto_exclude_domains": auto_exclude_domains,
            "noise_event_threshold": noise_threshold,
            "stale_days_threshold": stale_days,
            "vehicle_patterns": vehicle_patterns,
            "unavailable_grace_hours": unavailable_grace_hours,
        }

        # Build vehicle entity set for group detection
        vehicle_entity_ids = set()
        for eid, edata in entities_data.items():
            name = (edata.get("friendly_name") or eid).lower()
            if any(pat in name for pat in vehicle_patterns):
                vehicle_entity_ids.add(eid)

        # Build device_id → entity_ids map
        device_entities: Dict[str, List[str]] = {}
        for eid, edata in entities_data.items():
            did = edata.get("device_id")
            if did:
                device_entities.setdefault(did, []).append(eid)

        # Find device_ids that contain a vehicle entity
        vehicle_device_ids = set()
        for eid in vehicle_entity_ids:
            did = entities_data[eid].get("device_id")
            if did:
                vehicle_device_ids.add(did)

        classified = 0
        skipped = 0

        for entity_id, entity_data in entities_data.items():
            # Check for human override
            existing = await self.hub.cache.get_curation(entity_id)
            if existing and existing.get("human_override"):
                skipped += 1
                continue

            metrics = self._compute_metrics(entity_id, entity_data, activity_windows)
            tier, status, reason, group_id = self._classify(
                entity_id,
                metrics,
                config_thresholds,
                entities_data,
                vehicle_entity_ids,
                vehicle_device_ids,
            )

            await self.hub.cache.upsert_curation(
                entity_id=entity_id,
                status=status,
                tier=tier,
                reason=reason,
                auto_classification=f"tier{tier}_{status}",
                metrics=metrics,
                group_id=group_id,
                decided_by="data_quality",
            )
            classified += 1

        self.logger.info(f"Classification complete: {classified} classified, {skipped} human-override skipped")

    def _compute_metrics(
        self,
        entity_id: str,
        entity_data: Dict[str, Any],
        activity_windows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute per-entity metrics from entity data and activity windows.

        Returns:
            Dict with event_rate_day, unique_states, last_changed_days_ago,
            domain, area_id, device_class.
        """
        domain = entity_data.get("domain", entity_id.split(".")[0] if "." in entity_id else "")
        area_id = entity_data.get("area_id", "")
        device_class = entity_data.get("device_class", "")

        # Aggregate event counts from activity windows
        total_events = 0
        total_window_seconds = 0
        unique_states = set()

        for window in activity_windows:
            by_entity = window.get("by_entity", {})
            if entity_id in by_entity:
                total_events += by_entity[entity_id]

            # Accumulate window duration
            total_window_seconds += 900  # 15-min windows

            # Collect unique states from events
            for event in window.get("events", []):
                if event.get("entity_id") == entity_id:
                    to_state = event.get("to")
                    if to_state is not None:
                        unique_states.add(to_state)

        # Scale to daily rate
        if total_window_seconds > 0:
            event_rate_day = (total_events / total_window_seconds) * 86400
        else:
            event_rate_day = 0.0

        # Last changed days ago
        last_changed_days_ago = None
        last_changed = entity_data.get("last_changed")
        if last_changed:
            try:
                changed_dt = datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
                # Use naive UTC comparison
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                changed_naive = changed_dt.replace(tzinfo=None)
                last_changed_days_ago = (now - changed_naive).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

        # Compute hours unavailable (only when current state is "unavailable")
        unavailable_since_hours = None
        if entity_data.get("state") == "unavailable" and last_changed_days_ago is not None:
            unavailable_since_hours = round(last_changed_days_ago * 24, 1)

        return {
            "event_rate_day": round(event_rate_day, 1),
            "unique_states": len(unique_states),
            "last_changed_days_ago": round(last_changed_days_ago, 1) if last_changed_days_ago is not None else None,
            "unavailable_since_hours": unavailable_since_hours,
            "domain": domain,
            "area_id": area_id,
            "device_class": device_class,
        }

    def _classify(
        self,
        entity_id: str,
        metrics: Dict[str, Any],
        config_thresholds: Dict[str, Any],
        entities_data: Dict[str, Any],
        vehicle_entity_ids: set,
        vehicle_device_ids: set,
    ) -> Tuple[int, str, str, str]:
        """Classify entity into tier, status, reason, group_id.

        Returns:
            (tier, status, reason, group_id) tuple.
        """
        domain = metrics.get("domain", "")
        event_rate = metrics.get("event_rate_day", 0)
        unique_states = metrics.get("unique_states", 0)
        stale_days = metrics.get("last_changed_days_ago")

        auto_exclude_domains = config_thresholds["auto_exclude_domains"]
        noise_threshold = config_thresholds["noise_event_threshold"]
        stale_threshold = config_thresholds["stale_days_threshold"]
        vehicle_patterns = config_thresholds["vehicle_patterns"]
        unavailable_grace_hours = config_thresholds.get("unavailable_grace_hours", 0)

        # --- Tier 1: auto-excluded ---

        # Domain in auto-exclude list
        if domain in auto_exclude_domains:
            return (1, "auto_excluded", f"Domain '{domain}' is infrastructure", "")

        # No changes in stale_days_threshold
        if stale_days is not None and stale_days > stale_threshold:
            return (1, "auto_excluded", f"No state changes in {int(stale_days)} days", "")

        # Unavailable beyond grace period
        unavailable_hours = metrics.get("unavailable_since_hours")
        if unavailable_grace_hours and unavailable_hours is not None and unavailable_hours > unavailable_grace_hours:
            return (1, "auto_excluded", f"Unavailable for {int(unavailable_hours)}h (grace: {unavailable_grace_hours}h)", "")

        # Polling noise: high rate + very few unique states
        if event_rate > noise_threshold and unique_states < 3:
            return (
                1,
                "auto_excluded",
                f"Polling noise ({int(event_rate)}/day, {unique_states} states)",
                "",
            )

        # Name matches vehicle patterns
        name = (entities_data.get(entity_id, {}).get("friendly_name") or entity_id).lower()
        for pat in vehicle_patterns:
            if pat and pat in name:
                return (1, "auto_excluded", f"Matches vehicle pattern '{pat}'", "")

        # --- Tier 2: edge cases ---

        # Shares device_id with a vehicle entity
        device_id = entities_data.get(entity_id, {}).get("device_id", "")
        if device_id and device_id in vehicle_device_ids and entity_id not in vehicle_entity_ids:
            return (2, "excluded", "Part of vehicle device group", device_id)

        # High rate with low variety (moderate noise)
        if event_rate > 500 and unique_states < 5:
            return (
                2,
                "excluded",
                f"High event rate with low variety ({int(event_rate)}/day, {unique_states} states)",
                "",
            )

        # Presence tracking domains
        if domain in PRESENCE_DOMAINS:
            return (2, "included", "Presence tracking", "")

        # --- Tier 3: default include ---
        return (3, "included", "General entity", "")
