"""Discovery Module - HA entity and capability detection.

Wraps the standalone discover.py script and integrates it with the hub.
Runs discovery on a schedule and stores results in hub cache.
"""

import asyncio
import json
import logging
import os
import random
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

from aria.capabilities import Capability
from aria.hub.constants import CACHE_ACTIVITY_LOG, CACHE_ENTITIES, RECONNECT_STAGGER
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)

# --- Entity classification config (merged from data_quality) ---
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
PRESENCE_DOMAINS = {"person", "device_tracker"}
RECLASSIFY_INTERVAL = timedelta(hours=24)


class DiscoveryModule(Module):
    """Discovers HA entities, devices, areas, and capabilities."""

    CAPABILITIES = [
        Capability(
            id="discovery",
            name="HA Entity Discovery",
            description="Scans HA for entities, devices, areas, and seed capabilities via REST + WebSocket.",
            module="discovery",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_discover.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=[],
        ),
        Capability(
            id="data_quality",
            name="Data Quality",
            description="Entity classification pipeline — auto-exclude, edge cases, default include.",
            module="discovery",
            layer="hub",
            config_keys=[],
            test_paths=["tests/hub/test_data_quality.py"],
            systemd_units=["aria-hub.service"],
            status="stable",
            added_version="1.0.0",
            depends_on=["discovery"],
        ),
    ]

    def __init__(self, hub: IntelligenceHub, ha_url: str, ha_token: str):
        """Initialize discovery module.

        Args:
            hub: IntelligenceHub instance
            ha_url: Home Assistant URL (e.g., "http://<ha-host>:8123")
            ha_token: Home Assistant long-lived access token
        """
        super().__init__("discovery", hub)
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.discover_script = Path(__file__).parent.parent.parent / "bin" / "discover.py"

        if not self.discover_script.exists():
            raise FileNotFoundError(f"Discovery script not found: {self.discover_script}")

    async def initialize(self):
        """Initialize module - run initial discovery and schedule archive checks."""
        self.logger.info("Discovery module initializing...")

        # Run initial discovery
        try:
            await self.run_discovery()
            self.logger.info("Initial discovery complete")
        except Exception as e:
            self.logger.error(f"Initial discovery failed: {e}")

        # Schedule periodic archive expiry check (every 6 hours)
        async def _check_archives():
            for cache_key in ("entities", "devices", "areas"):
                try:
                    await self._archive_expired_entities(cache_key)
                except Exception as e:
                    self.logger.warning(f"Archive check failed for {cache_key}: {e}")

        await self.hub.schedule_task(
            task_id="discovery_archive_check",
            coro=_check_archives,
            interval=timedelta(hours=6),
            run_immediately=False,
        )

        # Run entity classification (merged from data_quality module)
        try:
            await self.run_classification()
        except Exception as e:
            self.logger.warning(f"Initial entity classification failed: {e}")

        await self.hub.schedule_task(
            task_id="data_quality_reclassify",
            coro=self.run_classification,
            interval=RECLASSIFY_INTERVAL,
            run_immediately=False,
        )

    async def run_discovery(self) -> dict[str, Any]:
        """Run discovery script and store results in hub cache.

        Returns:
            Discovery results dictionary
        """
        self.logger.info("Running discovery...")

        try:
            # Run discover.py subprocess
            result = subprocess.run(
                [sys.executable, str(self.discover_script)],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "HA_URL": self.ha_url, "HA_TOKEN": self.ha_token},
            )

            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                raise RuntimeError(f"Discovery failed: {error_msg}")

            # Parse JSON output
            capabilities = json.loads(result.stdout)

            # Store in hub cache
            await self._store_discovery_results(capabilities)

            self.logger.info(
                f"Discovery complete: {capabilities.get('entity_count', 0)} entities, "
                f"{len(capabilities.get('capabilities', {}))} capabilities"
            )

            return capabilities

        except subprocess.TimeoutExpired:
            self.logger.error("Discovery timed out after 120 seconds")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse discovery output: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Discovery error: {e}")
            raise

    async def _merge_with_lifecycle(self, cache_key: str, new_items: dict[str, Any], now: datetime) -> dict[str, Any]:
        """Merge new discovery items with existing cache, tracking lifecycle state.

        For each item:
        - In new discovery: set active, update fields, preserve first_discovered,
          clear stale_since/archived_at.
        - In old cache but NOT new: active→stale (set stale_since).
          Already stale/archived→keep as-is (don't re-stamp).

        Returns:
            Merged dict with lifecycle metadata on every item.
        """
        now_iso = now.isoformat()

        # Load existing cache
        existing_entry = await self.hub.get_cache(cache_key)
        existing = {}
        if existing_entry and existing_entry.get("data"):
            existing = existing_entry["data"]

        merged: dict[str, Any] = {}

        # Process items present in new discovery
        for item_id, item_data in new_items.items():
            old = existing.get(item_id, {})
            old_lc = old.get("_lifecycle", {})

            entry = dict(item_data)
            entry["_lifecycle"] = {
                "status": "active",
                "first_discovered": old_lc.get("first_discovered", now_iso),
                "last_seen_in_discovery": now_iso,
                "stale_since": None,
                "archived_at": None,
            }
            merged[item_id] = entry

        # Process items in old cache but NOT in new discovery
        for item_id, item_data in existing.items():
            if item_id in merged:
                continue  # already handled above

            old_lc = item_data.get("_lifecycle", {})
            status = old_lc.get("status", "active")

            entry = dict(item_data)
            if status == "active":
                # Transition active → stale
                entry["_lifecycle"] = {
                    **old_lc,
                    "status": "stale",
                    "stale_since": now_iso,
                }
            # Already stale or archived — keep as-is (don't re-stamp)
            merged[item_id] = entry

        return merged

    async def _archive_expired_entities(self, cache_key: str):
        """Archive stale entities that have exceeded the stale TTL.

        Reads discovery.stale_ttl_hours config (default 72). For each stale
        entity, if stale_since exceeds the TTL, transitions to archived.
        Only writes cache if something changed.
        """
        existing_entry = await self.hub.get_cache(cache_key)
        if not existing_entry or not existing_entry.get("data"):
            return

        import copy

        data = copy.deepcopy(existing_entry["data"])

        ttl_hours_str = await self.hub.cache.get_config_value("discovery.stale_ttl_hours", "72")
        try:
            ttl_hours = float(ttl_hours_str)
            if not (0 <= ttl_hours <= 720):
                ttl_hours = 72
        except (ValueError, TypeError):
            ttl_hours = 72

        now = datetime.now()
        changed = False

        for _item_id, item_data in data.items():
            lc = item_data.get("_lifecycle", {})
            if lc.get("status") != "stale":
                continue

            stale_since_str = lc.get("stale_since")
            if not stale_since_str:
                continue

            stale_since = datetime.fromisoformat(stale_since_str)
            if (now - stale_since) > timedelta(hours=ttl_hours):
                lc["status"] = "archived"
                lc["archived_at"] = now.isoformat()
                changed = True

        if changed:
            await self.hub.set_cache(cache_key, data, {"source": "lifecycle_archive"})

    async def _store_discovery_results(self, capabilities: dict[str, Any]):
        """Store discovery results in hub cache with lifecycle tracking.

        Stores separate cache entries for:
        - entities: Entity registry data (lifecycle-aware merge)
        - devices: Device registry data (lifecycle-aware merge)
        - areas: Area registry data (lifecycle-aware merge)
        - capabilities: Detected capabilities (organic preservation)
        - discovery_metadata: Discovery run metadata
        """
        now = datetime.now()

        # Store entities with lifecycle merge
        entities = capabilities.get("entities", {})
        if entities:
            merged = await self._merge_with_lifecycle("entities", entities, now)
            await self.hub.set_cache("entities", merged, {"count": len(merged), "source": "discovery"})

        # Store devices with lifecycle merge
        devices = capabilities.get("devices", {})
        if devices:
            merged = await self._merge_with_lifecycle("devices", devices, now)
            await self.hub.set_cache("devices", merged, {"count": len(merged), "source": "discovery"})

        # Store areas with lifecycle merge
        areas = capabilities.get("areas", {})
        if areas:
            merged = await self._merge_with_lifecycle("areas", areas, now)
            await self.hub.set_cache("areas", merged, {"count": len(merged), "source": "discovery"})

        # Store capabilities — merge with existing to preserve organic discoveries
        caps = capabilities.get("capabilities", {})
        if caps:
            existing_entry = await self.hub.get_cache("capabilities")
            if existing_entry and existing_entry.get("data"):
                existing = existing_entry["data"]
                # Preserve organic capabilities that seed discovery doesn't know about
                for name, cap_data in existing.items():
                    if cap_data.get("source") == "organic" and name not in caps:
                        caps[name] = cap_data
            await self.hub.set_cache("capabilities", caps, {"count": len(caps), "source": "discovery"})

        # Store metadata
        metadata = {
            "entity_count": capabilities.get("entity_count", 0),
            "device_count": capabilities.get("device_count", 0),
            "area_count": capabilities.get("area_count", 0),
            "capability_count": len(caps),
            "timestamp": capabilities.get("timestamp"),
            "ha_version": capabilities.get("ha_version"),
        }
        await self.hub.set_cache("discovery_metadata", metadata)

        # Notify consumers (e.g., PresenceModule refreshes camera mapping)
        await self.hub.publish("discovery_complete", metadata)

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events."""
        pass

    # ------------------------------------------------------------------
    # Event-driven discovery via HA WebSocket
    # ------------------------------------------------------------------

    async def start_event_listener(self):
        """Connect to HA WebSocket and listen for registry changes.

        Subscribes to entity_registry_updated, device_registry_updated,
        and area_registry_updated events. Debounces: waits 30s after
        the last event before triggering re-discovery.
        """
        # Registry event types that indicate entity/device changes
        self._registry_events = {
            "entity_registry_updated",
            "device_registry_updated",
            "area_registry_updated",
        }
        self._debounce_task: asyncio.Task | None = None
        self._debounce_seconds = 30

        # Run listener as a background hub task
        await self.hub.schedule_task(
            task_id="discovery_ws_listener",
            coro=self._ws_registry_listener,
            interval=None,  # one-shot (loop is internal)
            run_immediately=True,
        )
        self.logger.info("Started HA event listener for registry changes")

    async def _ws_registry_listener(self):
        """WebSocket listener loop for registry change events."""
        ws_url = self.ha_url.replace("http", "ws", 1) + "/api/websocket"
        stagger = RECONNECT_STAGGER.get("discovery", 0)
        retry_delay = 5
        first_connect = True

        while self.hub.is_running():
            try:
                retry_delay = await self._ws_registry_session(ws_url, retry_delay)
            except (TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning(f"HA WebSocket error: {e} — retrying in {retry_delay}s")
            except Exception as e:
                self.logger.error(f"HA WebSocket unexpected error: {e}")

            # Apply stagger on first reconnect attempt to avoid thundering herd
            base_delay = retry_delay + (stagger if first_connect else 0)
            first_connect = False

            # Backoff: 5s → 10s → 20s → 60s max, ±25% jitter
            jitter = base_delay * random.uniform(-0.25, 0.25)
            actual_delay = base_delay + jitter
            await asyncio.sleep(actual_delay)
            retry_delay = min(retry_delay * 2, 60)

    async def _ws_registry_session(self, ws_url: str, retry_delay: int) -> int:
        """Run a single WebSocket session for registry events.

        Returns:
            Updated retry_delay (reset to 5 on successful auth).
        """
        async with aiohttp.ClientSession() as session, session.ws_connect(ws_url) as ws:
            # 1. Wait for auth_required
            msg = await ws.receive_json()
            if msg.get("type") != "auth_required":
                self.logger.error(f"Unexpected WS message: {msg}")
                return retry_delay

            # 2. Authenticate
            await ws.send_json({"type": "auth", "access_token": self.ha_token})
            auth_resp = await ws.receive_json()
            if auth_resp.get("type") != "auth_ok":
                self.logger.error(f"WS auth failed: {auth_resp}")
                await asyncio.sleep(retry_delay)
                return retry_delay

            self.logger.info("HA WebSocket connected — listening for registry changes")
            retry_delay = 5  # reset backoff

            # 3. Subscribe to events
            cmd_id = 1
            for evt in self._registry_events:
                await ws.send_json({"id": cmd_id, "type": "subscribe_events", "event_type": evt})
                cmd_id += 1

            # 4. Listen loop
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "event":
                        evt_type = data.get("event", {}).get("event_type", "")
                        if evt_type in self._registry_events:
                            self._schedule_debounced_discovery(evt_type)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

        return retry_delay

    def _schedule_debounced_discovery(self, event_type: str):
        """Debounce registry events — wait 30s after last event before re-running."""
        self.logger.info(f"Registry event: {event_type} — debouncing {self._debounce_seconds}s")

        # Cancel previous debounce timer
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        async def _delayed_discovery():
            await asyncio.sleep(self._debounce_seconds)
            self.logger.info("Debounce expired — running event-triggered discovery")
            try:
                await self.run_discovery()
            except Exception as e:
                self.logger.error(f"Event-triggered discovery failed: {e}")

        self._debounce_task = asyncio.create_task(_delayed_discovery())

    async def schedule_periodic_discovery(self, interval_hours: int = 24):
        """Schedule periodic discovery runs.

        Args:
            interval_hours: Hours between discovery runs
        """

        async def discovery_task():
            try:
                await self.run_discovery()
            except Exception as e:
                self.logger.error(f"Scheduled discovery failed: {e}")

        await self.hub.schedule_task(
            task_id="discovery_periodic",
            coro=discovery_task,
            interval=timedelta(hours=interval_hours),
            run_immediately=False,  # Initial discovery already done
        )

        self.logger.info(f"Scheduled periodic discovery every {interval_hours} hours")

    # ------------------------------------------------------------------
    # Entity classification (merged from data_quality module)
    # ------------------------------------------------------------------

    async def _read_config_thresholds(self) -> dict[str, Any]:
        """Read classification thresholds from config store."""
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

        return {
            "auto_exclude_domains": auto_exclude_domains,
            "noise_event_threshold": noise_threshold,
            "stale_days_threshold": stale_days,
            "vehicle_patterns": vehicle_patterns,
            "unavailable_grace_hours": unavailable_grace_hours,
        }

    @staticmethod
    def _build_vehicle_sets(
        entities_data: dict[str, Any],
        vehicle_patterns: list[str],
    ) -> tuple[set[str], set[str]]:
        """Build vehicle entity IDs and device IDs from entity data."""
        vehicle_entity_ids = set()
        for eid, edata in entities_data.items():
            name = (edata.get("friendly_name") or eid).lower()
            if any(pat in name for pat in vehicle_patterns):
                vehicle_entity_ids.add(eid)

        vehicle_device_ids = set()
        for eid in vehicle_entity_ids:
            did = entities_data[eid].get("device_id")
            if did:
                vehicle_device_ids.add(did)

        return vehicle_entity_ids, vehicle_device_ids

    async def run_classification(self):
        """Main classification pipeline: read cache, compute metrics, classify, upsert."""
        self.logger.info("Starting entity classification...")

        entities_entry = await self.hub.get_cache(CACHE_ENTITIES)
        entities_data = {}
        if entities_entry and entities_entry.get("data"):
            entities_data = entities_entry["data"]

        if not entities_data:
            self.logger.warning("No entity data in cache — skipping classification")
            return

        activity_entry = await self.hub.get_cache(CACHE_ACTIVITY_LOG)
        activity_windows = []
        if activity_entry and activity_entry.get("data"):
            activity_windows = activity_entry["data"].get("windows", [])

        config_thresholds = await self._read_config_thresholds()
        vehicle_entity_ids, vehicle_device_ids = self._build_vehicle_sets(
            entities_data, config_thresholds["vehicle_patterns"]
        )

        classified = 0
        skipped = 0

        for entity_id, entity_data in entities_data.items():
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
                decided_by="discovery",
            )
            classified += 1

        self.logger.info(f"Classification complete: {classified} classified, {skipped} human-override skipped")

    def _compute_metrics(
        self,
        entity_id: str,
        entity_data: dict[str, Any],
        activity_windows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute per-entity metrics from entity data and activity windows."""
        domain = entity_data.get("domain", entity_id.split(".")[0] if "." in entity_id else "")
        area_id = entity_data.get("area_id", "")
        device_class = entity_data.get("device_class", "")

        total_events = 0
        total_window_seconds = 0
        unique_states = set()

        for window in activity_windows:
            by_entity = window.get("by_entity", {})
            if entity_id in by_entity:
                total_events += by_entity[entity_id]
            total_window_seconds += 900  # 15-min windows
            for event in window.get("events", []):
                if event.get("entity_id") == entity_id:
                    to_state = event.get("to")
                    if to_state is not None:
                        unique_states.add(to_state)

        event_rate_day = (total_events / total_window_seconds) * 86400 if total_window_seconds > 0 else 0.0

        last_changed_days_ago = None
        last_changed = entity_data.get("last_changed")
        if last_changed:
            try:
                changed_dt = datetime.fromisoformat(last_changed.replace("Z", "+00:00"))
                now = datetime.now(UTC).replace(tzinfo=None)
                changed_naive = changed_dt.replace(tzinfo=None)
                last_changed_days_ago = (now - changed_naive).total_seconds() / 86400
            except (ValueError, TypeError):
                pass

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

    def _classify(  # noqa: PLR0913, PLR0911
        self,
        entity_id: str,
        metrics: dict[str, Any],
        config_thresholds: dict[str, Any],
        entities_data: dict[str, Any],
        vehicle_entity_ids: set,
        vehicle_device_ids: set,
    ) -> tuple[int, str, str, str]:
        """Classify entity into tier, status, reason, group_id."""
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
        if domain in auto_exclude_domains:
            return (1, "auto_excluded", f"Domain '{domain}' is infrastructure", "")

        if stale_days is not None and stale_days > stale_threshold:
            return (1, "auto_excluded", f"No state changes in {int(stale_days)} days", "")

        unavailable_hours = metrics.get("unavailable_since_hours")
        if unavailable_grace_hours and unavailable_hours is not None and unavailable_hours > unavailable_grace_hours:
            return (
                1,
                "auto_excluded",
                f"Unavailable for {int(unavailable_hours)}h (grace: {unavailable_grace_hours}h)",
                "",
            )

        if event_rate > noise_threshold and unique_states < 3:
            return (
                1,
                "auto_excluded",
                f"Polling noise ({int(event_rate)}/day, {unique_states} states)",
                "",
            )

        name = (entities_data.get(entity_id, {}).get("friendly_name") or entity_id).lower()
        for pat in vehicle_patterns:
            if pat and pat in name:
                return (1, "auto_excluded", f"Matches vehicle pattern '{pat}'", "")

        # --- Tier 2: edge cases ---
        device_id = entities_data.get(entity_id, {}).get("device_id", "")
        if device_id and device_id in vehicle_device_ids and entity_id not in vehicle_entity_ids:
            return (2, "excluded", "Part of vehicle device group", device_id)

        if event_rate > 500 and unique_states < 5:
            return (
                2,
                "excluded",
                f"High event rate with low variety ({int(event_rate)}/day, {unique_states} states)",
                "",
            )

        if domain in PRESENCE_DOMAINS:
            return (2, "included", "Presence tracking", "")

        # --- Tier 3: default include ---
        return (3, "included", "General entity", "")
