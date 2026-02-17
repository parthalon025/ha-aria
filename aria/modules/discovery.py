"""Discovery Module - HA entity and capability detection.

Wraps the standalone discover.py script and integrates it with the hub.
Runs discovery on a schedule and stores results in hub cache.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp

from aria.capabilities import Capability
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)


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
    ]

    def __init__(self, hub: IntelligenceHub, ha_url: str, ha_token: str):
        """Initialize discovery module.

        Args:
            hub: IntelligenceHub instance
            ha_url: Home Assistant URL (e.g., "http://192.168.1.35:8123")
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

        data = existing_entry["data"]
        ttl_hours_str = await self.hub.cache.get_config_value("discovery.stale_ttl_hours", "72")
        ttl_hours = int(ttl_hours_str)
        now = datetime.utcnow()
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
        now = datetime.utcnow()

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
        retry_delay = 5

        while self.hub.is_running():
            try:
                retry_delay = await self._ws_registry_session(ws_url, retry_delay)
            except (TimeoutError, aiohttp.ClientError) as e:
                self.logger.warning(f"HA WebSocket error: {e} — retrying in {retry_delay}s")
            except Exception as e:
                self.logger.error(f"HA WebSocket unexpected error: {e}")

            # Backoff: 5s → 10s → 20s → 60s max
            await asyncio.sleep(retry_delay)
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
