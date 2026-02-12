"""Discovery Module - HA entity and capability detection.

Wraps the standalone discover.py script and integrates it with the hub.
Runs discovery on a schedule and stores results in hub cache.
"""

import os
import sys
import json
import subprocess
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta

import aiohttp

from hub.core import Module, IntelligenceHub


logger = logging.getLogger(__name__)


class DiscoveryModule(Module):
    """Discovers HA entities, devices, areas, and capabilities."""

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
        self.discover_script = Path(__file__).parent.parent / "bin" / "discover.py"

        if not self.discover_script.exists():
            raise FileNotFoundError(f"Discovery script not found: {self.discover_script}")

    async def initialize(self):
        """Initialize module - run initial discovery."""
        self.logger.info("Discovery module initializing...")

        # Run initial discovery
        try:
            await self.run_discovery()
            self.logger.info("Initial discovery complete")
        except Exception as e:
            self.logger.error(f"Initial discovery failed: {e}")

    async def run_discovery(self) -> Dict[str, Any]:
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
                env={
                    **os.environ,
                    "HA_URL": self.ha_url,
                    "HA_TOKEN": self.ha_token
                }
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

    async def _store_discovery_results(self, capabilities: Dict[str, Any]):
        """Store discovery results in hub cache.

        Stores separate cache entries for:
        - entities: Entity registry data
        - devices: Device registry data
        - areas: Area registry data
        - capabilities: Detected capabilities
        - discovery_metadata: Discovery run metadata
        """
        # Store entities
        entities = capabilities.get("entities", {})
        if entities:
            await self.hub.set_cache("entities", entities, {
                "count": len(entities),
                "source": "discovery"
            })

        # Store devices
        devices = capabilities.get("devices", {})
        if devices:
            await self.hub.set_cache("devices", devices, {
                "count": len(devices),
                "source": "discovery"
            })

        # Store areas
        areas = capabilities.get("areas", {})
        if areas:
            await self.hub.set_cache("areas", areas, {
                "count": len(areas),
                "source": "discovery"
            })

        # Store capabilities
        caps = capabilities.get("capabilities", {})
        if caps:
            await self.hub.set_cache("capabilities", caps, {
                "count": len(caps),
                "source": "discovery"
            })

        # Store metadata
        metadata = {
            "entity_count": capabilities.get("entity_count", 0),
            "device_count": capabilities.get("device_count", 0),
            "area_count": capabilities.get("area_count", 0),
            "capability_count": len(caps),
            "timestamp": capabilities.get("timestamp"),
            "ha_version": capabilities.get("ha_version")
        }
        await self.hub.set_cache("discovery_metadata", metadata)

    async def on_event(self, event_type: str, data: Dict[str, Any]):
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
        self._debounce_task: Optional[asyncio.Task] = None
        self._debounce_seconds = 30

        async def _listen():
            ws_url = self.ha_url.replace("http", "ws", 1) + "/api/websocket"
            retry_delay = 5

            while self.hub.is_running():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.ws_connect(ws_url) as ws:
                            # 1. Wait for auth_required
                            msg = await ws.receive_json()
                            if msg.get("type") != "auth_required":
                                self.logger.error(f"Unexpected WS message: {msg}")
                                continue

                            # 2. Authenticate
                            await ws.send_json({
                                "type": "auth",
                                "access_token": self.ha_token,
                            })
                            auth_resp = await ws.receive_json()
                            if auth_resp.get("type") != "auth_ok":
                                self.logger.error(f"WS auth failed: {auth_resp}")
                                await asyncio.sleep(retry_delay)
                                continue

                            self.logger.info("HA WebSocket connected — listening for registry changes")
                            retry_delay = 5  # reset backoff

                            # 3. Subscribe to events
                            cmd_id = 1
                            for evt in self._registry_events:
                                await ws.send_json({
                                    "id": cmd_id,
                                    "type": "subscribe_events",
                                    "event_type": evt,
                                })
                                cmd_id += 1

                            # 4. Listen loop
                            async for msg in ws:
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    if data.get("type") == "event":
                                        evt_type = (
                                            data.get("event", {})
                                            .get("event_type", "")
                                        )
                                        if evt_type in self._registry_events:
                                            self._schedule_debounced_discovery(evt_type)
                                elif msg.type in (
                                    aiohttp.WSMsgType.CLOSED,
                                    aiohttp.WSMsgType.ERROR,
                                ):
                                    break

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.logger.warning(f"HA WebSocket error: {e} — retrying in {retry_delay}s")
                except Exception as e:
                    self.logger.error(f"HA WebSocket unexpected error: {e}")

                # Backoff: 5s → 10s → 20s → 60s max
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

        # Run listener as a background hub task
        await self.hub.schedule_task(
            task_id="discovery_ws_listener",
            coro=_listen,
            interval=None,  # one-shot (loop is internal)
            run_immediately=True,
        )
        self.logger.info("Started HA event listener for registry changes")

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
            run_immediately=False  # Initial discovery already done
        )

        self.logger.info(f"Scheduled periodic discovery every {interval_hours} hours")
