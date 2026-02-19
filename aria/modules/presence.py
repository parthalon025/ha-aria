"""Presence Module - Real-time room presence tracking via Frigate + HA sensors.

Subscribes to Frigate MQTT events (person detection, face recognition) and
HA WebSocket events (motion sensors, light states, dimmer switches, device trackers).
Feeds all signals into BayesianOccupancy for per-room probability estimation.

Camera-to-room mapping is discovered dynamically from HA entity registry.
"""

import asyncio
import contextlib
import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from aria.capabilities import Capability
from aria.engine.analysis.occupancy import SENSOR_CONFIG, BayesianOccupancy
from aria.hub.constants import CACHE_PRESENCE, RECONNECT_STAGGER
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)

# MQTT topics for Frigate events
FRIGATE_EVENTS_TOPIC = "frigate/events"
FRIGATE_REVIEWS_TOPIC = "frigate/reviews"

# How often to recalculate and publish presence state
PRESENCE_FLUSH_INTERVAL_S = 30

# Decay: how long after last signal before a room goes "unoccupied"
SIGNAL_STALE_S = 600  # 10 minutes with no signal = likely empty

# Camera-to-room mapping is discovered dynamically from HA entity registry.
# Manual overrides via presence.camera_rooms config key.


class PresenceModule(Module):
    """Real-time presence tracking via camera + sensor fusion."""

    CAPABILITIES = [
        Capability(
            id="presence_tracking",
            name="Presence Tracking",
            description="Real-time per-room presence probability from Frigate cameras and HA sensors.",
            module="presence",
            layer="hub",
            config_keys=[
                "presence.mqtt_host",
                "presence.mqtt_port",
                "presence.mqtt_user",
                "presence.mqtt_password",
                "presence.camera_rooms",
            ],
            test_paths=["tests/hub/test_presence.py"],
        ),
    ]

    def __init__(  # noqa: PLR0913 — constructor with well-defined config params
        self,
        hub: IntelligenceHub,
        ha_url: str,
        ha_token: str,
        mqtt_host: str = "",
        mqtt_port: int = 1883,
        mqtt_user: str = "",
        mqtt_password: str = "",
        camera_rooms: dict[str, str] | None = None,
    ):
        super().__init__("presence", hub)
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_user = mqtt_user
        self.mqtt_password = mqtt_password
        self._config_camera_rooms = camera_rooms  # Explicit overrides (always win)
        self.camera_rooms: dict[str, str] = dict(camera_rooms) if camera_rooms else {}

        # Person home/away states: {entity_id: state_string} — seeded from HA on startup
        self._person_states: dict[str, str] = {}

        # Per-room signal state: {room: [(signal_type, value, detail, timestamp), ...]}
        self._room_signals: dict[str, list] = defaultdict(list)

        # Identified persons: {person_name: {room, last_seen, confidence}}
        self._identified_persons: dict[str, dict] = {}

        # Recent person detections across all cameras (ring buffer, newest last)
        self._recent_detections: list[dict] = []
        self._max_recent_detections = 20

        # Frigate API base URL (Docker runs locally even though MQTT is on HA Pi)
        self._frigate_url = os.environ.get("FRIGATE_URL", "http://127.0.0.1:5000")

        # Frigate camera names (populated from /api/config, used for alias resolution)
        self._frigate_camera_names: set[str] = set()

        # Face recognition config (fetched lazily from Frigate)
        self._face_config: dict | None = None
        self._labeled_faces: dict[str, int] = {}  # name -> count
        self._face_config_fetched = False

        # Bayesian estimator (shared with engine, but used in real-time here)
        self._occupancy = BayesianOccupancy()

        # Shared aiohttp session (created in initialize, closed in shutdown)
        self._http_session: aiohttp.ClientSession | None = None

        # MQTT client (lazy init)
        self._mqtt_client = None
        self._mqtt_connected = False

    async def _discover_camera_rooms(self) -> dict[str, str]:
        """Build camera->room mapping from HA entity/device registry cache.

        Filters entity cache for camera.* entities with active or stale
        lifecycle status, resolves area via entity->device->area chain.
        """
        mapping: dict[str, str] = {}

        entities_entry = await self.hub.get_cache("entities")
        devices_entry = await self.hub.get_cache("devices")

        if not entities_entry:
            return mapping

        entities_data = entities_entry.get("data", entities_entry) if isinstance(entities_entry, dict) else {}
        devices_data = {}
        if devices_entry:
            devices_data = devices_entry.get("data", devices_entry) if isinstance(devices_entry, dict) else {}

        for entity_id, entity_info in entities_data.items():
            if not entity_id.startswith("camera."):
                continue

            # Skip archived cameras
            lifecycle = entity_info.get("_lifecycle", {})
            if lifecycle.get("status") == "archived":
                continue

            camera_name = entity_id.removeprefix("camera.")

            # Resolve area: entity -> device -> area
            area = entity_info.get("area_id")
            if not area:
                device_id = entity_info.get("device_id")
                if device_id and device_id in devices_data:
                    area = devices_data[device_id].get("area_id")

            mapping[camera_name] = area if area else camera_name

        return mapping

    async def _refresh_camera_rooms(self):
        """Refresh camera->room mapping from discovery cache (merge, not replace).

        Config overrides always take priority over discovered mappings.
        """
        discovered = await self._discover_camera_rooms()

        # Merge: discovered cameras added/updated, existing preserved
        for cam, room in discovered.items():
            # Config overrides always win
            if self._config_camera_rooms and cam in self._config_camera_rooms:
                continue
            self.camera_rooms[cam] = room

        # Add Frigate short-name aliases
        self._add_frigate_aliases()

        self.logger.info(f"Camera rooms refreshed: {len(self.camera_rooms)} cameras ({len(discovered)} discovered)")

    def _add_frigate_aliases(self):
        """Add Frigate short-name aliases to camera_rooms.

        For each Frigate camera name, find the HA camera entity whose name
        contains the Frigate name as a substring. Add the short name as
        an alias pointing to the same room.
        """
        if not self._frigate_camera_names:
            return

        ha_camera_names = {
            name: room for name, room in self.camera_rooms.items() if name not in self._frigate_camera_names
        }

        aliases_added = 0
        for frigate_name in self._frigate_camera_names:
            if frigate_name in self.camera_rooms:
                continue  # already exists (config override or exact match)

            # Find HA entity name containing this Frigate name
            matches = [(ha_name, room) for ha_name, room in ha_camera_names.items() if frigate_name in ha_name]

            if len(matches) == 1:
                self.camera_rooms[frigate_name] = matches[0][1]
                aliases_added += 1
            elif len(matches) > 1:
                # Multiple matches — use shortest (most specific)
                best = min(matches, key=lambda m: len(m[0]))
                self.camera_rooms[frigate_name] = best[1]
                aliases_added += 1
            else:
                self.logger.debug(f"No HA entity match for Frigate camera: {frigate_name}")

        if aliases_added:
            self.logger.info(f"Added {aliases_added} Frigate camera aliases")

    async def _seed_presence_from_ha(self):
        """Fetch current person states from HA REST API to avoid cold-start zeros."""
        try:
            ha_url = os.environ.get("HA_URL", "")
            ha_token = os.environ.get("HA_TOKEN", "")
            if not ha_url or not ha_token:
                logger.warning("Cannot seed presence: HA_URL/HA_TOKEN not set")
                return
            session = self._http_session
            if not session:
                logger.warning("Cannot seed presence: HTTP session not initialized")
                return
            try:
                headers = {"Authorization": f"Bearer {ha_token}"}
                async with session.get(
                    f"{ha_url}/api/states", headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        states = await resp.json()
                        count = 0
                        for state in states:
                            eid = state.get("entity_id", "")
                            if eid.startswith("person."):
                                self._person_states[eid] = state.get("state", "unknown")
                                count += 1
                        logger.info("Seeded %d person states from HA", count)
                    else:
                        logger.warning("HA REST API returned %d during presence seeding", resp.status)
            finally:
                if not self._http_session:
                    await session.close()
        except Exception as e:
            logger.warning("Failed to seed presence from HA: %s", e)

    async def initialize(self):
        """Start MQTT listener and HA WebSocket listener."""
        self.logger.info("Presence module initializing...")

        # Create shared HTTP session for all outbound requests
        self._http_session = aiohttp.ClientSession()

        await self._seed_presence_from_ha()

        # Start MQTT listener for Frigate events
        await self.hub.schedule_task(
            task_id="presence_mqtt_listener",
            coro=self._mqtt_listen_loop,
            interval=None,
            run_immediately=True,
        )

        # Start HA WebSocket listener for sensor events
        await self.hub.schedule_task(
            task_id="presence_ws_listener",
            coro=self._ws_listen_loop,
            interval=None,
            run_immediately=True,
        )

        # Start periodic presence state flush
        await self.hub.schedule_task(
            task_id="presence_state_flush",
            coro=self._flush_presence_state,
            interval=timedelta(seconds=PRESENCE_FLUSH_INTERVAL_S),
            run_immediately=False,
        )

        # Fetch Frigate face recognition config (once, then periodically)
        await self.hub.schedule_task(
            task_id="presence_face_config",
            coro=self._fetch_face_config,
            interval=timedelta(minutes=5),
            run_immediately=True,
        )

        # Fetch Frigate config first (populates _frigate_camera_names for alias resolution)
        try:
            await self._fetch_face_config()
        except Exception as e:
            self.logger.warning(f"Frigate config fetch failed (non-fatal): {e}")

        # Discover camera->room mapping from entity cache (uses Frigate names for aliases)
        try:
            await self._refresh_camera_rooms()
        except Exception as e:
            self.logger.warning(f"Camera discovery failed (non-fatal): {e}")

        self.logger.info("Presence module started")

    async def shutdown(self):
        """Clean up MQTT connection and HTTP session."""
        if self._http_session:
            with contextlib.suppress(Exception):
                await self._http_session.close()
            self._http_session = None
        if self._mqtt_client:
            with contextlib.suppress(Exception):
                self._mqtt_client.disconnect()
        self.logger.info("Presence module shut down")

    async def on_event(self, event_type: str, data: dict[str, Any]):
        """Handle hub events — refresh cameras on discovery completion."""
        if event_type == "discovery_complete":
            try:
                await self._refresh_camera_rooms()
            except Exception as e:
                self.logger.warning(f"Camera refresh on discovery event failed: {e}")

    # ------------------------------------------------------------------
    # Frigate API helpers (face config, thumbnails, labeled faces)
    # ------------------------------------------------------------------

    async def _fetch_face_config(self):
        """Fetch face recognition config and labeled faces from Frigate."""
        if not self._http_session:
            return
        try:
            # Fetch face recognition config
            async with self._http_session.get(
                f"{self._frigate_url}/api/config", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    config = await resp.json()
                    self._face_config = config.get("face_recognition", {})
                    self._face_config_fetched = True

                    # Extract camera names for MQTT alias resolution
                    cameras = config.get("cameras", {})
                    if cameras:
                        self._frigate_camera_names = set(cameras.keys())

            # Fetch labeled faces (name -> list of face images)
            async with self._http_session.get(
                f"{self._frigate_url}/api/faces", timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    faces = await resp.json()
                    self._labeled_faces = {
                        name: len(images) if isinstance(images, list) else 0 for name, images in faces.items()
                    }
        except Exception as e:
            self.logger.debug(f"Failed to fetch Frigate face config: {e}")

    async def get_frigate_thumbnail(self, event_id: str) -> bytes | None:
        """Proxy a Frigate event thumbnail. Returns JPEG bytes or None."""
        if not self._http_session:
            return None
        try:
            async with self._http_session.get(
                f"{self._frigate_url}/api/events/{event_id}/thumbnail.jpg",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception:
            pass
        return None

    async def get_frigate_snapshot(self, event_id: str) -> bytes | None:
        """Proxy a Frigate event snapshot. Returns JPEG bytes or None."""
        if not self._http_session:
            return None
        try:
            async with self._http_session.get(
                f"{self._frigate_url}/api/events/{event_id}/snapshot.jpg",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # MQTT listener (Frigate events)
    # ------------------------------------------------------------------

    async def _mqtt_listen_loop(self):
        """Connect to MQTT broker and listen for Frigate events."""
        try:
            import aiomqtt
        except ImportError:
            self.logger.error("aiomqtt not installed. Run: pip install aiomqtt")
            return

        stagger = RECONNECT_STAGGER.get("presence_mqtt", 4)
        retry_delay = 5
        first_connect = True

        while self.hub.is_running():
            try:
                async with aiomqtt.Client(
                    hostname=self.mqtt_host,
                    port=self.mqtt_port,
                    username=self.mqtt_user,
                    password=self.mqtt_password,
                ) as client:
                    self._mqtt_connected = True
                    self.logger.info(f"MQTT connected to {self.mqtt_host}:{self.mqtt_port}")
                    retry_delay = 5
                    first_connect = False

                    # Subscribe to Frigate event topics
                    await client.subscribe("frigate/events")
                    await client.subscribe("frigate/+/person")

                    async for message in client.messages:
                        try:
                            payload = json.loads(message.payload.decode())
                            topic = str(message.topic)
                            await self._handle_mqtt_message(topic, payload)
                        except json.JSONDecodeError:
                            pass
                        except Exception as e:
                            self.logger.warning(f"MQTT message error: {e}")

            except Exception as e:
                self._mqtt_connected = False
                self.logger.warning(f"MQTT connection failed: {e}, retrying in {retry_delay}s")

                # Apply stagger on first reconnect attempt to avoid thundering herd
                base_delay = retry_delay + (stagger if first_connect else 0)
                first_connect = False

                jitter = base_delay * random.uniform(-0.25, 0.25)
                actual_delay = base_delay + jitter
                await asyncio.sleep(actual_delay)
                retry_delay = min(retry_delay * 2, 60)

    async def _handle_mqtt_message(self, topic: str, payload: dict):
        """Process a Frigate MQTT event."""
        if topic == "frigate/events":
            await self._handle_frigate_event(payload)
        elif "/person" in topic:
            # Topic format: frigate/<camera>/person
            parts = topic.split("/")
            if len(parts) >= 2:
                camera = parts[1]
                await self._handle_person_count(camera, payload)

    async def _handle_frigate_event(self, event: dict):
        """Handle a Frigate event (person detected, face recognized)."""
        after = event.get("after", {})
        if not after:
            return

        camera = after.get("camera", "")
        label = after.get("label", "")
        sub_label = after.get("sub_label")  # Face recognition result
        score = after.get("score", 0)
        room = self.camera_rooms.get(camera, camera)
        now = datetime.now()

        if label == "person":
            # Person detected in camera
            self._add_signal(
                room,
                "camera_person",
                min(score, 0.99),
                f"person detected on {camera} (score={score:.2f})",
                now,
            )

            # Track recent detection for cross-camera view
            event_id = after.get("id", "")
            self._recent_detections.append(
                {
                    "event_id": event_id,
                    "camera": camera,
                    "room": room,
                    "score": round(score, 3),
                    "sub_label": sub_label[0] if isinstance(sub_label, list) and sub_label else sub_label,
                    "has_snapshot": after.get("has_snapshot", False),
                    "timestamp": now.isoformat(),
                }
            )
            # Keep ring buffer bounded
            if len(self._recent_detections) > self._max_recent_detections:
                self._recent_detections = self._recent_detections[-self._max_recent_detections :]

            if sub_label and isinstance(sub_label, list) and sub_label:
                # Face recognized — sub_label is the person's name
                person_name = sub_label[0] if isinstance(sub_label, list) else sub_label
                confidence = after.get("sub_label_score", 0.9)
                self._add_signal(
                    room,
                    "camera_face",
                    min(confidence, 0.99),
                    f"{person_name} identified on {camera} (conf={confidence:.2f})",
                    now,
                )
                self._identified_persons[person_name] = {
                    "room": room,
                    "last_seen": now.isoformat(),
                    "confidence": round(confidence, 3),
                    "camera": camera,
                }
                self.logger.info(f"Face recognized: {person_name} in {room} (conf={confidence:.2f})")

    async def _handle_person_count(self, camera: str, count):
        """Handle person count update for a camera."""
        room = self.camera_rooms.get(camera, camera)
        now = datetime.now()
        try:
            n = int(count)
        except (ValueError, TypeError):
            return

        if n > 0:
            self._add_signal(
                room,
                "camera_person",
                0.95,
                f"{n} person(s) on {camera}",
                now,
            )

    # ------------------------------------------------------------------
    # HA WebSocket listener (motion, lights, dimmers, device_tracker)
    # ------------------------------------------------------------------

    async def _ws_listen_loop(self):  # noqa: C901
        """Connect to HA WebSocket and listen for presence-relevant events."""
        ws_url = self.ha_url.replace("http", "ws", 1) + "/api/websocket"
        stagger = RECONNECT_STAGGER.get("presence_ws", 6)
        retry_delay = 5
        first_connect = True

        while self.hub.is_running():
            if not self._http_session:
                self.logger.warning("HTTP session not available — waiting for init")
                await asyncio.sleep(retry_delay)
                continue
            try:
                async with self._http_session.ws_connect(ws_url) as ws:
                    msg = await ws.receive_json()
                    if msg.get("type") != "auth_required":
                        continue

                    await ws.send_json(
                        {
                            "type": "auth",
                            "access_token": self.ha_token,
                        }
                    )
                    auth_resp = await ws.receive_json()
                    if auth_resp.get("type") != "auth_ok":
                        self.logger.error(f"WS auth failed: {auth_resp}")
                        jitter = retry_delay * random.uniform(-0.25, 0.25)
                        actual_delay = retry_delay + jitter
                        await asyncio.sleep(actual_delay)
                        continue

                    # Subscribe to state_changed events
                    await ws.send_json(
                        {
                            "id": 1,
                            "type": "subscribe_events",
                            "event_type": "state_changed",
                        }
                    )
                    await ws.receive_json()  # subscription confirmation

                    self.logger.info("Presence WS connected — listening for sensor events")
                    retry_delay = 5
                    first_connect = False

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                if data.get("type") == "event":
                                    event_data = data.get("event", {}).get("data", {})
                                    await self._handle_ha_state_change(event_data)
                            except json.JSONDecodeError:
                                pass
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break

            except Exception as e:
                self.logger.warning(f"Presence WS error: {e}, retrying in {retry_delay}s")

                # Apply stagger on first reconnect attempt to avoid thundering herd
                base_delay = retry_delay + (stagger if first_connect else 0)
                first_connect = False

                jitter = base_delay * random.uniform(-0.25, 0.25)
                actual_delay = base_delay + jitter
                await asyncio.sleep(actual_delay)
                retry_delay = min(retry_delay * 2, 60)

    def _handle_person_state(self, entity_id: str, state: str, now: datetime):
        """Handle person entity home/away signals."""
        if state == "home":
            self._add_signal("overall", "device_tracker", 0.9, f"{entity_id} is home", now)
        elif state == "not_home":
            self._add_signal("overall", "device_tracker", 0.1, f"{entity_id} is away", now)

    def _handle_room_entity(  # noqa: PLR0913 — entity signal dispatch needs all context params
        self, entity_id: str, state: str, attrs: dict, device_class: str, room: str, now: datetime
    ):
        """Handle room-associated entity signals (motion, lights, dimmers, doors)."""
        if entity_id.startswith("binary_sensor.") and device_class == "motion" and state == "on":
            self._add_signal(room, "motion", 0.95, f"{entity_id} triggered", now)
        elif entity_id.startswith("light.") and state in ("on", "off"):
            self._add_signal(room, "light_interaction", 0.8, f"{entity_id} turned {state}", now)
        elif entity_id.startswith("event.hue_dimmer"):
            event_type = attrs.get("event_type", "")
            if event_type in ("initial_press", "short_release"):
                self._add_signal(room, "dimmer_press", 0.95, f"{entity_id} pressed", now)
        elif entity_id.startswith("binary_sensor.") and device_class == "door" and state in ("on", "off"):
            self._add_signal(room, "door", 0.7, f"{entity_id} {'opened' if state == 'on' else 'closed'}", now)

    async def _handle_ha_state_change(self, data: dict):
        """Process a state_changed event for presence-relevant entities."""
        new_state = data.get("new_state")
        if not new_state:
            return

        entity_id = new_state.get("entity_id", "")
        state = new_state.get("state", "")
        attrs = new_state.get("attributes", {})
        device_class = attrs.get("device_class", "")
        now = datetime.now()

        if entity_id.startswith("person."):
            self._handle_person_state(entity_id, state, now)
            return

        room = await self._resolve_room(entity_id, attrs)
        if not room:
            return

        self._handle_room_entity(entity_id, state, attrs, device_class, room, now)

    async def _resolve_room(self, entity_id: str, attrs: dict) -> str | None:
        """Resolve an entity to its room/area name.

        Uses the discovery cache if available, falls back to entity_id parsing.
        """
        # Try discovery cache (entities -> device -> area chain)
        try:
            entities_entry = await self.hub.get_cache("entities")
            if entities_entry:
                entities_cache = entities_entry.get("data", entities_entry) if isinstance(entities_entry, dict) else {}
                entity_data = entities_cache.get(entity_id, {})
                area = entity_data.get("area_id") or entity_data.get("area")
                if area:
                    return area

                # Fall back to device -> area
                device_id = entity_data.get("device_id")
                if device_id:
                    devices_entry = await self.hub.get_cache("devices")
                    if devices_entry:
                        devices_cache = (
                            devices_entry.get("data", devices_entry) if isinstance(devices_entry, dict) else {}
                        )
                        device = devices_cache.get(device_id, {})
                        area = device.get("area_id")
                        if area:
                            return area
        except Exception:
            pass

        self.logger.debug("Could not resolve room for %s", entity_id)
        return None

    # ------------------------------------------------------------------
    # Signal management + state flush
    # ------------------------------------------------------------------

    def _add_signal(
        self,
        room: str,
        signal_type: str,
        value: float,
        detail: str,
        timestamp: datetime,
    ):
        """Add a presence signal for a room."""
        self._room_signals[room].append((signal_type, value, detail, timestamp))

    def _get_active_signals(self, room: str, now: datetime) -> list:
        """Get non-stale signals for a room."""
        cutoff = now - timedelta(seconds=SIGNAL_STALE_S)
        active = []
        for sig_type, value, detail, ts in self._room_signals.get(room, []):
            # Apply per-type decay
            config = SENSOR_CONFIG.get(sig_type, {"decay_seconds": 300})
            decay = config.get("decay_seconds", 300)
            if decay == 0 or ts >= cutoff:
                # No decay or within stale window
                if decay > 0:
                    age = (now - ts).total_seconds()
                    decay_factor = max(0.1, 1.0 - (age / decay))
                    value = value * decay_factor
                active.append((sig_type, value, detail))
        return active

    async def _flush_presence_state(self):
        """Recalculate presence probabilities and write to cache."""
        now = datetime.now()
        results = {}

        # All rooms that have had any signal
        all_rooms = set(self._room_signals.keys())

        for room in all_rooms:
            signals = self._get_active_signals(room, now)
            if not signals:
                results[room] = {
                    "probability": 0.1,
                    "confidence": "none",
                    "signals": [],
                    "persons": [],
                }
                continue

            # Use BayesianOccupancy's fusion
            prior = self._occupancy._get_prior(now, room)
            probability = self._occupancy._bayesian_fuse(prior, signals)

            # Persons currently identified in this room
            persons_in_room = [
                {"name": name, **info}
                for name, info in self._identified_persons.items()
                if info.get("room") == room
                and (now - datetime.fromisoformat(info["last_seen"])).total_seconds() < SIGNAL_STALE_S
            ]

            results[room] = {
                "probability": round(probability, 3),
                "confidence": BayesianOccupancy._classify_confidence(probability, len(signals)),
                "signals": [{"type": t, "value": round(v, 2), "detail": d} for t, v, d in signals],
                "persons": persons_in_room,
            }

        # Build summary
        occupied_rooms = [r for r, d in results.items() if d["probability"] > 0.5 and r != "overall"]
        all_persons = {
            name: info
            for name, info in self._identified_persons.items()
            if (now - datetime.fromisoformat(info["last_seen"])).total_seconds() < SIGNAL_STALE_S
        }

        presence_data = {
            "timestamp": now.isoformat(),
            "rooms": results,
            "occupied_rooms": occupied_rooms,
            "identified_persons": {
                name: {
                    "room": info["room"],
                    "last_seen": info["last_seen"],
                    "confidence": info["confidence"],
                }
                for name, info in all_persons.items()
            },
            "mqtt_connected": self._mqtt_connected,
            "camera_rooms": self.camera_rooms,
            "face_recognition": {
                "enabled": bool(self._face_config and self._face_config.get("enabled")),
                "config": self._face_config or {},
                "labeled_faces": self._labeled_faces,
                "labeled_count": len(self._labeled_faces),
            },
            "recent_detections": list(reversed(self._recent_detections)),
        }

        # Write to cache
        await self.hub.set_cache(CACHE_PRESENCE, presence_data)

        # Publish event for other modules
        await self.hub.publish("presence_updated", presence_data)

        # Prune stale signals (keep last 10 minutes)
        cutoff = now - timedelta(seconds=SIGNAL_STALE_S)
        for room in list(self._room_signals.keys()):
            self._room_signals[room] = [s for s in self._room_signals[room] if s[3] >= cutoff]
            if not self._room_signals[room]:
                del self._room_signals[room]
