"""Presence Module - Real-time room presence tracking via Frigate + HA sensors.

Subscribes to Frigate MQTT events (person detection, face recognition) and
HA WebSocket events (motion sensors, light states, dimmer switches, device trackers).
Feeds all signals into BayesianOccupancy for per-room probability estimation.

Camera-to-room mapping is configured via the camera_rooms dict.
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import aiohttp

from aria.hub.core import Module, IntelligenceHub
from aria.hub.constants import CACHE_PRESENCE
from aria.capabilities import Capability
from aria.engine.analysis.occupancy import BayesianOccupancy, SENSOR_CONFIG


logger = logging.getLogger(__name__)

# MQTT topics for Frigate events
FRIGATE_EVENTS_TOPIC = "frigate/events"
FRIGATE_REVIEWS_TOPIC = "frigate/reviews"

# How often to recalculate and publish presence state
PRESENCE_FLUSH_INTERVAL_S = 30

# Decay: how long after last signal before a room goes "unoccupied"
SIGNAL_STALE_S = 600  # 10 minutes with no signal = likely empty

# Camera-to-room mapping (Frigate camera name -> ARIA area name)
# Update this when adding/removing cameras
DEFAULT_CAMERA_ROOMS = {
    "driveway": "driveway",
    "backyard": "backyard",
    "panoramic": "backyard",
    "front_doorbell": "front_door",
    "carters_room": "carters_room",
    "collins_room": "collins_room",
    "pool": "pool",
}


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

    def __init__(
        self,
        hub: IntelligenceHub,
        ha_url: str,
        ha_token: str,
        mqtt_host: str = "192.168.1.35",
        mqtt_port: int = 1883,
        mqtt_user: str = "frigate",
        mqtt_password: str = "frigate_mqtt_2026",
        camera_rooms: Optional[Dict[str, str]] = None,
    ):
        super().__init__("presence", hub)
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_user = mqtt_user
        self.mqtt_password = mqtt_password
        self.camera_rooms = camera_rooms or DEFAULT_CAMERA_ROOMS

        # Per-room signal state: {room: [(signal_type, value, detail, timestamp), ...]}
        self._room_signals: Dict[str, List] = defaultdict(list)

        # Identified persons: {person_name: {room, last_seen, confidence}}
        self._identified_persons: Dict[str, Dict] = {}

        # Bayesian estimator (shared with engine, but used in real-time here)
        self._occupancy = BayesianOccupancy()

        # MQTT client (lazy init)
        self._mqtt_client = None
        self._mqtt_connected = False

    async def initialize(self):
        """Start MQTT listener and HA WebSocket listener."""
        self.logger.info("Presence module initializing...")

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

        self.logger.info("Presence module started")

    async def shutdown(self):
        """Clean up MQTT connection."""
        if self._mqtt_client:
            try:
                self._mqtt_client.disconnect()
            except Exception:
                pass
        self.logger.info("Presence module shut down")

    # ------------------------------------------------------------------
    # MQTT listener (Frigate events)
    # ------------------------------------------------------------------

    async def _mqtt_listen_loop(self):
        """Connect to MQTT broker and listen for Frigate events."""
        try:
            import aiomqtt
        except ImportError:
            self.logger.error(
                "aiomqtt not installed. Run: pip install aiomqtt"
            )
            return

        retry_delay = 5

        while self.hub.is_running():
            try:
                async with aiomqtt.Client(
                    hostname=self.mqtt_host,
                    port=self.mqtt_port,
                    username=self.mqtt_user,
                    password=self.mqtt_password,
                ) as client:
                    self._mqtt_connected = True
                    self.logger.info(
                        f"MQTT connected to {self.mqtt_host}:{self.mqtt_port}"
                    )
                    retry_delay = 5

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
                self.logger.warning(
                    f"MQTT connection failed: {e}, retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

    async def _handle_mqtt_message(self, topic: str, payload: Dict):
        """Process a Frigate MQTT event."""
        if topic == "frigate/events":
            await self._handle_frigate_event(payload)
        elif "/person" in topic:
            # Topic format: frigate/<camera>/person
            parts = topic.split("/")
            if len(parts) >= 2:
                camera = parts[1]
                await self._handle_person_count(camera, payload)

    async def _handle_frigate_event(self, event: Dict):
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
                room, "camera_person", min(score, 0.99),
                f"person detected on {camera} (score={score:.2f})", now,
            )

            if sub_label and isinstance(sub_label, list) and sub_label:
                # Face recognized — sub_label is the person's name
                person_name = sub_label[0] if isinstance(sub_label, list) else sub_label
                confidence = after.get("sub_label_score", 0.9)
                self._add_signal(
                    room, "camera_face", min(confidence, 0.99),
                    f"{person_name} identified on {camera} (conf={confidence:.2f})",
                    now,
                )
                self._identified_persons[person_name] = {
                    "room": room,
                    "last_seen": now.isoformat(),
                    "confidence": round(confidence, 3),
                    "camera": camera,
                }
                self.logger.info(
                    f"Face recognized: {person_name} in {room} "
                    f"(conf={confidence:.2f})"
                )

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
                room, "camera_person", 0.95,
                f"{n} person(s) on {camera}", now,
            )

    # ------------------------------------------------------------------
    # HA WebSocket listener (motion, lights, dimmers, device_tracker)
    # ------------------------------------------------------------------

    async def _ws_listen_loop(self):
        """Connect to HA WebSocket and listen for presence-relevant events."""
        ws_url = self.ha_url.replace("http", "ws", 1) + "/api/websocket"
        retry_delay = 5

        while self.hub.is_running():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        msg = await ws.receive_json()
                        if msg.get("type") != "auth_required":
                            continue

                        await ws.send_json({
                            "type": "auth",
                            "access_token": self.ha_token,
                        })
                        auth_resp = await ws.receive_json()
                        if auth_resp.get("type") != "auth_ok":
                            self.logger.error(f"WS auth failed: {auth_resp}")
                            await asyncio.sleep(retry_delay)
                            continue

                        # Subscribe to state_changed events
                        await ws.send_json({
                            "id": 1,
                            "type": "subscribe_events",
                            "event_type": "state_changed",
                        })
                        await ws.receive_json()  # subscription confirmation

                        self.logger.info(
                            "Presence WS connected — listening for sensor events"
                        )
                        retry_delay = 5

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    if data.get("type") == "event":
                                        event_data = data.get("event", {}).get(
                                            "data", {}
                                        )
                                        await self._handle_ha_state_change(
                                            event_data
                                        )
                                except json.JSONDecodeError:
                                    pass
                            elif msg.type in (
                                aiohttp.WSMsgType.CLOSED,
                                aiohttp.WSMsgType.ERROR,
                            ):
                                break

            except Exception as e:
                self.logger.warning(
                    f"Presence WS error: {e}, retrying in {retry_delay}s"
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

    async def _handle_ha_state_change(self, data: Dict):
        """Process a state_changed event for presence-relevant entities."""
        new_state = data.get("new_state")
        if not new_state:
            return

        entity_id = new_state.get("entity_id", "")
        state = new_state.get("state", "")
        attrs = new_state.get("attributes", {})
        device_class = attrs.get("device_class", "")
        now = datetime.now()

        # Resolve entity to room (use area_id from attributes or device)
        room = self._resolve_room(entity_id, attrs)
        if not room:
            return

        # Motion sensors
        if entity_id.startswith("binary_sensor.") and device_class == "motion":
            if state == "on":
                self._add_signal(
                    room, "motion", 0.95,
                    f"{entity_id} triggered", now,
                )

        # Light state changes (someone turned it on/off)
        elif entity_id.startswith("light."):
            if state in ("on", "off"):
                self._add_signal(
                    room, "light_interaction", 0.8,
                    f"{entity_id} turned {state}", now,
                )

        # Hue dimmer switch button presses
        elif entity_id.startswith("event.hue_dimmer"):
            event_type = attrs.get("event_type", "")
            if event_type in ("initial_press", "short_release"):
                self._add_signal(
                    room, "dimmer_press", 0.95,
                    f"{entity_id} pressed", now,
                )

        # Person/device tracker (home/away)
        elif entity_id.startswith("person."):
            if state == "home":
                self._add_signal(
                    "overall", "device_tracker", 0.9,
                    f"{entity_id} is home", now,
                )
            elif state == "not_home":
                self._add_signal(
                    "overall", "device_tracker", 0.1,
                    f"{entity_id} is away", now,
                )

        # Door sensors
        elif entity_id.startswith("binary_sensor.") and device_class == "door":
            if state in ("on", "off"):
                self._add_signal(
                    room, "door", 0.7,
                    f"{entity_id} {'opened' if state == 'on' else 'closed'}",
                    now,
                )

    def _resolve_room(self, entity_id: str, attrs: Dict) -> Optional[str]:
        """Resolve an entity to its room/area name.

        Uses the discovery cache if available, falls back to entity_id parsing.
        """
        # Try discovery cache (entities -> device -> area chain)
        try:
            entities_cache = self.hub.cache.get_cache("entities")
            if entities_cache:
                entity_data = entities_cache.get(entity_id, {})
                area = entity_data.get("area_id") or entity_data.get("area")
                if area:
                    return area

                # Fall back to device -> area
                device_id = entity_data.get("device_id")
                if device_id:
                    devices_cache = self.hub.cache.get_cache("devices")
                    if devices_cache:
                        device = devices_cache.get(device_id, {})
                        area = device.get("area_id")
                        if area:
                            return area
        except Exception:
            pass

        # Fallback: parse entity_id for known patterns
        eid = entity_id.lower()
        if "bedroom" in eid:
            return "bedroom"
        if "closet" in eid:
            return "closet"
        if "front_door" in eid or "doorbell" in eid:
            return "front_door"

        return None

    # ------------------------------------------------------------------
    # Signal management + state flush
    # ------------------------------------------------------------------

    def _add_signal(
        self, room: str, signal_type: str, value: float,
        detail: str, timestamp: datetime,
    ):
        """Add a presence signal for a room."""
        self._room_signals[room].append(
            (signal_type, value, detail, timestamp)
        )

    def _get_active_signals(self, room: str, now: datetime) -> List:
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
                and (now - datetime.fromisoformat(info["last_seen"])).total_seconds()
                < SIGNAL_STALE_S
            ]

            results[room] = {
                "probability": round(probability, 3),
                "confidence": BayesianOccupancy._classify_confidence(
                    probability, len(signals)
                ),
                "signals": [
                    {"type": t, "value": round(v, 2), "detail": d}
                    for t, v, d in signals
                ],
                "persons": persons_in_room,
            }

        # Build summary
        occupied_rooms = [
            r for r, d in results.items()
            if d["probability"] > 0.5 and r != "overall"
        ]
        all_persons = {
            name: info
            for name, info in self._identified_persons.items()
            if (now - datetime.fromisoformat(info["last_seen"])).total_seconds()
            < SIGNAL_STALE_S
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
        }

        # Write to cache
        await self.hub.cache.set_cache(CACHE_PRESENCE, presence_data)

        # Publish event for other modules
        await self.hub.publish("presence_updated", presence_data)

        # Prune stale signals (keep last 10 minutes)
        cutoff = now - timedelta(seconds=SIGNAL_STALE_S)
        for room in list(self._room_signals.keys()):
            self._room_signals[room] = [
                s for s in self._room_signals[room] if s[3] >= cutoff
            ]
            if not self._room_signals[room]:
                del self._room_signals[room]
