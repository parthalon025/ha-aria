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
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import aiohttp

from aria.capabilities import Capability
from aria.engine.analysis.occupancy import SENSOR_CONFIG, BayesianOccupancy
from aria.hub.constants import CACHE_PRESENCE, RECONNECT_STAGGER
from aria.hub.core import IntelligenceHub, Module

logger = logging.getLogger(__name__)


def _evict_oldest_snapshots(snapshots_dir: Path, max_bytes: int) -> None:
    """Delete oldest face snapshot JPEGs until the directory is under max_bytes.

    Files are ranked by mtime (oldest first). Only *.jpg files are measured and
    deleted — other files in the directory are left untouched.
    """
    stats = sorted(
        ((p, p.stat()) for p in snapshots_dir.glob("*.jpg")),
        key=lambda t: t[1].st_mtime,
    )
    total = sum(s.st_size for _, s in stats)
    for path, st in stats:
        if total <= max_bytes:
            break
        try:
            path.unlink()
            total -= st.st_size
        except OSError:
            logger.warning("Failed to evict snapshot %s", path)


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

    CAPABILITIES: ClassVar[list] = [
        Capability(
            id="presence_tracking",
            name="Presence Tracking",
            description="Real-time per-room presence probability from Frigate cameras and HA sensors.",
            module="presence",
            layer="hub",
            config_keys=(
                "presence.mqtt_host",
                "presence.mqtt_port",
                "presence.mqtt_user",
                "presence.mqtt_password",
                "presence.camera_rooms",
            ),
            test_paths=("tests/hub/test_presence.py",),
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

        # Config-driven weight/decay loading (refreshed every 60s)
        self._presence_config_loaded: datetime | None = None

        # Shared aiohttp session (created in initialize, closed in shutdown)
        self._http_session: aiohttp.ClientSession | None = None

        # MQTT client (lazy init)
        self._mqtt_client = None
        self._mqtt_connected = False

        # Local entity→room cache (built from discovery cache, refreshed periodically)
        self._entity_room_cache: dict[str, str] = {}
        self._entity_room_cache_built = False

        # Enabled signal types cache (from config, refreshed every 60s)
        self._enabled_signals: list[str] | None = None
        self._enabled_signals_ts: float = 0

        # Cached FacePipeline (lazy-initialized in initialize())
        self._face_pipeline: FacePipeline | None = None  # type: ignore[name-defined]  # noqa: F821
        self._face_last_processed: str | None = None  # exposed to hub for /api/faces/stats
        self._face_pipeline_errors: int = 0

        # Hub event subscriber callback references (for unsubscribe in shutdown)
        self._sub_unifi_protect = None

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
                        person_count = 0
                        room_count = 0
                        now = datetime.now(tz=UTC)
                        # Build entity→room cache first (single bulk read)
                        await self._build_entity_room_cache()
                        # Only seed presence-relevant domains
                        _SEED_PREFIXES = self._PRESENCE_DOMAINS
                        for state in states:
                            eid = state.get("entity_id", "")
                            st = state.get("state", "")
                            attrs = state.get("attributes", {})
                            if eid.startswith("person."):
                                self._person_states[eid] = st
                                person_count += 1
                            elif st and st not in ("unavailable", "unknown") and eid.startswith(_SEED_PREFIXES):
                                room = self._entity_room_cache.get(eid)
                                if room:
                                    device_class = attrs.get("device_class", "")
                                    self._handle_room_entity(eid, st, attrs, device_class, room, now)
                                    room_count += 1
                        logger.info("Seeded %d person states + %d room signals from HA", person_count, room_count)
                    else:
                        logger.warning("HA REST API returned %d during presence seeding", resp.status)
            finally:
                if session is not self._http_session and not session.closed:
                    await session.close()
        except Exception as e:
            logger.warning("Failed to seed presence from HA: %s", e)

    async def initialize(self):
        """Start MQTT listener and HA WebSocket listener."""
        self.logger.info("Presence module initializing...")

        # Create shared HTTP session for all outbound requests
        self._http_session = aiohttp.ClientSession()

        # Cache FacePipeline once (avoids reconstructing TF model per event)
        if hasattr(self.hub, "faces_store"):
            from aria.faces.pipeline import FacePipeline

            self._face_pipeline = FacePipeline(
                store=self.hub.faces_store,
                frigate_url=self._frigate_url,
            )
            # Expose health metrics on hub for /api/faces/stats
            self.hub._face_last_processed = None
            self.hub._face_pipeline_errors = 0

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

        # Subscribe to UniFi Protect person events (published by UniFiModule)
        self._sub_unifi_protect = self._handle_unifi_protect_person
        self.hub.subscribe("unifi_protect_person", self._sub_unifi_protect)

        self.logger.info("Presence module started")

    async def shutdown(self):
        """Clean up MQTT connection and HTTP session."""
        if self._sub_unifi_protect:
            self.hub.unsubscribe("unifi_protect_person", self._sub_unifi_protect)
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

    async def _handle_unifi_protect_person(self, payload: dict) -> None:
        """Ingest protect_person signal from UniFiModule into room signal history."""
        room = payload.get("room")
        value = payload.get("value", 0.85)
        detail = payload.get("detail", "protect_person")
        if room:
            self._add_signal(room, "protect_person", value, detail, datetime.now(tz=UTC))

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
            self.logger.warning(f"Failed to fetch Frigate face config: {e}")

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
            logger.warning("Failed to fetch Frigate thumbnail for %s", event_id, exc_info=True)
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
            logger.warning("Failed to fetch Frigate snapshot for %s", event_id, exc_info=True)
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
                            logger.debug("Invalid JSON in MQTT message payload")
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
        await self._refresh_enabled_signals()
        after = event.get("after", {})
        if not after:
            return

        camera = after.get("camera", "")
        label = after.get("label", "")
        sub_label = after.get("sub_label")  # Face recognition result
        score = after.get("score", 0)
        room = self.camera_rooms.get(camera, camera)
        now = datetime.now(UTC)

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
                # Highest-confidence wins — don't overwrite a more confident sighting
                existing = self._identified_persons.get(person_name, {})
                if confidence > existing.get("confidence", 0):
                    self._identified_persons[person_name] = {
                        "room": room,
                        "last_seen": now.isoformat(),
                        "confidence": round(confidence, 3),
                        "camera": camera,
                    }
                self.logger.info(f"Face recognized: {person_name} in {room} (conf={confidence:.2f})")
                task = asyncio.create_task(self._record_face_sighting(person_name, room, confidence, now))
                from aria.shared.utils import log_task_exception

                task.add_done_callback(log_task_exception)

            # Trigger ARIA face pipeline when snapshot is available (no sub_label required —
            # ARIA may recognise faces that Frigate's own recogniser missed)
            if event_id and after.get("has_snapshot", False):
                snapshot_url = f"{self._frigate_url}/api/events/{event_id}/snapshot.jpg"
                from aria.shared.utils import log_task_exception

                task = asyncio.create_task(self._process_face_async(event_id, snapshot_url, camera, room))
                task.add_done_callback(log_task_exception)

    async def _process_face_async(  # noqa: PLR0915
        self, event_id: str, snapshot_url: str, camera: str, room: str
    ) -> None:
        """Extract face from Frigate snapshot and run ARIA live pipeline.

        Fire-and-forget — called via create_task from _handle_frigate_event.
        Errors are logged but do not propagate to the caller.

        Requires hub.faces_store to be wired (aria/hub/core.py).  If the
        attribute is absent the method exits silently so that the rest of
        presence tracking is unaffected.
        """
        import tempfile

        pipeline = self._face_pipeline
        if pipeline is None:
            # Lazy fallback: faces_store may have been attached after initialize()
            if not hasattr(self.hub, "faces_store"):
                return
            from aria.faces.pipeline import FacePipeline

            pipeline = FacePipeline(
                store=self.hub.faces_store,
                frigate_url=self._frigate_url,
            )
            self._face_pipeline = pipeline

        tmp_path = None
        persistent_path = None
        try:
            session = self._http_session
            if session is None or session.closed:
                self.logger.warning("Presence._http_session unavailable — skipping face snapshot fetch")
                return
            async with (
                session.get(snapshot_url, timeout=aiohttp.ClientTimeout(total=10)) as resp,
            ):
                if resp.status != 200:
                    self.logger.debug(
                        "Face snapshot fetch failed: event=%s status=%d",
                        event_id,
                        resp.status,
                    )
                    return
                img_data = await resp.read()

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(img_data)
                tmp_path = f.name

            # extract_embedding is CPU-bound (DeepFace/TF inference) — run off event loop
            embedding = await asyncio.to_thread(pipeline.extractor.extract_embedding, tmp_path)
            if embedding is None:
                # No face detected — expected for many person events; discard temp file
                return

            # Save a persistent copy so the review queue can display the image.
            # Temp file is deleted in finally; persistent copy survives for labeling.
            snapshots_dir = Path(
                os.environ.get("ARIA_FACES_SNAPSHOTS_DIR", str(Path.home() / ".local/share/aria/faces/snapshots"))
            )
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            persistent_path = str(snapshots_dir / f"{event_id}.jpg")
            import shutil

            # Enforce storage cap — evict oldest snapshots before writing new one.
            # Cap is configurable via ARIA_FACES_SNAPSHOTS_MAX_GB (default 20 GB).
            _GB = 1024**3
            max_gb = float(os.environ.get("ARIA_FACES_SNAPSHOTS_MAX_GB", "20"))
            _evict_oldest_snapshots(snapshots_dir, max_bytes=int(max_gb * _GB))

            shutil.copy2(tmp_path, persistent_path)

            # process_embedding does SQLite I/O — run off event loop
            result = await asyncio.to_thread(pipeline.process_embedding, embedding, event_id, persistent_path)

            if result["action"] == "auto_label":
                person_name = result["person_name"]
                confidence = result["confidence"]
                now = datetime.now(UTC)
                # Highest-confidence wins — don't overwrite a more confident sighting
                existing = self._identified_persons.get(person_name, {})
                if confidence > existing.get("confidence", 0):
                    self._identified_persons[person_name] = {
                        "room": room,
                        "last_seen": now.isoformat(),
                        "confidence": min(round(confidence, 3), 0.99),
                        "camera": camera,
                    }
                self._add_signal(
                    room,
                    "camera_face",
                    min(confidence, 0.99),
                    f"{person_name} identified on {camera} via ARIA (conf={confidence:.2f})",
                    now,
                )
                self.logger.debug(
                    "Face auto-labeled: %s confidence=%.2f room=%s",
                    person_name,
                    confidence,
                    room,
                )
                task = asyncio.create_task(self._record_face_sighting(person_name, room, confidence, now))
                from aria.shared.utils import log_task_exception

                task.add_done_callback(log_task_exception)

            # Update pipeline health metrics
            now_iso = datetime.now(UTC).isoformat()
            self._face_last_processed = now_iso
            self.hub._face_last_processed = now_iso  # type: ignore[attr-defined]

        except Exception:
            self.logger.exception("Face pipeline error for event %s", event_id)
            self._face_pipeline_errors += 1
            self.hub._face_pipeline_errors = self._face_pipeline_errors  # type: ignore[attr-defined]
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    async def _record_face_sighting(self, person_name: str, room: str, confidence: float, ts) -> None:
        """Write face sighting to EventStore for intelligence module consumption."""
        import json as _json

        if not hasattr(self.hub, "event_store"):
            return
        try:
            await self.hub.event_store.insert_event(
                timestamp=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                entity_id=f"person.{person_name.lower().replace(' ', '_')}",
                domain="person",
                old_state=None,
                new_state=room,
                device_id=None,
                area_id=room,
                attributes_json=_json.dumps({"confidence": round(confidence, 3), "source": "face_recognition"}),
            )
        except Exception as e:
            self.logger.debug("Failed to record face sighting to EventStore: %s", e)

    async def _handle_person_count(self, camera: str, count):
        """Handle person count update for a camera."""
        await self._refresh_enabled_signals()
        room = self.camera_rooms.get(camera, camera)
        now = datetime.now(tz=UTC)
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

    async def _ws_auth_and_subscribe(self, ws) -> bool:
        """Authenticate and subscribe to state_changed on a HA WebSocket.

        Returns True if ready to receive events, False to retry.
        """
        msg = await ws.receive_json()
        if msg.get("type") != "auth_required":
            return False

        await ws.send_json({"type": "auth", "access_token": self.ha_token})
        auth_resp = await ws.receive_json()
        if auth_resp.get("type") != "auth_ok":
            self.logger.error(f"WS auth failed: {auth_resp}")
            return False

        await ws.send_json({"id": 1, "type": "subscribe_events", "event_type": "state_changed"})
        await ws.receive_json()  # subscription confirmation
        return True

    async def _ws_listen_loop(self):
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
                    if not await self._ws_auth_and_subscribe(ws):
                        jitter = retry_delay * random.uniform(-0.25, 0.25)
                        await asyncio.sleep(retry_delay + jitter)
                        continue

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
                                logger.debug("Invalid JSON in HA WebSocket message")
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
        elif entity_id.startswith("media_player."):
            if state in ("playing", "paused", "idle", "buffering"):
                self._add_signal(room, "media_active", 0.85, f"{entity_id} is {state}", now)
            elif state in ("off", "standby"):
                self._add_signal(room, "media_inactive", 0.15, f"{entity_id} turned {state}", now)

    # Domains that contribute to presence signals — skip everything else early
    _PRESENCE_DOMAINS = ("light.", "binary_sensor.", "media_player.", "event.")

    async def _handle_ha_state_change(self, data: dict):
        """Process a state_changed event for presence-relevant entities."""
        await self._refresh_enabled_signals()
        new_state = data.get("new_state")
        if not new_state:
            return

        entity_id = new_state.get("entity_id", "")

        if entity_id.startswith("person."):
            state = new_state.get("state", "")
            now = datetime.now(tz=UTC)
            self._handle_person_state(entity_id, state, now)
            return

        # Early exit for irrelevant domains — avoids expensive room resolution
        if not entity_id.startswith(self._PRESENCE_DOMAINS):
            return

        state = new_state.get("state", "")
        attrs = new_state.get("attributes", {})
        device_class = attrs.get("device_class", "")
        now = datetime.now(tz=UTC)

        room = await self._resolve_room(entity_id, attrs)
        if not room:
            return

        self._handle_room_entity(entity_id, state, attrs, device_class, room, now)

    async def _build_entity_room_cache(self):
        """Build local entity→room lookup from discovery cache.

        Fetches entities and devices caches once, builds a flat dict.
        Called during initialization and periodically to stay current.
        """
        try:
            entities_entry = await self.hub.get_cache("entities")
            if not entities_entry:
                return
            entities_cache = entities_entry.get("data", entities_entry) if isinstance(entities_entry, dict) else {}

            devices_entry = await self.hub.get_cache("devices")
            devices_cache = {}
            if devices_entry:
                devices_cache = devices_entry.get("data", devices_entry) if isinstance(devices_entry, dict) else {}

            new_cache: dict[str, str] = {}
            for eid, edata in entities_cache.items():
                if not isinstance(edata, dict):
                    continue
                area = edata.get("area_id") or edata.get("area")
                if not area:
                    device_id = edata.get("device_id")
                    if device_id and device_id in devices_cache:
                        area = devices_cache[device_id].get("area_id")
                if area:
                    new_cache[eid] = area

            self._entity_room_cache = new_cache
            self._entity_room_cache_built = True
            logger.debug("Entity room cache built: %d mappings", len(new_cache))
        except Exception as e:
            logger.warning("Failed to build entity room cache: %s", e)

    async def _resolve_room(self, entity_id: str, attrs: dict) -> str | None:
        """Resolve an entity to its room/area name.

        Uses local cache (fast dict lookup) first, falls back to hub cache on miss.
        """
        # Fast path: local cache hit
        cached = self._entity_room_cache.get(entity_id)
        if cached:
            return cached

        # Slow path: hub cache lookup (for entities added since last cache build)
        try:
            entities_entry = await self.hub.get_cache("entities")
            if entities_entry:
                entities_cache = entities_entry.get("data", entities_entry) if isinstance(entities_entry, dict) else {}
                entity_data = entities_cache.get(entity_id, {})
                area = entity_data.get("area_id") or entity_data.get("area")
                if area:
                    self._entity_room_cache[entity_id] = area
                    return area

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
                            self._entity_room_cache[entity_id] = area
                            return area
        except Exception:
            logger.warning("Failed to resolve room for entity %s", entity_id, exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Signal management + state flush
    # ------------------------------------------------------------------

    async def _refresh_enabled_signals(self):
        """Refresh the enabled signal types from config, cached for 60 seconds."""
        now = time.time()
        if self._enabled_signals is not None and now - self._enabled_signals_ts <= 60:
            return
        try:
            value = await self.hub.cache.get_config_value("presence.enabled_signals")
            if value:
                self._enabled_signals = [s.strip() for s in str(value).split(",")]
            else:
                self._enabled_signals = None  # None means all enabled
        except Exception:
            logger.warning("Failed to refresh enabled signals config", exc_info=True)
            self._enabled_signals = None  # On error, allow all
        self._enabled_signals_ts = now

    def _is_signal_enabled(self, signal_type: str) -> bool:
        """Check if a signal type is enabled (uses cached value)."""
        if self._enabled_signals is None:
            return True  # None means all enabled
        return signal_type in self._enabled_signals

    def _add_signal(
        self,
        room: str,
        signal_type: str,
        value: float,
        detail: str,
        timestamp: datetime,
    ):
        """Add a presence signal for a room (skips disabled signal types)."""
        if not self._is_signal_enabled(signal_type):
            return
        self._room_signals[room].append((signal_type, value, detail, timestamp))

    async def _load_presence_config(self):
        """Load presence weight/decay config from DB (cached 60s)."""
        now = datetime.now(tz=UTC)
        if self._presence_config_loaded and (now - self._presence_config_loaded).total_seconds() < 60:
            return
        overrides = {}
        for signal_type in SENSOR_CONFIG:
            weight = await self.hub.cache.get_config_value(
                f"presence.weight.{signal_type}",
                SENSOR_CONFIG[signal_type]["weight"],
            )
            decay = await self.hub.cache.get_config_value(
                f"presence.decay.{signal_type}",
                SENSOR_CONFIG[signal_type]["decay_seconds"],
            )
            overrides[signal_type] = {
                "weight": float(weight),
                "decay_seconds": int(float(decay)),
            }
        self._occupancy.update_sensor_config(overrides)
        self._presence_config_loaded = now

    def _get_active_signals(self, room: str, now: datetime) -> list:
        """Get non-stale signals for a room."""
        cutoff = now - timedelta(seconds=SIGNAL_STALE_S)
        active = []
        for sig_type, value, detail, ts in self._room_signals.get(room, []):
            # Apply per-type decay from instance config
            config = self._occupancy.sensor_config.get(sig_type, {"decay_seconds": 300})
            decay = config.get("decay_seconds", 300)
            if decay == 0 or ts >= cutoff:
                # No decay or within stale window
                if decay > 0:
                    age = (now - ts).total_seconds()
                    decay_factor = max(0.1, 1.0 - (age / decay))
                    value = value * decay_factor
                active.append((sig_type, value, detail))
        return active

    async def _apply_unifi_cross_validation(self) -> None:
        """Apply UniFi home/away gate and per-room cross-validation to _room_signals."""
        unifi_state = await self.hub.get_cache("unifi_client_state")
        if not unifi_state:
            return
        if unifi_state.get("home") is False:
            # All known devices absent — clear signal history for this cycle.
            # Note: this destroys accumulated signals, not a temporary suppression.
            # Recovery requires new _add_signal calls on the next flush cycle.
            total_signals = sum(len(sigs) for sigs in self._room_signals.values())
            num_rooms = len(self._room_signals)
            for room in list(self._room_signals.keys()):
                self._room_signals[room] = []
            logger.info(
                "UniFi home=False — clearing %d signals across %d rooms",
                total_signals,
                num_rooms,
            )
            return
        unifi_mod = self.hub.get_module("unifi") if hasattr(self.hub, "get_module") else None
        if unifi_mod is None:
            return
        # Shallow copy of the dict (not lists) to guard against dict-level mutation.
        # cross_validate_signals builds new lists internally so list contents are safe.
        adjusted = unifi_mod.cross_validate_signals(dict(self._room_signals))  # type: ignore[attr-defined]
        # cross_validate_signals returns 2-tuples (sig_type, value) in the same
        # order as the input. Zip by position to reconstruct 4-tuples — this
        # correctly handles rooms with duplicate signal types (e.g. two cameras).
        new_room_signals: dict = {}
        for room, signals in self._room_signals.items():
            adj_list = adjusted.get(room, [(s[0], s[1]) for s in signals])
            new_room_signals[room] = [
                (orig[0], adj[1], orig[2], orig[3]) for orig, adj in zip(signals, adj_list, strict=False)
            ]
        self._room_signals = new_room_signals

    async def _flush_presence_state(self):
        """Recalculate presence probabilities and write to cache."""
        await self._load_presence_config()
        now = datetime.now(tz=UTC)

        # Refresh entity→room cache every 5 minutes (piggyback on 30s flush cycle)
        if not self._entity_room_cache_built or (
            hasattr(self, "_last_room_cache_refresh") and (now - self._last_room_cache_refresh).total_seconds() > 300
        ):
            await self._build_entity_room_cache()
            self._last_room_cache_refresh = now
        results = {}

        # All rooms that have had any signal
        all_rooms = set(self._room_signals.keys())

        await self._apply_unifi_cross_validation()

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
                "aria_known_people": len(await asyncio.to_thread(self.hub.faces_store.get_known_people))
                if hasattr(self.hub, "faces_store")
                else 0,
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
