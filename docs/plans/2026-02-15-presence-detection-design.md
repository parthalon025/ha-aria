# Presence Detection & Person Identification — Design Document

**Date:** 2026-02-15
**Status:** Implementing
**Author:** Justin McFarland + Claude

---

## Goal

Room-level presence detection and person identification using existing hardware only (no new purchases), integrated into ARIA's existing BayesianOccupancy estimator.

## Constraints

- **No new hardware** — use existing UniFi Protect cameras, ONVIF cameras, Hue V2 hub (not Bridge Pro), Hue motion sensors, and dimmer switches
- **Room-level granularity** — in-room positioning ruled out (MotionAware requires Bridge Pro, Zigbee RSSI inaccessible)
- **Don't reinvent the wheel** — use HA native integrations (Frigate, Mosquitto) wherever possible; ARIA adds the intelligence layer on top
- **Feed into existing BayesianOccupancy** — not duplicate HA Bayesian sensors

## Architecture

### Data Flow

```
Cameras (RTSP) → Frigate (Docker) → MQTT (Mosquitto on HA)
                                        ↓
HA Sensors (motion, lights, dimmers) → HA WebSocket
                                        ↓
                   ARIA Presence Module ←┘
                        ↓
                BayesianOccupancy (per-room probability)
                        ↓
                Cache (CACHE_PRESENCE) → API → Dashboard
```

### Components

1. **Frigate NVR** (`~/frigate/docker-compose.yml`)
   - Docker container on workstation (CPU-only, no GPU)
   - Person detection (YOLOv8) + face recognition (FaceNet/ArcFace)
   - Publishes events to MQTT topic `frigate/events`
   - Face recognition: collect unknowns → user labels → continuous refinement
   - Config: `~/frigate/config/config.yml`

2. **Mosquitto MQTT** (HA addon `core_mosquitto`)
   - Broker on HA Pi at 192.168.1.35:1883
   - Credentials: frigate / frigate_mqtt_2026
   - Bridge between Frigate and ARIA

3. **ARIA Presence Module** (`aria/modules/presence.py`)
   - Subscribes to Frigate MQTT events (person/face detection)
   - Subscribes to HA WebSocket (motion sensors, lights, dimmers, device trackers, door sensors)
   - Feeds signals into `BayesianOccupancy._bayesian_fuse()` with per-type weights
   - Writes to `CACHE_PRESENCE` every 30 seconds
   - Tracks identified persons with room + confidence + last_seen
   - Auto-exposed at `GET /api/cache/presence`

### Signal Types & Weights

| Signal | Weight | Decay | Source |
|--------|--------|-------|--------|
| `camera_person` | 0.95 | 120s | Frigate person detection |
| `camera_face` | 1.00 | none | Frigate face recognition (identified person) |
| `light_interaction` | 0.70 | 600s | HA light on/off |
| `dimmer_press` | 0.85 | 300s | Hue dimmer physical button |
| `motion` | 0.95 | 300s | HA binary_sensor motion (pre-existing) |
| `door` | 0.70 | 120s | HA binary_sensor door (pre-existing) |
| `device_tracker` | 0.90 | 300s | HA person/device tracker (pre-existing) |

### Camera Inventory

| Camera | Type | IP | Status in Frigate |
|--------|------|----|--------------------|
| Driveway (I91BF) | ONVIF | 192.168.1.253 | Active |
| Backyard (I91BF) | ONVIF | 192.168.1.125 | Active |
| Axis M4308-PLE | ONVIF/Axis | 192.168.1.212 | Active |
| Front Doorbell | UniFi Protect | 192.168.1.93 | Disabled (needs RTSP alias) |
| G4 Instant Carter's | UniFi Protect | 192.168.1.216 | Disabled (needs RTSP alias) |
| G4 Instant Collins's | UniFi Protect | 192.168.1.28 | Disabled (needs RTSP alias) |
| Pool Camera | UniFi Protect | 192.168.3.196 | Disabled (needs RTSP alias) |

### Room Resolution

Entity → room mapping follows the HA data model lesson (entity→device→area chain):
1. Check discovery cache for entity's `area_id`
2. Fall back to device's `area_id`
3. Fall back to entity_id pattern matching (bedroom, front_door, etc.)

## Decisions Made

1. **MotionAware ruled out** — Requires Hue Bridge Pro (user has V2). MotionAware only provides binary motion per zone anyway — no spatial/per-light data. Zigbee signal analysis not possible from outside the bridge firmware.

2. **Frigate over custom CV** — 30k GitHub stars, built-in face recognition, HA integration, MQTT-native. No reason to build custom.

3. **Feed into ARIA, not HA Bayesian sensors** — ARIA already has BayesianOccupancy with log-odds fusion, per-sensor decay, confidence classification. Creating HA Bayesian binary sensors would duplicate this.

4. **CPU-only Frigate** — No GPU on workstation. person detection + face recognition both run on CPU. Performance is adequate for 3 active cameras.

5. **Face collection strategy** — Frigate auto-saves unknown faces (200 attempts). User labels them in Frigate UI. Recognition threshold: 0.85. Unknown score: 0.6. Continuous refinement as more labeled samples accumulate.

6. **Non-fatal module** — Presence module failure doesn't take down the hub. MQTT disconnects retry with exponential backoff (5s → 60s max).

## Remaining Work

- [ ] Enable UniFi Protect RTSP streams (4 cameras need aliases from Protect UI — deferred, do when convenient)
- [ ] Add Frigate HA integration (HACS or manual)
- [ ] Restart ARIA hub to load presence module
- [ ] End-to-end verification: person walks past camera → MQTT event → ARIA cache update → API response
- [ ] Dashboard presence card (show per-room occupancy + identified persons)
- [ ] Face labeling workflow documentation

## Files Changed

| File | Change |
|------|--------|
| `aria/modules/presence.py` | **New** — Full presence module (~545 lines) |
| `tests/hub/test_presence.py` | **New** — 51 tests |
| `aria/engine/analysis/occupancy.py` | Added 4 signal types to SENSOR_CONFIG |
| `aria/hub/constants.py` | Added CACHE_PRESENCE |
| `aria/hub/config_defaults.py` | Added 5 presence config parameters |
| `aria/capabilities.py` | Added PresenceModule to capability collector |
| `aria/cli.py` | Added presence module registration (non-fatal) |
| `pyproject.toml` | Added aiomqtt dependency |
| `~/frigate/config/config.yml` | Frigate camera + face recognition config |
| `~/frigate/docker-compose.yml` | Frigate Docker deployment |
