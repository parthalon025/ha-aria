#!/usr/bin/env python3
"""Home Assistant Discovery Module - Phase 0 MVP

Scans HA instance via REST + WebSocket to discover all entities, devices, areas,
and detect capabilities for adaptive predictions.

Usage:
  . ~/.env && ./bin/discover.py > /tmp/capabilities.json
  cat /tmp/capabilities.json | jq '.capabilities | keys'
"""

import json
import os
import random
import socket
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
from base64 import b64encode
from datetime import UTC, datetime

# === Config ===
HA_URL = os.environ.get("HA_URL", "")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Domains to exclude from unavailable counts (normally unavailable)
UNAVAILABLE_EXCLUDE_DOMAINS = {"update", "tts", "stt"}


def log(msg):
    """Print to stderr for debugging."""
    print(f"[discover] {msg}", file=__import__("sys").stderr)


# === REST API Client ===


def fetch_rest_api(endpoint, retries=3):
    """Fetch data from HA REST API with retries and exponential backoff."""
    url = f"{HA_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise Exception(f"Authentication failed: {e}") from e
            log(f"HTTP error {e.code} on {endpoint}, attempt {attempt + 1}/{retries}")
        except urllib.error.URLError as e:
            log(f"Connection error on {endpoint}, attempt {attempt + 1}/{retries}: {e}")
        except Exception as e:
            log(f"Unexpected error on {endpoint}, attempt {attempt + 1}/{retries}: {e}")

        if attempt < retries - 1:
            backoff = 2**attempt
            time.sleep(backoff)

    raise Exception(f"Failed to fetch {endpoint} after {retries} attempts")


# === WebSocket Client (Manual Implementation) ===


def create_websocket_handshake(host, path):
    """Create WebSocket handshake HTTP request."""
    key = b64encode(random.randbytes(16)).decode("utf-8")

    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )

    return request.encode("utf-8"), key


def parse_websocket_frame(sock):  # noqa: C901
    """Parse a WebSocket frame from socket."""
    # Read first 2 bytes
    header = sock.recv(2)
    if len(header) < 2:
        raise Exception("Connection closed")

    byte1, byte2 = header[0], header[1]

    # FIN bit and opcode
    (byte1 & 0b10000000) >> 7
    opcode = byte1 & 0b00001111

    # Mask bit and payload length
    masked = (byte2 & 0b10000000) >> 7
    payload_len = byte2 & 0b01111111

    # Extended payload length
    if payload_len == 126:
        ext_len = sock.recv(2)
        payload_len = struct.unpack(">H", ext_len)[0]
    elif payload_len == 127:
        ext_len = sock.recv(8)
        payload_len = struct.unpack(">Q", ext_len)[0]

    # Masking key (if present)
    if masked:
        mask = sock.recv(4)

    # Payload data
    payload = b""
    while len(payload) < payload_len:
        chunk = sock.recv(payload_len - len(payload))
        if not chunk:
            raise Exception("Connection closed during payload read")
        payload += chunk

    # Unmask if needed
    if masked:
        payload = bytes([payload[i] ^ mask[i % 4] for i in range(len(payload))])

    # Handle opcodes
    if opcode == 0x1:  # Text frame
        return payload.decode("utf-8")
    elif opcode == 0x8:  # Close frame
        raise Exception("WebSocket closed by server")
    elif opcode == 0x9:  # Ping frame
        # Send pong
        send_websocket_frame(sock, b"", opcode=0xA)
        return None
    else:
        return None


def send_websocket_frame(sock, data, opcode=0x1):
    """Send a WebSocket frame."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    frame = bytearray()

    # FIN bit + opcode
    frame.append(0b10000000 | opcode)

    # Mask bit + payload length
    payload_len = len(data)
    if payload_len < 126:
        frame.append(0b10000000 | payload_len)
    elif payload_len < 65536:
        frame.append(0b10000000 | 126)
        frame.extend(struct.pack(">H", payload_len))
    else:
        frame.append(0b10000000 | 127)
        frame.extend(struct.pack(">Q", payload_len))

    # Masking key (required for client-to-server)
    mask = random.randbytes(4)
    frame.extend(mask)

    # Masked payload
    masked_data = bytes([data[i] ^ mask[i % 4] for i in range(len(data))])
    frame.extend(masked_data)

    sock.sendall(frame)


def websocket_connect(url, token):  # noqa: C901, PLR0912, PLR0915
    """Connect to WebSocket, authenticate, return socket."""
    # Parse URL
    if url.startswith("http://"):
        url = url.replace("http://", "ws://")
    elif url.startswith("https://"):
        url = url.replace("https://", "wss://")

    # Extract host and path
    if url.startswith("ws://"):
        use_ssl = False
        url = url[5:]
    elif url.startswith("wss://"):
        use_ssl = True
        url = url[6:]

    if "/" in url:
        host, path = url.split("/", 1)
        path = "/" + path
    else:
        host = url
        path = "/api/websocket"

    # Handle port
    if ":" in host:
        host, port = host.rsplit(":", 1)
        port = int(port)
    else:
        port = 443 if use_ssl else 80

    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect((host, port))

    # Wrap in SSL if needed
    if use_ssl:
        context = ssl.create_default_context()
        sock = context.wrap_socket(sock, server_hostname=host)

    # WebSocket handshake
    handshake, key = create_websocket_handshake(host, path)
    sock.sendall(handshake)

    # Read handshake response
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(1024)
        if not chunk:
            raise Exception("Connection closed during handshake")
        response += chunk

    # Verify handshake
    if b"HTTP/1.1 101" not in response:
        raise Exception(f"WebSocket handshake failed: {response.decode('utf-8', errors='ignore')}")

    # Wait for auth_required message
    msg = parse_websocket_frame(sock)
    while msg is None:  # Skip ping/pong
        msg = parse_websocket_frame(sock)

    auth_msg = json.loads(msg)
    if auth_msg.get("type") != "auth_required":
        raise Exception(f"Expected auth_required, got: {auth_msg}")

    # Send auth message
    auth_payload = json.dumps({"type": "auth", "access_token": token})
    send_websocket_frame(sock, auth_payload)

    # Wait for auth response
    msg = parse_websocket_frame(sock)
    while msg is None:
        msg = parse_websocket_frame(sock)

    auth_response = json.loads(msg)
    if auth_response.get("type") == "auth_invalid":
        raise Exception("Authentication failed")
    elif auth_response.get("type") != "auth_ok":
        raise Exception(f"Unexpected auth response: {auth_response}")

    return sock


def fetch_websocket_data(command_type, retries=3):
    """Fetch data from HA WebSocket API with retries."""
    for attempt in range(retries):
        sock = None
        try:
            sock = websocket_connect(HA_URL, HA_TOKEN)

            # Send command
            request_id = random.randint(1, 1000000)
            command = json.dumps({"id": request_id, "type": command_type})
            send_websocket_frame(sock, command)

            # Wait for response
            while True:
                msg = parse_websocket_frame(sock)
                if msg is None:  # Ping/pong
                    continue

                response = json.loads(msg)
                if response.get("id") == request_id:
                    if response.get("success"):
                        return response.get("result", [])
                    else:
                        raise Exception(f"WebSocket command failed: {response.get('error')}")

        except Exception as e:
            log(f"WebSocket error on {command_type}, attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
        finally:
            if sock:
                try:
                    # Send close frame
                    send_websocket_frame(sock, b"", opcode=0x8)
                    sock.close()
                except Exception:
                    pass

    raise Exception(f"Failed to fetch {command_type} after {retries} attempts")


# === Capability Detection ===


def detect_capabilities(states, entity_registry, device_registry):  # noqa: C901, PLR0912
    """Detect capabilities based on discovered entities."""
    capabilities = {}

    # Power monitoring
    power_entities = [
        e["entity_id"]
        for e in states
        if e.get("attributes", {}).get("device_class") == "power"
        and e.get("attributes", {}).get("unit_of_measurement") == "W"
    ]
    if power_entities:
        capabilities["power_monitoring"] = {
            "available": True,
            "entities": power_entities,
            "total_count": len(power_entities),
            "measurement_unit": "W",
            "can_predict": True,
        }

    # Lighting
    light_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("light.")]
    if light_entities:
        supports_color = sum(
            1 for e in states if e["entity_id"].startswith("light.") and "rgb_color" in e.get("attributes", {})
        )
        supports_color_temp = sum(
            1 for e in states if e["entity_id"].startswith("light.") and "color_temp" in e.get("attributes", {})
        )
        supports_brightness = sum(
            1 for e in states if e["entity_id"].startswith("light.") and "brightness" in e.get("attributes", {})
        )

        capabilities["lighting"] = {
            "available": True,
            "entities": light_entities,
            "total_count": len(light_entities),
            "supports_color": supports_color,
            "supports_color_temp": supports_color_temp,
            "supports_brightness": supports_brightness,
            "can_predict": True,
        }

    # Occupancy
    person_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("person.")]
    device_tracker_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("device_tracker.")]
    if person_entities or device_tracker_entities:
        capabilities["occupancy"] = {
            "available": True,
            "method": [],
            "people": person_entities,
            "tracked_devices": len(device_tracker_entities),
            "can_predict": True,
        }
        if person_entities:
            capabilities["occupancy"]["method"].append("person")
        if device_tracker_entities:
            capabilities["occupancy"]["method"].append("device_tracker")

    # Climate
    climate_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("climate.")]
    if climate_entities:
        modes = set()
        for e in states:
            if e["entity_id"].startswith("climate."):
                hvac_modes = e.get("attributes", {}).get("hvac_modes", [])
                modes.update(hvac_modes)

        capabilities["climate"] = {
            "available": True,
            "entities": climate_entities,
            "total_count": len(climate_entities),
            "modes": list(modes),
            "can_predict": True,
        }

    # EV Charging
    ev_entities = [
        e["entity_id"]
        for e in states
        if ("battery" in e["entity_id"].lower() and "vehicle" in e["entity_id"].lower())
        or "tars" in e["entity_id"].lower()
    ]
    if ev_entities:
        capabilities["ev_charging"] = {
            "available": True,
            "entities": ev_entities,
            "vehicle_count": 1,  # Simplified for MVP
            "can_predict": True,
        }

    # Battery devices
    battery_entities = [
        e["entity_id"]
        for e in states
        if "battery" in e.get("attributes", {}) or "battery_level" in e.get("attributes", {})
    ]
    if battery_entities:
        capabilities["battery_devices"] = {
            "available": True,
            "entities": battery_entities,
            "total_count": len(battery_entities),
            "can_predict": True,
        }

    # Motion sensors
    motion_entities = [e["entity_id"] for e in states if e.get("attributes", {}).get("device_class") == "motion"]
    if motion_entities:
        capabilities["motion"] = {
            "available": True,
            "entities": motion_entities,
            "total_count": len(motion_entities),
            "can_predict": False,
        }

    # Doors/Windows
    door_window_entities = [
        e["entity_id"] for e in states if e.get("attributes", {}).get("device_class") in ["door", "window"]
    ]
    if door_window_entities:
        capabilities["doors_windows"] = {
            "available": True,
            "entities": door_window_entities,
            "total_count": len(door_window_entities),
            "can_predict": False,
        }

    # Locks
    lock_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("lock.")]
    if lock_entities:
        capabilities["locks"] = {
            "available": True,
            "entities": lock_entities,
            "total_count": len(lock_entities),
            "can_predict": False,
        }

    # Media
    media_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("media_player.")]
    if media_entities:
        capabilities["media"] = {
            "available": True,
            "entities": media_entities,
            "total_count": len(media_entities),
            "can_predict": False,
        }

    # Vacuum
    vacuum_entities = [e["entity_id"] for e in states if e["entity_id"].startswith("vacuum.")]
    if vacuum_entities:
        capabilities["vacuum"] = {
            "available": True,
            "entities": vacuum_entities,
            "total_count": len(vacuum_entities),
            "can_predict": False,
        }

    return capabilities


# === Main Discovery ===


def discover_all():  # noqa: PLR0915
    """Run full discovery - fetch all data from HA."""
    log("Starting discovery...")

    discovery = {
        "discovery_timestamp": datetime.now(UTC).isoformat(),
        "ha_version": None,
        "entity_count": 0,
        "capabilities": {},
        "entities": {},
        "devices": {},
        "areas": {},
        "integrations": [],
        "labels": [],
    }

    # Fetch REST data
    log("Fetching states...")
    states = fetch_rest_api("/api/states")
    discovery["entity_count"] = len(states)
    log(f"Found {len(states)} entities")

    log("Fetching config...")
    config = fetch_rest_api("/api/config")
    discovery["ha_version"] = config.get("version")
    log(f"HA version: {discovery['ha_version']}")

    # Fetch WebSocket data (with delays to avoid connection issues)
    log("Fetching entity registry...")
    entity_registry = fetch_websocket_data("config/entity_registry/list")
    log(f"Found {len(entity_registry)} registry entries")
    time.sleep(1)  # Small delay between WebSocket calls

    log("Fetching device registry...")
    device_registry = fetch_websocket_data("config/device_registry/list")
    log(f"Found {len(device_registry)} devices")
    time.sleep(1)

    log("Fetching area registry...")
    area_registry = fetch_websocket_data("config/area_registry/list")
    log(f"Found {len(area_registry)} areas")
    time.sleep(1)

    # Try to fetch labels (may not exist in all HA versions)
    try:
        log("Fetching label registry...")
        label_registry = fetch_websocket_data("config/label_registry/list")
        log(f"Found {len(label_registry)} labels")
        discovery["labels"] = label_registry
    except Exception:
        log("Labels not available (HA version may not support them)")
        discovery["labels"] = []

    # Detect capabilities
    log("Detecting capabilities...")
    capabilities = detect_capabilities(states, entity_registry, device_registry)
    discovery["capabilities"] = capabilities
    log(f"Detected {len(capabilities)} capabilities")

    # Process entities (combine state + registry metadata)
    log("Processing entities...")
    entity_map = {e["entity_id"]: e for e in entity_registry}
    for state in states:
        entity_id = state["entity_id"]
        registry_entry = entity_map.get(entity_id, {})

        discovery["entities"][entity_id] = {
            "entity_id": entity_id,
            "state": state.get("state"),
            "attributes": state.get("attributes", {}),
            "last_changed": state.get("last_changed"),
            "last_updated": state.get("last_updated"),
            "friendly_name": registry_entry.get("name") or state.get("attributes", {}).get("friendly_name"),
            "device_id": registry_entry.get("device_id"),
            "area_id": registry_entry.get("area_id"),
            "labels": registry_entry.get("labels", []),
            "domain": entity_id.split(".")[0],
            "device_class": state.get("attributes", {}).get("device_class"),
            "unit_of_measurement": state.get("attributes", {}).get("unit_of_measurement"),
            "disabled": registry_entry.get("disabled_by") is not None,
            "hidden": registry_entry.get("hidden_by") is not None,
            "icon": registry_entry.get("icon") or state.get("attributes", {}).get("icon"),
        }

    # Process devices
    log("Processing devices...")
    for device in device_registry:
        device_id = device["id"]
        discovery["devices"][device_id] = {
            "device_id": device_id,
            "name": device.get("name"),
            "manufacturer": device.get("manufacturer"),
            "model": device.get("model"),
            "area_id": device.get("area_id"),
            "via_device_id": device.get("via_device_id"),
        }

    # Backfill entity area_id from parent device when entity has no direct area
    log("Resolving entity areas from devices...")
    backfilled = 0
    for _eid, entity in discovery["entities"].items():
        if not entity.get("area_id") and entity.get("device_id"):
            device = discovery["devices"].get(entity["device_id"])
            if device and device.get("area_id"):
                entity["area_id"] = device["area_id"]
                backfilled += 1
    log(f"Backfilled area_id for {backfilled} entities from their parent devices")

    # Process areas
    log("Processing areas...")
    for area in area_registry:
        area_id = area["area_id"]
        # Count entities in this area (includes device-inherited areas)
        entities_in_area = [eid for eid, e in discovery["entities"].items() if e.get("area_id") == area_id]

        discovery["areas"][area_id] = {
            "area_id": area_id,
            "name": area.get("name"),
            "entity_count": len(entities_in_area),
        }

    # Extract integrations (unique domains)
    log("Extracting integrations...")
    domains = {}
    for entity_id in discovery["entities"]:
        domain = entity_id.split(".")[0]
        if domain not in UNAVAILABLE_EXCLUDE_DOMAINS:
            domains[domain] = domains.get(domain, 0) + 1

    discovery["integrations"] = [
        {"domain": domain, "entity_count": count}
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True)
    ]

    # Set top-level counts for metadata consumers
    discovery["device_count"] = len(discovery["devices"])
    discovery["area_count"] = len(discovery["areas"])

    log("Discovery complete!")
    return discovery


if __name__ == "__main__":
    try:
        if not HA_URL:
            print("Error: HA_URL environment variable not set", file=sys.stderr)
            print("Usage: . ~/.env && ./bin/discover.py", file=sys.stderr)
            sys.exit(1)
        if not HA_TOKEN:
            print("Error: HA_TOKEN environment variable not set", file=sys.stderr)
            print("Usage: . ~/.env && ./bin/discover.py", file=sys.stderr)
            sys.exit(1)

        result = discover_all()
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
