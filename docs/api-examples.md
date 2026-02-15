# HA API Response Examples

## In Plain English

This is a collection of real answers that Home Assistant gives when ARIA asks it questions. It is like keeping a copy of a form letter so you know exactly what format the reply will come in before you send the request.

## Why This Exists

ARIA needs to talk to Home Assistant's API to learn about your home -- what devices exist, what rooms they are in, and what states they are in right now. The API responses have specific structures that change between HA versions, and getting the parsing wrong means ARIA sees garbage data. This document captures actual live responses so developers can build and test against real formats instead of guessing from documentation that may be outdated.

Documentation of actual API responses from HA 2026.2.1 (justin-linux instance).

## Discovery Summary (Live Run)

**Run:** 2026-02-11T11:22
**Instance:** 192.168.1.35:8123 (HAOS 2026.2.1)

```
Entities: 3,065 (states)
Entity Registry: 4,568 (includes disabled/hidden)
Devices: 758
Areas: 26
Labels: 4
Service Domains: 68
```

## Capabilities Detected

```json
{
  "power_monitoring": 21 entities,
  "lighting": 73 entities,
  "occupancy": 563 entities (person + device_tracker),
  "climate": 4 entities,
  "ev_charging": 42 entities,
  "battery_devices": 32 entities,
  "motion": 13 entities,
  "doors_windows": 9 entities,
  "locks": 9 entities,
  "media": 18 entities,
  "vacuum": 3 entities
}
```

## REST API Examples

### GET /api/states

Returns all entity states (current snapshot).

**Response:** Array of entity state objects

```json
[
  {
    "entity_id": "sensor.power_meter",
    "state": "150.5",
    "attributes": {
      "friendly_name": "Power Meter",
      "device_class": "power",
      "unit_of_measurement": "W"
    },
    "last_changed": "2026-02-11T11:20:00.000Z",
    "last_updated": "2026-02-11T11:20:00.000Z"
  },
  {
    "entity_id": "light.living_room",
    "state": "on",
    "attributes": {
      "friendly_name": "Living Room Light",
      "brightness": 255,
      "color_mode": "brightness"
    },
    "last_changed": "2026-02-11T10:30:00.000Z",
    "last_updated": "2026-02-11T11:20:00.000Z"
  }
]
```

### GET /api/config

Returns HA configuration.

**Response:** Config object

```json
{
  "version": "2026.2.1",
  "location_name": "Home",
  "time_zone": "America/New_York",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "elevation": 10,
  "unit_system": {
    "length": "mi",
    "mass": "lb",
    "temperature": "°F",
    "volume": "gal"
  },
  "currency": "USD",
  "config_dir": "/config",
  "whitelist_external_dirs": ["/config/www"],
  "components": ["automation", "light", "sensor", ...]
}
```

### GET /api/services

Returns available service domains and their services.

**Response:** Array of service domain objects

```json
[
  {
    "domain": "light",
    "services": {
      "turn_on": {
        "name": "Turn on",
        "description": "Turn on one or more lights",
        "fields": {
          "brightness": {
            "description": "Brightness (0-255)",
            "example": 120
          }
        }
      },
      "turn_off": {
        "name": "Turn off",
        "description": "Turn off one or more lights"
      }
    }
  }
]
```

## WebSocket API Examples

### Protocol Flow

1. Client connects to `ws://192.168.1.35:8123/api/websocket`
2. Server sends: `{"type": "auth_required", "ha_version": "2026.2.1"}`
3. Client sends: `{"type": "auth", "access_token": "TOKEN"}`
4. Server sends: `{"type": "auth_ok", "ha_version": "2026.2.1"}`
5. Client sends: `{"id": 1, "type": "config/entity_registry/list"}`
6. Server sends: `{"id": 1, "type": "result", "success": true, "result": [...]}`
7. Client sends close frame

### config/entity_registry/list

Returns all entity registry entries (includes disabled/hidden metadata).

**Command:**
```json
{"id": 1, "type": "config/entity_registry/list"}
```

**Response:**
```json
{
  "id": 1,
  "type": "result",
  "success": true,
  "result": [
    {
      "entity_id": "light.living_room",
      "platform": "hue",
      "device_id": "abc123",
      "area_id": "living_room",
      "disabled_by": null,
      "hidden_by": null,
      "original_name": "Hue color lamp",
      "unique_id": "00:17:88:01:00:00:00:01-0b"
    },
    {
      "entity_id": "sensor.disabled_sensor",
      "platform": "template",
      "device_id": null,
      "area_id": null,
      "disabled_by": "user",
      "hidden_by": null,
      "original_name": "Disabled Sensor",
      "unique_id": "template_disabled_1"
    }
  ]
}
```

**Note:** Entity registry has MORE entries than /api/states because it includes disabled and hidden entities.

### config/device_registry/list

Returns all device registry entries.

**Command:**
```json
{"id": 2, "type": "config/device_registry/list"}
```

**Response:**
```json
{
  "id": 2,
  "type": "result",
  "success": true,
  "result": [
    {
      "id": "abc123",
      "name": "Philips Hue Bridge",
      "manufacturer": "Signify Netherlands B.V.",
      "model": "BSB002",
      "sw_version": "1.50.0",
      "area_id": "utility_room",
      "config_entries": ["hue_entry_1"],
      "connections": [["mac", "00:17:88:12:34:56"]],
      "identifiers": [["hue", "001788123456"]],
      "via_device_id": null,
      "disabled_by": null
    }
  ]
}
```

### config/area_registry/list

Returns all area definitions.

**Command:**
```json
{"id": 3, "type": "config/area_registry/list"}
```

**Response:**
```json
{
  "id": 3,
  "type": "result",
  "success": true,
  "result": [
    {
      "area_id": "living_room",
      "name": "Living Room",
      "picture": null,
      "icon": null,
      "aliases": ["lounge", "family room"]
    },
    {
      "area_id": "bedroom",
      "name": "Master Bedroom",
      "picture": null,
      "icon": "mdi:bed",
      "aliases": []
    }
  ]
}
```

### config/label_registry/list

Returns all label definitions (optional, may not exist in all HA versions).

**Command:**
```json
{"id": 4, "type": "config/label_registry/list"}
```

**Response:**
```json
{
  "id": 4,
  "type": "result",
  "success": true,
  "result": [
    {
      "label_id": "energy_monitoring",
      "name": "Energy Monitoring",
      "color": "#FF5722",
      "icon": "mdi:flash"
    }
  ]
}
```

**Note:** Label registry was added in HA 2024.x. Older instances may return an error.

## Implementation Notes

### REST API

- **Authentication:** Bearer token in `Authorization` header
- **Timeout:** 30 seconds
- **Retry logic:** 3 attempts with exponential backoff (2^n seconds)
- **Non-retryable errors:** 401, 403 (auth failures)

### WebSocket API

- **Protocol:** RFC 6455 WebSocket
- **Authentication:** Two-step (auth_required → auth with token → auth_ok)
- **Framing:** Client MUST mask outgoing frames
- **Message format:** JSON with `id` field for matching requests/responses
- **Connection:** One-shot (connect → auth → fetch → close)

### Capability Detection Rules

See `bin/discover.py::detect_capabilities()` for full logic. Key patterns:

- **power_monitoring:** `domain=sensor AND device_class=power AND unit=W|kW`
- **lighting:** `domain=light`
- **occupancy:** `domain=person OR device_tracker`
- **climate:** `domain=climate`
- **ev_charging:** `domain=sensor AND (battery|charger in entity_id OR friendly_name)`
- **battery_devices:** `device_class=battery OR battery_level in attributes`
- **motion:** `domain=binary_sensor AND device_class=motion`
- **doors_windows:** `domain=binary_sensor AND device_class IN [door, window]`
- **locks:** `domain=lock`
- **media:** `domain=media_player`
- **vacuum:** `domain=vacuum`

### Data Volume

From live HA instance (3,065 entities):
- JSON output: ~2.6 MB
- Processing time: ~5-10 seconds
- Network round-trips: 3 REST + 4 WebSocket = 7 API calls
