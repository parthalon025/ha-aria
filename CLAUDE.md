# ha-intelligence-hub

Home Assistant discovery module - MVP for validating dynamic capability detection.

## Context

**Parent project:** `ha-intelligence` at `~/Documents/projects/ha-intelligence/`
**Design doc:** `~/Documents/docs/plans/2026-02-11-ha-intelligence-hub-design.md`
**Lean roadmap:** `~/Documents/docs/plans/2026-02-11-ha-hub-lean-roadmap.md`

**Current phase:** Phase 0 (Discovery MVP) - proving hypothesis before building full hub

## Phase 0 Goal

Build standalone `discover.py` that scans HA instance and outputs capabilities JSON.

**Hypothesis:** Dynamic discovery > hardcoded extraction
**Success:** Finds ≥3 new capabilities with ≥10 entities each
**Timeline:** 3-5 days
**Decision:** Continue to Phase 1 (integrate) or stop (hypothesis false)

## Structure

```
bin/
  discover.py        # Main script (~300 lines)
tests/
  test_discover.py   # Unit tests
docs/
  api-examples.md    # HA API response samples
```

## Key Implementation Details

**HA APIs used:**
- REST: `/api/states`, `/api/config`, `/api/services`
- WebSocket: `config/entity_registry/list`, `config/device_registry/list`, `config/area_registry/list`

**Capability detection rules:**
- `power_monitoring`: domain=sensor, device_class=power, unit=W
- `lighting`: domain=light
- `occupancy`: domain=person OR device_tracker
- `climate`: domain=climate
- `ev_charging`: domain=sensor, attributes contain "battery" + "charger"
- `battery_devices`: any entity with battery_level attribute
- `motion`: domain=binary_sensor, device_class=motion
- `doors_windows`: domain=binary_sensor, device_class in [door, window]
- `locks`: domain=lock
- `media`: domain=media_player
- `vacuum`: domain=vacuum

## Environment

- **HA instance:** 192.168.1.35:8123 (HAOS on Raspberry Pi)
- **Env vars:** HA_URL, HA_TOKEN from `~/.env`
- **Python:** 3.12 (system default)
- **Deps:** stdlib only (no external packages for MVP)

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Manual test against live HA
. ~/.env && ./bin/discover.py > /tmp/out.json
cat /tmp/out.json | jq '.capabilities | keys'

# Validation: compare with ha-intelligence extraction
# Expected: discovery finds domains ha-intelligence doesn't track
```

## Gotchas

- HA WebSocket requires authentication (send `auth` message with token)
- WebSocket commands are async - must wait for `result` message with matching `id`
- Some entities are normally unavailable (update, tts, stt domains) - filter these
- Entity registry may have hidden/disabled entities - include metadata to show this
