# Frigate Camera Name Resolution Design

**Date:** 2026-02-17
**Status:** Approved

## Problem

Frigate MQTT events use short camera names (`backyard`, `driveway`). HA entity registry uses long entity IDs (`camera.backyard_high_resolution_channel`). The dynamic discovery maps HA entity names to rooms, but MQTT lookups use Frigate short names — causing a miss.

## Solution

Add a Frigate camera name resolver that:

1. Fetches Frigate's camera list from `/api/config` (already called for face config)
2. For each Frigate camera name, substring-matches it against HA `camera.*` entity IDs
3. Adds the Frigate short name as an alias key in `camera_rooms`, pointing to the same HA area

### Data Flow

```
Frigate /api/config → camera names: [backyard, driveway, pool, ...]
HA entity cache → camera.backyard_high_resolution_channel → area: pool

Match: "backyard" is substring of "backyard_high_resolution_channel"
Result: camera_rooms["backyard"] = "pool"  (alias)
        camera_rooms["backyard_high_resolution_channel"] = "pool"  (original)
```

### Matching Rules

1. For each Frigate camera name `F`, find HA entities where `camera.{X}` and `F` is a substring of `X`
2. If exactly one match: use it
3. If multiple matches: use the shortest entity name (most specific)
4. If no match: skip (logged as warning)
5. Config overrides always win (existing behavior)

### Integration Points

- `_fetch_face_config()` already calls Frigate `/api/config` — extract camera names from the same response
- `_refresh_camera_rooms()` adds alias keys after discovery merge
- No new API calls, no new dependencies

### What Changes

- `presence.py`: Store Frigate camera names from config fetch, add alias keys during refresh
- Tests: Verify alias resolution, verify MQTT events resolve to correct rooms
