# Dynamic Entity Discovery with Lifecycle Management

**Date:** 2026-02-17
**Status:** Approved
**Scope:** DiscoveryModule (entity/device/area lifecycle) + PresenceModule (camera discovery)

## Problem

All HA entity data in ARIA is handled with a full-replace strategy on re-discovery. This causes:

1. **Transient data loss** — If HA returns fewer entities during a bad moment, the entire cache is wiped and replaced with incomplete data
2. **Hard-coded camera mappings** — `DEFAULT_CAMERA_ROOMS` in presence.py requires code changes when cameras are added/removed
3. **Hard-coded room parsing** — `_resolve_room()` uses string matching (`"bedroom" in eid`) as a fallback
4. **No lifecycle visibility** — Consumers can't distinguish between "entity exists" and "entity was seen recently"

## Design

### Entity Lifecycle (universal, all three caches)

```
Discovered → Active → [missing from discovery] → Stale → [TTL expires] → Archived
                                                    ↑                         │
                                                    └─── [re-discovered] ─────┘
```

**Applies to:** `entities`, `devices`, `areas` caches.

### Lifecycle Metadata

Each entity/device/area record gains a `_lifecycle` field:

```python
{
    "entity_id": "camera.driveway",
    "area_id": "driveway",
    "device_id": "abc123",
    # ... all existing fields preserved ...
    "_lifecycle": {
        "status": "active",           # active | stale | archived
        "first_discovered": "2026-02-17T10:00:00",
        "last_seen_in_discovery": "2026-02-17T10:00:00",
        "stale_since": null,          # set when missing from discovery
        "archived_at": null           # set when TTL expires
    }
}
```

### Merge Behavior (re-discovery)

On each discovery run:

1. **New entities** (not in existing cache) → added with `status: active`, `first_discovered: now`
2. **Existing entities** (in both) → fields updated, `last_seen_in_discovery: now`, `status: active` (clears stale/archived)
3. **Missing entities** (in cache but not in latest discovery) → marked `stale_since: now` if currently active. No deletion.
4. **Stale entities past TTL** → moved to `status: archived`. Periodic check runs alongside discovery.

### Staleness Timer + Archive

- **Default TTL:** 72 hours (`discovery.stale_ttl_hours` config key)
- **While stale:** Entity is still usable by consumers (camera events still map to rooms, ML still trains on it)
- **After archive:** Entity excluded from active consumer queries. Consumers filter on `_lifecycle.status != "archived"`
- **Auto-promotion:** If an archived entity reappears in discovery OR sends events (MQTT for cameras), it immediately restores to `status: active`

### Config Keys

| Key | Default | Type | Purpose |
|-----|---------|------|---------|
| `discovery.stale_ttl_hours` | `72` | float | Hours before stale entities are archived |
| `presence.camera_rooms` | `""` | string | Manual camera:room overrides (existing, unchanged) |

### Priority Chain (camera room resolution)

```
Config override (presence.camera_rooms)
  → Discovery cache (entity→device→area, cumulative)
    → Camera name as-is (fallback for unresolved cameras)
```

### PresenceModule Changes

1. **Delete** `DEFAULT_CAMERA_ROOMS` dict entirely
2. **Delete** hard-coded room name parsing in `_resolve_room()` (`"bedroom" in eid`, etc.)
3. **Add** `_discover_camera_rooms()` — queries entities cache for `camera.*`, resolves area via device→area chain
4. **Add** subscription to `discovery_complete` event → refresh camera mapping (merge, not replace)
5. **Preserve** config override: `presence.camera_rooms` CSV always takes priority

### DiscoveryModule Changes

1. **Replace** `_store_discovery_results()` full-replace with merge logic + lifecycle metadata
2. **Add** `_mark_stale()` — marks entities missing from latest discovery as stale
3. **Add** `_archive_expired()` — periodic check, moves stale entities past TTL to archived
4. **Add** `_promote_if_archived()` — restores archived entities that reappear

### Consumer Impact

All modules reading entities/devices/areas cache automatically get lifecycle metadata:

- **Organic discovery** — can filter `status == "active"` for clustering
- **ML engine** — can exclude archived entities from training data
- **Activity labeler** — reads from entity cache, now lifecycle-aware
- **Presence** — camera mapping is discovery-driven, no hard-coded data

### Files Changed

| File | Change |
|------|--------|
| `aria/modules/discovery.py` | Merge logic, lifecycle metadata, stale/archive management |
| `aria/hub/config_defaults.py` | Add `discovery.stale_ttl_hours` config key |
| `aria/modules/presence.py` | Delete hard-coded data, add discovery-based camera mapping |
| `tests/hub/test_discover.py` | Merge, staleness, archive, auto-promotion tests |
| `tests/hub/test_presence.py` | Discovery-based camera mapping tests |

### What We're NOT Doing

- No UI for managing archived entities (YAGNI — config API + logs sufficient)
- No manual "force archive" API (can be added later if needed)
- No changes to capabilities merge logic (already works correctly)
- No changes to MQTT topic subscriptions (protocol constants, not data)
