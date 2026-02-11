# HA Intelligence Hub - Discovery Module (MVP)

> **Status:** Phase 0 - Validating discovery hypothesis
> **Goal:** Prove that dynamic HA discovery reveals capabilities not tracked by hardcoded extraction

## Quick Start

```bash
# Run discovery against your HA instance
. ~/.env && ./bin/discover.py > /tmp/capabilities.json

# View discovered capabilities
cat /tmp/capabilities.json | jq '.capabilities | keys'

# See details for a specific capability
cat /tmp/capabilities.json | jq '.capabilities.lighting'
```

## What This Does

Scans your Home Assistant instance via REST + WebSocket APIs to discover:
- All entities (states + metadata)
- All devices (manufacturer, model, area assignments)
- All areas, labels, zones
- All integrations, automations, scenes
- Detected capabilities (what can be predicted/automated)

## Output

JSON structure:
```json
{
  "discovery_timestamp": "2026-02-11T...",
  "ha_version": "2026.2.1",
  "entity_count": 3065,
  "capabilities": {
    "power_monitoring": { "available": true, "entities": [...], "total_count": 12 },
    "lighting": { "available": true, "entities": [...], "total_count": 73 },
    ...
  },
  "entities": { ... },
  "devices": { ... },
  "areas": { ... },
  "integrations": [ ... ]
}
```

## Hypothesis Being Tested

**Hypothesis:** Dynamic discovery reveals capabilities not tracked by hardcoded extraction in the existing `ha-intelligence` script.

**Success metric:** Discovery finds ≥3 capabilities (with ≥10 entities each) that `ha-intelligence` doesn't currently track.

**If TRUE:** Proceed to Phase 1 (integrate discovery into ha-intelligence)
**If FALSE:** Stop - existing hardcoded approach is sufficient

## Next Steps (Post-MVP)

See `~/Documents/docs/plans/2026-02-11-ha-hub-lean-roadmap.md` for:
- Phase 1: Integration into ha-intelligence
- Phase 2: Full hub-and-spoke architecture (if needed)
- Pivot triggers and decision criteria

## Files

- `bin/discover.py` - Main discovery script (~300 lines)
- `tests/test_discover.py` - Unit tests
- `docs/` - API documentation and examples

## Requirements

- Python 3.12+
- Home Assistant instance accessible via network
- Environment variables: `HA_URL`, `HA_TOKEN`
