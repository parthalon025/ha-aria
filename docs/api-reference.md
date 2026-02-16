# ARIA API Reference

## In Plain English

This is the menu of questions you can ask ARIA and the exact way to ask them. Think of it like a phone tree for a customer service line -- each URL is a different extension that gives you a specific piece of information about what your home is doing.

## Why This Exists

ARIA collects a lot of data about your home -- predictions, anomalies, entity states, ML model health, and more. Without a clear reference for how to retrieve that data programmatically, anyone building on top of ARIA (including the dashboard itself) would have to guess at URLs and response formats. This document eliminates that guesswork by listing every available endpoint with working examples you can copy and paste.

Base URL: `http://127.0.0.1:8001`

## Cache & Core

```bash
# Full cache
curl -s http://127.0.0.1:8001/api/cache | python3 -m json.tool

# Specific cache category
curl -s http://127.0.0.1:8001/api/cache/activity_summary | python3 -m json.tool

# Health check
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```

## Presence

```bash
# Per-room presence/occupancy estimates
curl -s http://127.0.0.1:8001/api/cache/presence | python3 -m json.tool

# Response shape:
# {
#   "category": "presence",
#   "data": {
#     "timestamp": "2026-02-16T10:30:00",
#     "rooms": {
#       "room_name": {
#         "probability": 0.85,
#         "confidence": "high",
#         "signals": ["motion_sensor", "light_on"]
#       }
#     },
#     "occupied_rooms": ["living_room", "kitchen"],
#     "identified_persons": {"person.alice": "bedroom"},
#     "mqtt_connected": true,
#     "camera_rooms": {"living_room": {"person": 2, "face": 1}}
#   }
# }
```

## Shadow Mode

```bash
# Predictions with outcomes
curl -s http://127.0.0.1:8001/api/shadow/predictions?limit=10 | python3 -m json.tool

# Accuracy metrics
curl -s http://127.0.0.1:8001/api/shadow/accuracy | python3 -m json.tool

# High-confidence disagreements (most informative wrong predictions)
curl -s http://127.0.0.1:8001/api/shadow/disagreements | python3 -m json.tool

# Pipeline stage progression
curl -s http://127.0.0.1:8001/api/pipeline | python3 -m json.tool

# Pipeline control (advance/retreat with gate validation)
curl -s -X POST http://127.0.0.1:8001/api/pipeline/advance -H 'Content-Type: application/json' -d '{}'
curl -s -X POST http://127.0.0.1:8001/api/pipeline/retreat -H 'Content-Type: application/json' -d '{}'

# Shadow propagation
curl -s http://127.0.0.1:8001/api/shadow/propagation | python3 -m json.tool
```

## ML Features

```bash
curl -s http://127.0.0.1:8001/api/ml/drift | python3 -m json.tool
curl -s http://127.0.0.1:8001/api/ml/features | python3 -m json.tool
curl -s http://127.0.0.1:8001/api/ml/models | python3 -m json.tool
curl -s http://127.0.0.1:8001/api/ml/anomalies | python3 -m json.tool
curl -s http://127.0.0.1:8001/api/ml/shap | python3 -m json.tool

# ML pipeline overview (aggregated pipeline state)
curl -s http://127.0.0.1:8001/api/ml/pipeline | python3 -m json.tool

# Response shape:
# {
#   "snapshot": {"last_run": "...", "entity_count": 3065, "status": "ok"},
#   "training": {"last_trained": "...", "stale": false, "models": [...]},
#   "features": {"source": "vector_builder", "count": 14},
#   "validation": {"min_entities": 100, "max_unavailable_ratio": 0.5},
#   "drift": {"detected": false, "last_check": "..."}
# }
```

## Organic Discovery

```bash
# Candidate capabilities (discovered but not promoted)
curl -s http://127.0.0.1:8001/api/capabilities/candidates | python3 -m json.tool

# Discovery run history
curl -s http://127.0.0.1:8001/api/capabilities/history | python3 -m json.tool

# Promote/archive a capability
curl -s -X PUT http://127.0.0.1:8001/api/capabilities/kitchen_lights/promote
curl -s -X PUT http://127.0.0.1:8001/api/capabilities/kitchen_lights/archive

# Discovery settings (autonomy mode, naming backend, thresholds)
curl -s http://127.0.0.1:8001/api/settings/discovery | python3 -m json.tool
curl -s -X PUT http://127.0.0.1:8001/api/settings/discovery -H 'Content-Type: application/json' \
  -d '{"autonomy_mode": "auto_promote"}'

# Trigger on-demand discovery run
curl -s -X POST http://127.0.0.1:8001/api/discovery/run

# Discovery module status
curl -s http://127.0.0.1:8001/api/discovery/status | python3 -m json.tool
```

## Capability Registry

```bash
# All registered capabilities (with layer/status summaries)
curl -s http://127.0.0.1:8001/api/capabilities/registry | python3 -m json.tool

# Filter by layer or status
curl -s 'http://127.0.0.1:8001/api/capabilities/registry?layer=hub' | python3 -m json.tool
curl -s 'http://127.0.0.1:8001/api/capabilities/registry?status=stable' | python3 -m json.tool

# Single capability detail
curl -s http://127.0.0.1:8001/api/capabilities/registry/shadow_predictions | python3 -m json.tool

# Dependency graph (nodes + edges)
curl -s http://127.0.0.1:8001/api/capabilities/registry/graph | python3 -m json.tool

# Runtime health per capability (maps hub module status)
curl -s http://127.0.0.1:8001/api/capabilities/registry/health | python3 -m json.tool
```

## Config & Curation

```bash
# All config parameters with metadata
curl -s http://127.0.0.1:8001/api/config | python3 -m json.tool

# Entity curation summary (tier/status counts)
curl -s http://127.0.0.1:8001/api/curation/summary | python3 -m json.tool
```
