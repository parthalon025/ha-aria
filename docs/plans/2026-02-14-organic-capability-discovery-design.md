# Organic Capability Discovery — Design Document

**Date:** 2026-02-14
**Status:** Approved
**Scope:** Phase 1 (domain clustering) + Phase 2 (behavioral clustering)

## In Plain English

This plan teaches ARIA to discover what your smart home can do by looking at the data, instead of relying on a pre-written list. It is like the difference between a restaurant with a fixed menu and a chef who walks through the market, sees what is fresh, and creates dishes from what is actually available.

## Why This Exists

ARIA originally detected capabilities through 10 hand-coded rules -- "if it is a light entity, call it lighting." Adding a new capability meant editing code and redeploying. Worse, the system could never notice interesting patterns that nobody anticipated, like a group of devices that always activate together in the evening. This design replaces the fixed menu with organic clustering that examines what entities are and how they behave together, scores each grouping on usefulness, and surfaces the best ones as new capabilities -- all without requiring code changes.

## Problem

ARIA's capability detection is hard-coded: 10 detection functions in `bin/discover.py` that match entities by domain, device_class, and attributes. Adding a new capability requires editing 3-4 files and redeploying. The system can't notice patterns in the data that weren't anticipated.

## Solution

Replace static detection with organic discovery — ARIA clusters entities from raw HA data, scores each cluster's usefulness, and surfaces them as capabilities. The existing 10 capabilities become seed examples for validation, not a ceiling.

## Architecture

### Two Discovery Layers

**Layer 1 — Domain Capabilities (what entities ARE):**
- Build a feature vector per entity from attributes (domain, device_class, unit, area, manufacturer, state cardinality, activity rate, availability)
- Cluster using HDBSCAN (no pre-specified cluster count)
- Validate against seed capabilities (Jaccard similarity > 80%)
- Runs weekly

**Layer 2 — Behavioral Capabilities (what entities DO together):**
- Build co-occurrence matrix from 14-day logbook (15-min time windows)
- Weight edges by conditional probability P(A|B)
- Cluster co-occurrence matrix with HDBSCAN
- Extract temporal signatures (peak hours, weekday bias)
- Runs weekly, after Layer 1

### Usefulness Score (0-100%)

Composite metric per capability, updated each discovery run:

| Component | Weight | Source |
|-----------|--------|--------|
| Predictability | 30% | ML model accuracy (R²/F1) for entities in cluster |
| Stability | 25% | % of runs where cluster appeared in last 14 days |
| Entity coverage | 15% | Entities in cluster / total entities, scaled |
| Activity | 15% | Average daily state changes per entity |
| Cohesion | 15% | Silhouette score of cluster |

### Autonomy Modes (user-selectable)

1. **Suggest & wait** — candidates shown in UI, user promotes manually
2. **Auto-promote with guardrails** — promote at configurable threshold (default ≥50% for 7 consecutive days), archive at ≤10% for 14 days
3. **Fully autonomous** — promote at ≥30%, archive at ≤10% for 14 days

### Naming Pipeline (user-selectable with fallback)

1. **Heuristic** (default) — most common domain + area + temporal pattern
2. **Ollama** — deepseek-r1:8b generates natural language name + description
3. **External LLM** — OpenAI/Claude API for highest quality naming

Heuristic always runs as fallback regardless of selection.

## Data Pipeline

```
WEEKLY TIMER (aria-organic-discovery.timer)
│
├── LAYER 1: DOMAIN CLUSTERING
│   ├── Fetch entity states + registries from HA
│   ├── Build feature vectors (3,065 × N features)
│   ├── HDBSCAN clustering
│   └── Validate against seed capabilities (Jaccard)
│
├── LAYER 2: BEHAVIORAL CLUSTERING
│   ├── Load 14-day logbook from ~/ha-logs/
│   ├── Build co-occurrence matrix (15-min windows)
│   ├── HDBSCAN clustering on co-occurrence
│   └── Extract temporal patterns (peak hours, weekday bias)
│
├── MERGE & SCORE
│   ├── Compute usefulness % (5 components)
│   ├── Name clusters via selected backend
│   ├── Compare to previous run (stability tracking)
│   └── Deduplicate across layers
│
├── AUTONOMY ENGINE
│   ├── Apply promotion/archival rules per mode
│   └── Log decisions to discovery_history
│
└── WRITE TO CACHE
    ├── hub.db: capabilities (same key, extended schema)
    └── hub.db: discovery_history (new, tracks runs over time)
```

## Cache Schema

Extended capability record (backwards compatible):

```json
{
  "power_monitoring": {
    "available": true,
    "entities": ["sensor.pdu_outlet_1_power"],
    "total_count": 20,
    "can_predict": true,

    "source": "seed | organic",
    "usefulness": 87,
    "usefulness_components": {
      "predictability": 92,
      "stability": 100,
      "entity_coverage": 65,
      "activity": 78,
      "cohesion": 88
    },
    "layer": "domain | behavioral",
    "status": "candidate | promoted | archived",
    "first_seen": "2026-02-14",
    "promoted_at": "2026-02-14",
    "naming_method": "seed | heuristic | ollama | external_llm",
    "description": "Human-readable description",
    "stability_streak": 14,
    "temporal_pattern": {
      "peak_hours": [18, 19, 20, 21],
      "weekday_bias": 0.3
    }
  }
}
```

New fields only — existing consumers unaffected.

## API

```
GET  /api/capabilities                    — all capabilities (includes organic)
GET  /api/capabilities/candidates         — candidate capabilities only
GET  /api/capabilities/history            — discovery run history

PUT  /api/capabilities/{name}/promote     — manually promote candidate
PUT  /api/capabilities/{name}/archive     — manually archive capability
PUT  /api/capabilities/{name}/can-predict — (existing) toggle prediction

GET  /api/settings/discovery              — autonomy mode, naming backend, thresholds
PUT  /api/settings/discovery              — update discovery settings

POST /api/discovery/run                   — trigger on-demand discovery run
GET  /api/discovery/status                — last/next run, currently running?
```

## Dashboard UI

### Capabilities Page (extended)

- **Promoted section:** Existing capability cards with usefulness bar, source badge (seed/organic), layer badge (domain/behavioral)
- **Candidates section:** Discovered-but-not-promoted capabilities with [Promote] and [Archive] buttons, stability streak counter
- **Archived section:** Collapsed by default, expandable
- **Each card shows:** Name, usefulness %, entity count, source, layer, streak

### Capability Detail View

- Usefulness breakdown (5 horizontal bars with labels and percentages)
- Entity list with co-occurrence strength (for behavioral)
- Temporal pattern visualization (for behavioral — peak hours, weekday bias)
- History sparkline (usefulness over past runs)
- Action buttons: [Enable Predictions] [Promote/Archive]

### Discovery Settings Panel

- Autonomy mode radio buttons with inline descriptions
- Naming backend radio buttons with pro/con text
- Threshold sliders (promote %, archive %, streak days)
- Schedule display with [Run Now] button
- Last run summary (clusters found, promoted, archived)

## Scheduling

- **Without LLM naming:** Pure sklearn, ~5 min. Can run anytime.
- **With Ollama naming:** +45 min for deepseek-r1:8b. Sunday 4:00 AM (gap between 3:45 AM aria tasks and 7:00 AM morning brief).
- **Timer:** `aria-organic-discovery.timer` — weekly, Sunday 4:00 AM default.

## Seed Validation

The 10 existing hard-coded capabilities become labeled ground truth:

1. After each Layer 1 clustering run, compute Jaccard similarity between each seed capability's entity set and the closest discovered cluster
2. If any seed has < 80% overlap with its best-match cluster, log a warning
3. Seed capabilities always appear in the UI regardless of clustering results (source: "seed")
4. Over time, as organic clusters mature and match seeds at > 95%, the seed label can be retired (manual action)

## Dependencies

- **hdbscan** — Python package (pip install hdbscan), or use sklearn.cluster.HDBSCAN (sklearn ≥ 1.3)
- **No new system dependencies** — all data sources already exist (entity states, registries, logbook, co-occurrences)
- **No new Ollama models** — uses existing deepseek-r1:8b if LLM naming enabled

## Success Criteria

Within 2 weeks of running:
- Organic discovery reproduces all 10 seed capabilities at > 80% Jaccard overlap
- Surfaces at least 3 new capability groupings scoring > 50% usefulness
- Dashboard displays promoted and candidate capabilities with usefulness scores
- User can toggle autonomy mode and naming backend from settings UI

## Pivot Trigger

If after 2 weeks every discovered cluster maps 1:1 to existing HA domains with nothing novel, the feature vector needs richer signals (temporal features, device graph edges) or the approach needs rethinking.
