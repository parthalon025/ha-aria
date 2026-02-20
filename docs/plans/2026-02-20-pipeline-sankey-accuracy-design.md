# Pipeline Sankey Accuracy Overhaul — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix every inaccuracy in the pipeline Sankey visualization and make all nodes clickable for detail navigation.

**Architecture:** Update `pipelineGraph.js` (single source of truth for topology), fix metric extraction in `PipelineSankey.jsx`, add click handlers to `PipelineStepper.jsx` (mobile view). All changes flow from the shared graph definition.

**Branch:** `fix/pipeline-sankey-accuracy`

---

## 1. Topology Fixes

### Remove incorrect links

| Link | Why wrong |
|------|-----------|
| `shadow_engine` → `orchestrator` | Orchestrator only reads patterns cache, not shadow data |
| `shadow_engine` → `out_pipeline_stage` | Pipeline stage is API-managed, not written by shadow |
| `logbook_json` → `patterns` | Patterns reads engine disk output, not raw logbook |
| `snapshot_json` → `ml_engine` | ML engine reads engine output files, not raw snapshots |
| `ml_engine` → `discovery` (feedback) | Feedback writes to capabilities cache, not discovery module |
| `shadow_engine` → `discovery` (feedback) | Same — capabilities cache, not discovery |

### Add missing links

| Link | Type | Why |
|------|------|-----|
| `engine` → `ml_engine` | cache | Engine produces snapshot files ML trains on |
| `engine` → `patterns` | cache | Engine produces intraday files patterns reads |
| `discovery` → `presence` | cache | Presence reads entity→room mapping from discovery |
| `discovery` → `shadow_engine` | cache | Shadow reads included entity IDs from curation |
| `discovery` → `activity_monitor` | cache | Activity reads curation rules |
| `intelligence` → `orchestrator` | cache | Intelligence feeds automation suggestions |

### Fix feedback arrows

Both `ml_engine` and `shadow_engine` write accuracy feedback to the **capabilities** cache (managed by discovery). Change feedback targets from `discovery` to a conceptual self-loop or annotate as "capabilities feedback" targeting the discovery/capabilities data store.

---

## 2. Output Column Rework

Replace 14 stale output nodes with 7 OODA-aligned outputs:

| Output Node ID | Label | Feeds From | Page Route |
|---------------|-------|-----------|------------|
| `out_observe` | Observe | presence, activity_monitor, intelligence | `#/observe` |
| `out_understand` | Understand | intelligence, ml_engine, shadow_engine, patterns | `#/understand` |
| `out_decide` | Decide | orchestrator | `#/decide` |
| `out_capabilities` | Capabilities | discovery | `#/capabilities` |
| `out_ml_models` | ML Models | ml_engine | `#/ml-engine` |
| `out_curation` | Data Curation | discovery | `#/data-curation` |
| `out_validation` | Validation | (hub-level) | `#/validation` |

### Fix ACTION_CONDITIONS routes

| Old href | New href |
|----------|----------|
| `#/shadow` | `#/understand` |
| `#/automations` | `#/decide` |

---

## 3. Metrics & Freshness Fixes

### Freshness categoryMap corrections

| Old key | New key | Category |
|---------|---------|----------|
| `pattern_recognition` | `patterns` | `patterns` |
| `orchestrator` | `orchestrator` | `automation_suggestions` |
| Remove `data_quality` | — | Dead entry |
| Remove `organic_discovery` | — | Dead entry |
| Remove `activity_labeler` | — | Dead entry |

### Remove dead sparkline mappings

All three sparkline data paths reference non-existent cache arrays:
- `shadow_accuracy?.history` — doesn't exist
- `ml_pipeline?.training?.r2_history` — doesn't exist
- `pipeline?.event_rate_history` — doesn't exist

Remove all sparkline mappings from `getNodeSparklineData()`. The function stays but returns null for all nodes until real time-series data is added to the backend.

### NODE_DETAIL text corrections

- `logbook_json.writes`: "Engine + Trajectory Classifier" → "Engine + Patterns"
- `orchestrator.reads`: "patterns cache" → "patterns, automation_suggestions, pending_automations, created_automations"
- Review all other descriptions against real module code

---

## 4. Clickability

### Desktop Sankey

| Column | Click behavior |
|--------|---------------|
| Sources (col 0) | Show detail panel (hover already works) — no navigation (no dedicated page) |
| Intake (col 1) | Navigate to `#/detail/module/{id}` (already works) |
| Processing (col 2) | Navigate to `#/detail/module/{id}` (already works) |
| Enrichment (col 3) | Navigate to `#/detail/module/{id}` (already works) |
| Outputs (col 4) | Navigate to OODA page (`#/observe`, `#/understand`, etc.) |

Output nodes change from trace-back-only to page navigation. Trace-back can still be triggered via right-click or a separate trace button if needed, but primary click = navigate.

### Mobile PipelineStepper

- Each expanded node row becomes clickable → same navigation as desktop
- Output nodes navigate to their OODA page
- Source nodes are info-only (detail text shown inline on expand)
- Status LED colors match desktop (currently hardcoded green on mobile)

Both views consume `pipelineGraph.js` — topology and route fixes propagate automatically.

---

## Scope Boundaries

**In scope:** Topology accuracy, output rework, metric fixes, clickability, mobile sync.

**Out of scope:** Adding backend time-series endpoints for sparklines, redesigning the Sankey layout algorithm, adding new modules to the pipeline.
