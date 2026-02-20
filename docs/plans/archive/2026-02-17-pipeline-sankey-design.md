# ARIA Pipeline Sankey — Design Document

**Date:** 2026-02-17
**Status:** Approved
**Replaces:** `BusArchitecture` component in `Home.jsx` (~430 lines)

## Problem

The current Home page flow diagram is an engineer's wiring diagram — 12 module nodes in 3 swim lanes showing system architecture. It doesn't tell the story of what happens to home data from raw sensor event to suggested automation. It can't answer: "Why no suggestion?", "Is the data fresh?", "Why this suggestion?", or "What's broken?" — the four core troubleshooting questions.

## Audience

All three simultaneously via progressive disclosure:
- **Household:** Understands "lots of data → refined intelligence" at a glance
- **Portfolio:** Sees the scale (3,000 entities → 4 suggestions) and the engineering depth
- **Power user:** Can drill into any module's reads/writes/protocol/freshness to debug

## Architecture: Bus-Bar Sankey

### Core Concept

A left-to-right Sankey flow diagram with a **horizontal bus bar** representing the Hub Cache running across the center. This is architecturally honest — cache isn't a sequential processing step, it's a shared bus. Everything writes down into it; everything reads up from it.

```
    SOURCES         INTAKE            PROCESSING       ENRICHMENT        OUTPUTS
  ┌─────────┐    ┌──────────┐      ┌──────────┐     ┌──────────┐    ┌──────────┐
  │ 8 inputs │──▶│ 4 modules│──┐   │ 5 modules│──┐  │ 3 modules│──▶│15 outputs│
  └─────────┘    └──────────┘  │   └──────────┘  │  └──────────┘   └──────────┘
                                ▼                  ▼        ▲              ▲
  ══════════════════════════════════════════════════════════════════════════════
  ░░░░░░░░░░  HUB CACHE · hub.db · 15 categories  ░░░░░░░░░░░░░░░░░░░░░░░░░░
  ══════════════════════════════════════════════════════════════════════════════
                                ▲                  ▲
                          ◄── amber feedback arcs ──┘
```

### Why Bus-Bar, Not Linear Columns

Cache is a hub-and-spoke topology. A linear Sankey column implies data flows "through" cache sequentially. The bus bar shows the truth: all modules read/write to a shared spine. This matches how you'd explain ARIA: "All the modules talk through a shared cache."

## Node Map (45 Nodes, 6 Groups)

### Column 1: External Data Sources (8 nodes)

| Node | Protocol | What Flows |
|------|----------|------------|
| REST /api/states | HTTP GET (timers) | All 3,065 entity states |
| REST Registries | HTTP GET (startup) | Entity/device/area registries |
| WS state_changed | WebSocket sub | Real-time state change events |
| WS registry_updated | WebSocket sub | Registry change notifications |
| MQTT frigate/events | MQTT sub | Person/face detection from cameras |
| Logbook JSON | Disk (ha-log-sync) | ~/ha-logs/logbook/*.json |
| Snapshot JSON | Disk (engine timers) | ~/ha-logs/intelligence/daily/*.json |
| Ollama Queue | HTTP (port 7683) | LLM inference for naming + labels |

### Column 2: Intake Modules (4 nodes)

| Node | Reads | Writes to Cache |
|------|-------|-----------------|
| Discovery | REST Registries, WS registry_updated | entities, devices, areas, capabilities (seed), discovery_metadata |
| Activity Monitor | WS state_changed, REST /api/states (seed) | activity_log, activity_summary |
| Presence | MQTT frigate/events, WS state_changed, entities, devices | presence |
| Engine (Batch) | REST /api/states, Logbook JSON, Snapshot JSON | Engine JSON files on disk |

### Bus Bar: Hub Cache

Horizontal strip, 28px tall, `t-terminal-bg` texture, `--status-healthy` border.
Label: "HUB CACHE · hub.db · 15 categories"

15 cache categories: activity_log, activity_summary, areas, capabilities, devices, discovery_metadata, entities, intelligence, presence, predictions, pipeline_state, entity_curation, config, discovery_history, discovery_settings

### Column 3: Processing Modules (5 nodes)

| Node | Reads from Cache | Writes to Cache |
|------|-----------------|-----------------|
| Intelligence | activity_log, activity_summary, Engine JSON files | intelligence |
| ML Engine | capabilities, activity_log, Snapshot JSON | ml_predictions, ml_training_metadata, feature_config, capabilities (feedback) |
| Shadow Engine | activity_summary, activity_log, patterns, capabilities | predictions, pipeline_state, capabilities (feedback) |
| Pattern Recognition | Logbook JSON, intraday snapshots | patterns |
| Data Quality | entities, activity_log | entity_curation |

### Column 4: Enrichment Modules (3 nodes)

| Node | Reads from Cache | Writes to Cache |
|------|-----------------|-----------------|
| Orchestrator | patterns, automation_suggestions, pending_automations, created_automations | automation_suggestions, pending_automations, created_automations |
| Organic Discovery | entities, devices, capabilities, activity_summary, discovery_history, discovery_settings | capabilities (organic), discovery_history, discovery_settings |
| Activity Labeler | activity_summary, intelligence, presence, activity_labels | activity_labels |

### Column 5: API Outputs (15 nodes)

| Node | Source | Dashboard Page |
|------|--------|---------------|
| Automation Suggestions | Orchestrator | Automations |
| Pending Automations | Orchestrator | Automations |
| Created Automations | Orchestrator | Automations |
| ML Predictions | ML Engine | Predictions |
| ML Drift / Anomalies | ML Engine | ML Engine |
| Shadow Predictions | Shadow Engine | Shadow |
| Shadow Accuracy | Shadow Engine | Shadow |
| Pipeline Stage | Shadow Engine | Home |
| Activity Labels | Activity Labeler | Home, Intelligence |
| Patterns | Pattern Recognition | Patterns |
| Intelligence Summary | Intelligence | Intelligence |
| Presence Map | Presence | Home |
| Entity Curation | Data Quality | Data Curation |
| Capability Registry | merged sources | Capabilities, Discovery |
| Validation Results | Validation runner | Validation |

### Feedback Loops (8 reverse flows)

| From | To | Data |
|------|----|------|
| ML Engine | capabilities (cache) | ml_accuracy per capability |
| Shadow Engine | capabilities (cache) | hit_rate per capability |
| Orchestrator | HA (future) | /api/automation/trigger |
| User | Activity Labeler | POST /api/activity/label |
| User | Automations | POST /api/automations/feedback |
| User | Curation | PUT /api/curation/{entity_id} |
| User | Pipeline | POST /api/pipeline/advance |
| User | Capabilities | PUT /api/capabilities/{name}/promote |

## Progressive Disclosure (3 Layers)

### Layer 1: Glance (<5 seconds, 0 interactions)

5 collapsed column groups + bus bar + action strip. Each column shows:
- Name (bold, --text-primary, 11px mono)
- Aggregate health LED (worst-of-children)
- Headline metric
- Aggregate flow ribbons routed through the bus bar

Flow widths proportional to data volume:
- Sources → Intake: 40px (~3,065 entities + event stream)
- Intake → Bus Bar: 35px
- Bus Bar → Processing: 28px
- Processing → Bus Bar: 20px
- Bus Bar → Enrichment: 15px
- Enrichment → Outputs: 8px

The visual narrowing from 40px → 8px IS the story.

**Cognitive load:** 6 visual chunks (5 columns + bar). Within Miller's 7±2.

### Layer 2: Expand (<30 seconds, click/tap column)

Click any column → expands to reveal individual nodes. Each node shows:
- LED (health status)
- Name
- Live metric (from existing getNodeMetric logic)
- 48×16px sparkline (last 7 days, TimeChart compact mode)
- Freshness timestamp (green <15m, amber <1h, red >1h)

**Only one column expanded at a time.** Click another column to switch, click header to collapse.

**Transition:** Fade-out old flows (150ms) → expand column → fade-in new flows (150ms). No morph — avoids Bezier recalculation jank.

### Layer 3: Hover/Long-Press Detail (<60 seconds)

Hover expanded node (long-press on mobile) → detail panel slides up:

```
┌─────────────────────────────────────────────────────┐
│ ML ENGINE                                  2m ago ● │
│ ▶ Reads: capabilities, activity_log, daily/*.json   │
│ ▼ Writes: ml_predictions, ml_training_metadata,     │
│          feature_config, capabilities (ml_accuracy)  │
│ ⚡ Protocol: Trains from daily snapshot files        │
│ R²: 0.71 · Models: 3 · Last train: 4h ago          │
└─────────────────────────────────────────────────────┘
```

### Layer 3b: Trace-Back (click output node)

Click any Output node → highlights the **shortest critical path** back through the diagram:
1. Dim all non-ancestor nodes/flows to opacity 0.12
2. Highlight critical path in --accent with stroke-width +2
3. Show data values at each hop along the trace
4. Click outside or same node to dismiss

Example trace for "Automation Suggestions":
```
REST /api/states → Engine → [Bus Bar] → Patterns → [Bus Bar] → Orchestrator → Suggestions
 "3,065 states"   "47 days"            "12 sequences"          "3 conf>.7"    "3 suggestions"
```

Critical path algorithm: precompute dependency chain from NODE_DETAIL reads/writes graph. Only highlight nodes whose writes are in the transitive closure of the output's reads.

## Visual Encoding

### Flow Colors (3-color limit, Treisman 1980)

| Flow Type | Color | CSS Token | Pattern |
|-----------|-------|-----------|---------|
| Data flow (forward) | Cyan | --accent | Solid |
| Cache write/read | Green | --status-healthy | Solid |
| Feedback (backward) | Amber | --status-warning | Dashed 4 3 |

Flow opacity encodes volume: 0.3 (thin) → 0.7 (thick).

### Node Health LEDs (existing system)

| Status | Color | Animation | Meaning |
|--------|-------|-----------|---------|
| Healthy | --status-healthy | Slow pulse 3s | Running, data fresh |
| Warning | --status-warning | Medium pulse 1.5s | Stale or degraded |
| Error | --status-error | Static | Failed or disconnected |
| Waiting | --status-waiting | None | Not started |

### Freshness Timestamps

| Age | Color | Meaning |
|-----|-------|---------|
| <15 min | --status-healthy | Fresh |
| 15m–1h | --status-warning | Getting stale |
| >1h | --status-error | Stale, investigate |

### Sankey Link Rendering

Cubic Bezier ribbons, no library:
```
M x0,(y0 - halfWidth)
C cx0,(y0 - halfWidth)  cx1,(y1 - halfWidth)  x1,(y1 - halfWidth)
L x1,(y1 + halfWidth)
C cx1,(y1 + halfWidth)  cx0,(y0 + halfWidth)  x0,(y0 + halfWidth)
Z
```
Control points at 40% and 60% of horizontal distance for smooth S-curves.
Terminal scan-line texture via SVG `<pattern>` fill on flows.

## Animation

### Data Update Pulse (Tier 2 — triggered, runs once)

When WebSocket pushes a cache update:
- Affected node border flashes --accent for 500ms (t2-tick-flash)
- The flow ribbon to that node gets +2px width bump for 300ms
- No persistent animation — fires once per update

### No Tracer Particles

Removed. Flow width + opacity + freshness timestamps communicate liveness without constant motion. Dead flows rendered as gray dashed — zero pulses fire on them, making them visually "holes" (Gestalt figure-ground).

### Reduced Motion (prefers-reduced-motion)

- Data pulse: static accent border (no flash)
- Expand/collapse: instant (no transition)
- LED pulses: static color, no animation
- Feedback arcs: static dashed lines

## Action Strip

Context-aware `t-frame` card below the Sankey showing highest-priority user action.

Priority order:
1. Pipeline ready to advance → "Pipeline gate met — advance to next stage →" → /shadow
2. Edge-case entities pending → "Review {n} edge-case entities →" → /curation
3. Shadow disagreements > 5 → "Review {n} high-confidence disagreements →" → /shadow
4. Activity labels need correction → "Correct {n} low-confidence labels →" → /intelligence
5. Organic candidates pending → "Review {n} discovered capabilities →" → /discovery
6. Automation suggestions available → "Review {n} automation suggestions →" → /automations
7. Everything healthy → "System healthy — no action needed"

Shows top 1-2 actions. Each is a link navigating to the relevant page.

## Mobile (<640px): Pipeline Stepper

No fake Sankey. Vertical stepper instead:

```
┌─ PIPELINE ──────────────────────┐
│  ● Sources        8 inputs  2m  │
│  │                               │
│  ● Intake         3,065 ent 2m  │
│  │                               │
│  ░░░ HUB CACHE ░░░░░░░░░░░░░░  │
│  │                               │
│  ● Processing     R²: .71  4h  │
│  │                               │
│  ● Enrichment     4 sugg   2m  │
│  │                               │
│  ● Outputs        13 pages      │
│                                  │
│  ┌─ Action ────────────────┐    │
│  │ Review 3 edge entities →│    │
│  └─────────────────────────┘    │
└──────────────────────────────────┘
```

Tap any stage → expands to show individual nodes with LED + metric + freshness. Same data, honest layout.

## Data Fetching

| Data | API Source | Already Fetched? |
|------|-----------|-----------------|
| Module health | GET /health → modules | Yes |
| Entity counts | useCache('entities') | Yes |
| Pipeline stage | GET /api/pipeline | Yes |
| Shadow accuracy | GET /api/shadow/accuracy | Yes |
| Activity data | useCache('activity_summary') | Yes |
| Curation counts | GET /api/curation/summary | Yes |
| Intelligence | useCache('intelligence') | Yes |
| ML pipeline | GET /api/ml/pipeline | **New fetch** |
| Sparkline history | GET /api/cache/{cat} with history | **New endpoint** |

One new fetch (ml/pipeline) and one new endpoint (cache history for sparklines).

## Implementation Estimate

| Component | ~Lines | Notes |
|-----------|--------|-------|
| Sankey layout engine | 150 | Node positioning, flow width calc, bus bar routing |
| SVG rendering (Preact) | 250 | Nodes, flows, bus bar, labels, LEDs |
| Expand/collapse | 80 | State management, fade transition |
| Trace-back | 60 | Dependency graph, highlight/dim |
| Action strip | 50 | Condition evaluation, links |
| Mobile stepper | 100 | Separate component, tap-to-expand |
| Sparkline integration | 30 | Reuse TimeChart compact |
| **Total** | **~720** | Replaces ~430 line BusArchitecture |

## Science-Backed Principles Applied

| # | Principle | Application |
|---|-----------|-------------|
| 1 | Data-ink ratio (Tufte 1983) | No tracer particles. Flow width = real data. Every pixel earns its place. |
| 2 | Perceptual hierarchy (Cleveland & McGill 1984) | Position = pipeline stage, Width = volume, Color = flow type. Most important → most accurate channel. |
| 3 | Preattentive processing (Treisman 1980) | 3 colors max. Dead flows (gray dashed) create visual "holes" that draw attention via figure-ground. |
| 4 | Gestalt enclosure (Wertheimer 1923) | Bus bar enclosure groups all cache operations. Column groups enclose related modules. |
| 5 | Cognitive load (Miller 1956, Sweller 1988) | Default view = 6 chunks. Expanded = 6-10 chunks. Never >15. |
| 6 | Progressive disclosure (Shneiderman 1996) | 3 layers: glance (overview), expand (filter), hover/trace (detail on demand). |
| 7 | Sparklines (Tufte 2006) | Every expanded node metric has a 7-day trend. Number without trend is half a story. |

## Troubleshooting Matrix

| Question | How the Sankey Answers It |
|----------|--------------------------|
| "Why no suggestion?" | Trace-back from Suggestions node. See where flow narrows to zero. |
| "Is the data fresh?" | Freshness timestamps on expanded nodes. Dead flows (no pulses) = inactive pipeline leg. |
| "Why this suggestion?" | Trace-back shows critical path + data values at each hop. |
| "What's broken?" | Red LED on node. Gray dashed dead flow. Expand column to see which module failed. |
