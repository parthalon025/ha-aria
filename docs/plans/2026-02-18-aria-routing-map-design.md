# ARIA System Routing Map & Token Optimization — Design

**Date:** 2026-02-18
**Status:** Approved
**Goal:** Reduce Claude Code token consumption when searching ARIA + create a full system interconnection reference that surfaces integration risks

## Problem

1. **Broad exploration:** Claude reads 10-20+ files to answer "how does X connect to Y" — the explorer agent just spent 80k tokens and 56 tool calls to map the project
2. **Repetitive re-reading:** Every session re-reads CLAUDE.md (152 lines) + architecture-detailed.md to build context
3. **Hidden seam risks:** Integration boundaries between modules, subprocess calls, and event bus subscriptions are not cataloged — bugs hide at these seams (Cluster B)

## Solution

### 1. `docs/system-routing-map.md` — Dual-Purpose Reference

Single document (~500 lines) with compact lookup tables at top (Claude-optimized) and detailed sections below (human-readable).

**Part A — Quick Lookup Tables:**
- Topic → File(s) index (~40 rows)
- HTTP Route Table (all 70+ API endpoints → handler → source:line)
- Event Bus Contract (all named events → emitter → subscribers, noting dual propagation via subscribe() + on_event())
- MQTT Topics (Frigate patterns → publisher → subscriber)
- Systemd Timer Map (timer → schedule → command → writes → reads)
- Cache Category Owners (category → writer module(s) → reader module(s))
- Subprocess Calls (caller → command → produces)

**Part B — Data Flow Diagrams (ASCII):**
5 runtime paths: batch pipeline, real-time stream, presence signals, organic discovery, closed-loop feedback

**Part C — Seam Risk Catalog:**
Every integration boundary with: what crosses it, what can go wrong, current test coverage (covered/partial/none), failure mode (loud crash vs silent corruption), suggested mitigation

### 2. CLAUDE.md Slimming

- Remove architecture paragraph duplicating architecture-detailed.md
- Remove full API endpoint list from Pipeline Verification (replaced by pointer to routing map)
- Add 5-line `## Quick Routing` section pointing to the routing map
- Net savings: ~50 lines from always-loaded context

### 3. No Separate Search Index File

The Topic → File(s) table in the routing map header serves this purpose. One doc to maintain, not three.

## Expected Token Savings

| Query Type | Before | After | Savings |
|------------|--------|-------|---------|
| "How does X connect to Y" | 10-20 file reads (~40-80k tokens) | 1-2 file reads (~4-8k tokens) | ~80% |
| Session startup (CLAUDE.md load) | 152 lines | ~100 lines | ~35% |
| "Where is the code for X" | 3-5 grep + read cycles (~15k tokens) | 1 read of routing map header (~2k tokens) | ~85% |

## Maintenance

The routing map must be updated when:
- New hub module added/removed
- New API endpoint added
- Event bus subscription changed
- New systemd timer added
- New subprocess call added

Pipeline Sankey topology (`pipelineGraph.js`) must stay in sync — same constraint already documented in gotchas.
