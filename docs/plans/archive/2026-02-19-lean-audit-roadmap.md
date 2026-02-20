# ARIA Lean Audit — Strategy Roadmap

**Date:** 2026-02-19
**Status:** Active
**Parent design:** `2026-02-19-lean-audit-restructure-design.md`

## Purpose

ARIA exists to produce two outputs:
1. **Automation recommendations** — pattern-based predictions of what to automate
2. **Anomaly detection** — deviations from learned baselines

This roadmap tracks the 5-phase engineering process to achieve those outputs reliably with minimal complexity.

---

## Phase Summary

| Phase | Name | Status | Deliverable |
|-------|------|--------|-------------|
| **1** | Module Triage | **Done** | 14 → 10 modules, 4 archived, 1 merged, 1 renamed |
| **2** | Known-Answer Test Harness | **Done** | 37 known-answer tests + golden snapshots |
| **3+4** | Issue Triage & Fix | **Done** | 37 issues fixed, 18 remain open |
| **5** | UI — Decision Tool | Queued | OODA-based dashboard redesign |

---

## Phase 1: Module Triage (Done)

**Completed:** 2026-02-19
**Report:** `2026-02-19-module-triage-report.md`
**Branch:** `feature/lean-audit-phase1` (merged to main)

### Results

| Metric | Before | After |
|--------|--------|-------|
| Hub modules | 14 | 10 |
| Tests | 1,668 | 1,416 |
| Pipeline Sankey nodes | 31 | 25 |

### Decisions Made

- **Archived 4 modules:** online_learner (marginal value), organic_discovery (not feeding outputs), transfer_engine (future feature), activity_labeler (labels don't feed outputs)
- **Merged 1:** data_quality → discovery (entity classification now runs inside discovery)
- **Renamed 1:** pattern_recognition → trajectory_classifier (clarity, avoids collision with patterns.py)
- **Kept 8 as-is:** discovery, patterns, orchestrator, shadow_engine, ml_engine, intelligence, activity_monitor, presence
- **Architecture docs updated:** system-routing-map.md, architecture-detailed.md, pipelineGraph.js

### Surviving Module Inventory (10)

| Module | Role | Feeds |
|--------|------|-------|
| discovery | Entity/device/area scanning + classification | Foundation data |
| activity_monitor | Real-time state_changed listener | Feeds patterns + shadow |
| patterns | Recurring sequence detection (clustering + association rules) | Recommendations |
| orchestrator | Automation YAML suggestions from patterns | **Primary recommendation output** |
| shadow_engine | Predict-compare-score loop | **Primary anomaly output** |
| trajectory_classifier | Trajectory classification + anomaly explanation (Tier 3+) | Anomaly explanation |
| ml_engine | Model training, ensemble weights | Both |
| intelligence | Engine output → hub cache bridge | Both (bridge) |
| presence | MQTT/Frigate person tracking | Context signal |
| audit_logger | Tamper-evident audit trail | Cross-cutting |

---

## Phase 2: Known-Answer Test Harness (Done)

**Completed:** 2026-02-19
**Plan:** `2026-02-19-known-answer-test-harness.md`
**Design:** `2026-02-19-known-answer-test-harness-design.md`
**Branch:** `feature/known-answer-tests` (merged to main)

### Results

| Metric | Target | Actual |
|--------|--------|--------|
| Module KA tests | 10 | 10 |
| Full pipeline test | 1 | 1 |
| Golden snapshots | 11 | 11 |
| Total KA tests | ~11 | 37 |

### Success Criteria

- [x] All 10 modules have known-answer tests passing
- [x] Full pipeline test passes end-to-end
- [x] Golden snapshots committed for all modules
- [x] Dashboard shows greyed-out tier-gated modules
- [x] CLAUDE.md reflects Phase 1 changes
- [x] No regressions in existing 1,416 tests

---

## Phase 3+4: Issue Triage & Fix (Done)

**Completed:** 2026-02-19
**Plan:** `2026-02-19-issue-triage-fix-plan.md`
**Branch:** `feature/phase3-4-fixes`

### Triage Results

- 8 issues closed (already-fixed or archived module)
- 10 issues annotated with Phase 1 context
- 32 issues re-labeled with priority + category
- Milestones created: Phase 4, Phase 5
- Open issues reduced from 63 → 55

### Fix Results

| Category | Issues Fixed | Key Changes |
|----------|-------------|-------------|
| Security | 4 (#43, #44, #64, #65) | Config redaction, CORS, auth gate, input validation |
| Reliability Critical | 3 (#19, #20, #21) | Schema contracts, reconnect stagger, Telegram health |
| Reliability High | 5 (#22-25, #27) | Race conditions, cold-start, config propagation |
| Reliability Medium | 4 (#45-47, #56) | Silent failures, typed errors, retry logic, bounds |
| Performance | 6 (#51-55, #58) | Async I/O, batch queries, parallel init, sessions |
| Architecture | 6 (#29, #30, #42, #59, #60, #62) | Config wiring, auto-discovery, shared constants |
| Remaining | 9 (#9, #36-38, #41, #48-50, #57) | Timezone, pruning, data quality, watchdog, model status |

### Metrics

| Metric | Before | After |
|--------|--------|-------|
| Open issues | 63 | 18 |
| Test count | 1,416 | 1,543 |
| Issues fixed | — | 37 |
| Reduction | — | 71% |

### Follow-up Issues Filed
- #74: cache.py naive timestamps
- #75: snapshot pruning string comparison
- #76: presence C901 complexity

---

## Phase 5: UI — Science-Based Decision Tool (Done)

**Prerequisite:** Phase 4 complete (reliable, secure backend)

### OODA Framework

| Stage | User Sees | ARIA Provides |
|-------|-----------|---------------|
| **Observe** | Raw signals, entity states, activity | Data collection with context |
| **Orient** | Baselines, trends, correlations | Statistical framing: "normal" vs "now" |
| **Understand** | Flagged anomalies, identified patterns | ML output with explainability and confidence |
| **Decide** | Automation recommendations with evidence | Approve/reject with predicted impact |

### KPIs

- **Leading:** pattern shifts, correlation changes, drift signals
- **Lagging:** recommendation acceptance rate, anomaly true positive rate, prediction accuracy

### Approach

- Archive dashboard pages that don't serve recommendations or anomaly detection
- Design follows `docs/design-language.md` principles (Tufte, Cleveland & McGill, Gestalt)
- Progressive disclosure: summary → detail → raw data

---

## Long-Term Vision

HA is the proving ground. Once ARIA reliably produces recommendations and detects anomalies against HA data, the architecture generalizes to **any system that produces time-series state data** with patterns worth detecting and actions worth recommending.

Phase 2's known-answer test harness is the key enabler: it proves the intelligence works independent of the data source.

---

## Success Criteria (Overall)

- [x] Module count reduced to < 10 active modules *(Phase 1: 14 → 10)*
- [x] Every surviving module has a known-answer integration test *(Phase 2: 10/10 modules)*
- [x] Full pipeline known-answer test passes *(Phase 2: engine → hub → output)*
- [x] Open issue count reduced by 50%+ *(Phase 3+4: 63 → 18, 71% reduction)*
- [x] UI surfaces recommendations and anomalies as primary views *(Phase 5: OODA nav + SUPERHOT)*
- [x] GitHub roadmap with milestones covers remaining work *(Phase 3: milestones created)*
