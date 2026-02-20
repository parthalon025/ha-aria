# ARIA Lean Audit & Restructure

**Date:** 2026-02-19
**Status:** Active (Phase 1 complete, Phase 2 planned)
**Roadmap:** `2026-02-19-lean-audit-roadmap.md`
**Author:** Justin McFarland + Claude

## Problem

ARIA has grown to 14 hub modules, 30+ engine files, and 30 open issues. The system works — 1668 tests pass, the pipeline flows — but:

1. **Intelligence quality is unvalidated.** Predictions and anomaly detection have never been compared against known-correct outputs.
2. **Mental model lost.** The module landscape outgrew the builder's understanding.
3. **Complexity exceeds purpose.** Two pattern modules, two discovery modules, a shadow engine, a transfer engine — for a system whose job is two things.

## Purpose

ARIA exists to produce two outputs:

1. **Automation recommendations** — pattern-based predictions of what to automate
2. **Anomaly detection** — deviations from learned baselines

Everything else — modules, pipeline, UI — exists to serve those two outputs. If it doesn't contribute, it gets archived.

## Approach: 5-Step Engineering Process

1. **Make requirements less dumb** — Two outputs. That's it.
2. **Delete (archive) the part** — Modules that don't feed recommendations or anomalies move to `_archived/modules/`
3. **Simplify/optimize** — Merge redundant modules, reduce coupling
4. **Accelerate** — Faster pipeline, faster feedback loops
5. **Automate** — Polish, UI, operational maturity

## Phases

### Phase 1: Module Triage (Delete-First Audit)

Walk each of the 14 modules with one question: **does this directly produce or feed automation recommendations or anomaly detection?**

| Triage Result | Action |
|---------------|--------|
| **Keep** | Core output module or essential plumbing |
| **Simplify/Merge** | Does something needed, but overcomplicated or redundant |
| **Archive** | Move to `_archived/modules/` with one-line rationale |

**Current module inventory (14):**

| Module | Area |
|--------|------|
| `activity_monitor` | Real-time HA state tracking |
| `activity_labeler` | Labels activity windows |
| `intelligence` | Reads batch engine output |
| `ml_engine` | Runs ML models in hub |
| `online_learner` | Incremental learning |
| `shadow_engine` | Shadow-tests changes safely |
| `pattern_recognition` | Detects recurring patterns |
| `patterns` | Also pattern detection (potential duplicate) |
| `orchestrator` | Coordinates cross-module workflows |
| `discovery` | Finds HA entities via subprocess |
| `organic_discovery/` | Discovers capabilities organically (6 sub-files) |
| `presence` | Tracks people via MQTT/Frigate |
| `data_quality` | Validates incoming data quality |
| `transfer_engine` | Transfers learning across domains |

**Audit per module produces:** purpose, inputs, outputs, feeds-recommendation?, feeds-anomaly?, triage decision, rationale.

**Deliverable:** Triage table with keep/simplify/archive decisions.

### Phase 2: Known-Answer Test Harness

For surviving modules, build deterministic integration tests:

- **Hand-crafted HA state fixtures** with known patterns and known anomalies embedded
- **Expected outputs calculated by hand** (not by ARIA) — the "answer key"
- **Per-module tests** — feed fixture, compare output to expected
- **Full pipeline test** — fixture through entire pipeline, verify final recommendations and anomalies match expected

Discrepancies between ARIA output and expected output = intelligence bugs.

**Deliverable:** Integration test suite with known-answer fixtures in `tests/integration/known_answer/`.

### Phase 3: Issue Triage & GitHub Roadmap

Re-evaluate all 30 open issues against the leaner architecture:

- Issues targeting archived modules → close with "archived" label
- Issues that dissolved due to simplification → close
- Surviving issues → re-prioritize with understanding from Phase 1
- New issues discovered during audit → file

Create GitHub milestones and project board to track remaining work.

**Deliverable:** GitHub project board with milestones.

### Phase 4: Fix & Optimize

Address surviving issues in priority order:

1. **Security** (auth, CORS, credential exposure)
2. **Reliability** (silent failures, unbounded collections)
3. **Performance** (blocking I/O, N+1 queries)
4. **Architecture** (coupling, hardcoded registries)

### Phase 5: UI — Science-Based Decision Tool

Redesign dashboard from status display to decision-support tool focused on the two outputs.

**Structure follows an analytical process:**

| Stage | User Sees | ARIA Provides |
|-------|-----------|---------------|
| **Observe** | Raw signals, entity states, activity | Data collection with context |
| **Orient** | Baselines, trends, correlations | Statistical framing, "normal" vs "now" |
| **Understand** | Flagged anomalies, identified patterns | ML output with explainability and confidence |
| **Decide** | Automation recommendations with evidence | Approve/reject with predicted impact |

**KPIs:**

- **Leading indicators** (predictive): pattern shifts, correlation changes, drift signals
- **Lagging indicators** (retrospective): recommendation acceptance rate, anomaly true positive rate, prediction accuracy, time-to-automation

Archive or strip dashboard pages that don't serve recommendations or anomaly detection.

**Deliverable:** Redesigned UI with OODA-based layout and KPI framework.

## Long-Term Vision

HA is the proving ground. Once ARIA reliably produces recommendations and detects anomalies against HA data, the architecture generalizes to other input datasets — any system that produces time-series state data with patterns worth detecting and actions worth recommending.

Phase 2's known-answer test harness is the key enabler: it proves the intelligence works independent of the data source.

## Success Criteria

- [x] Module count reduced (target: fewer than 10 active modules) — **Done: 14 → 10**
- [ ] Every surviving module has a known-answer integration test
- [ ] Full pipeline known-answer test passes
- [ ] Open issue count reduced by 50%+ (via archive + fix)
- [ ] UI surfaces recommendations and anomalies as primary views
- [ ] GitHub roadmap with milestones covers remaining work
