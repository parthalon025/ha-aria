# ARIA Lean Audit — Phase 1 Module Triage Report

**Date:** 2026-02-19
**Branch:** `feature/lean-audit-phase1`
**Design doc:** `docs/plans/2026-02-19-lean-audit-restructure-design.md`

## Triage Criteria

Every module was evaluated against one question: **Does this module directly produce or feed the two desired outputs?**

1. **Automation recommendations** — patterns → orchestrator → automation YAML
2. **Anomaly detection** — shadow predictions, IsolationForest, drift alerts

Modules that don't feed either output get archived. Modules that overlap get merged or renamed.

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Hub modules | 14 | 10 |
| Test count | 1,668 | 1,416 |
| Tests removed | — | 252 |
| Pipeline Sankey nodes | 31 | 25 |
| Pipeline Sankey links | 37 | 28 |

## Final Module Inventory (10 modules)

| # | Module ID | File | Purpose | Feeds |
|---|-----------|------|---------|-------|
| 1 | `discovery` | `aria/modules/discovery.py` | Entity/device/area scanning + entity classification (merged from data_quality) | Both (foundation data) |
| 2 | `ml_engine` | `aria/modules/ml_engine.py` | Model training, periodic retraining, ensemble weights | Both |
| 3 | `patterns` | `aria/modules/patterns.py` | Recurring event sequence detection from logbook | Recommendations |
| 4 | `orchestrator` | `aria/modules/orchestrator.py` | Automation suggestions from detected patterns | Recommendations (primary output) |
| 5 | `shadow_engine` | `aria/modules/shadow_engine.py` | Predict-compare-score loop on state changes | Anomalies (primary output) |
| 6 | `trajectory_classifier` | `aria/modules/trajectory_classifier.py` | Trajectory classification + anomaly explanation (Tier 3+) | Anomalies |
| 7 | `intelligence` | `aria/modules/intelligence.py` | Engine output → hub cache bridge, Telegram digest | Both (bridge) |
| 8 | `activity_monitor` | `aria/modules/activity_monitor.py` | Real-time state_changed listener, 15-min windows | Both (feeds patterns + shadow) |
| 9 | `presence` | `aria/modules/presence.py` | MQTT/Frigate person tracking, Bayesian occupancy | Both (context signal) |
| 10 | `audit_logger` | `aria/hub/audit.py` | Tamper-evident audit trail | Cross-cutting |

## Archived Modules (4)

| Module | Reason | Impact |
|--------|--------|--------|
| `online_learner` | Tier 3+ only, marginal value over shadow_engine scoring. River models added ~6% accuracy but shadow_engine already provides the signal. Accuracy dropped from 90% → 84% after removal. | Golden baseline re-established at 84% |
| `organic_discovery` | Heavyweight capability discovery (19 files, HDBSCAN + LLM naming) not feeding either output. Capabilities are manually seeded. | Removed CLI command `discover-organic`, weekly timer, 148 tests |
| `transfer_engine` | Cross-domain pattern transfer — building for a future that hasn't arrived. Also archived engine-side files (`transfer.py`, `transfer_generator.py`). | 34 tests removed |
| `activity_labeler` | LLM-based activity labels don't feed recommendations or anomaly detection directly. | Removed from validation scenarios |

## Structural Changes (2)

### data_quality → merged into discovery

Entity classification logic (auto-exclude domains, noise thresholds, vehicle detection, tier assignment) moved into `DiscoveryModule`. Classification runs on startup and daily via `hub.schedule_task("data_quality_reclassify")`. One fewer module, same functionality.

- Constants, config reading, and classification methods added to `discovery.py`
- `data_quality` Capability declaration moved to discovery's CAPABILITIES list (with `module="discovery"`)
- `entity_curation` cache now written by discovery instead of data_quality
- Tests adapted to call classification through `DiscoveryModule`

### pattern_recognition → renamed to trajectory_classifier

The old name collided conceptually with `patterns.py` (PatternRecognition). The rename clarifies that this module does trajectory classification (DTW + heuristic) on shadow_resolved events, not general pattern detection.

- Class: `PatternRecognitionModule` → `TrajectoryClassifier`
- module_id: `"pattern_recognition"` → `"trajectory_classifier"`
- All imports, lookups, patch paths, and test references updated

## Updated Artifacts

- `aria/cli.py` — Registration blocks removed/updated
- `aria/capabilities.py` — Imports and module lists updated
- `aria/hub/api.py` — Module lookups renamed
- `aria/modules/ml_engine.py` — Module lookup renamed
- `aria/modules/orchestrator.py` — Dependency renamed
- `aria/dashboard/spa/src/lib/pipelineGraph.js` — Sankey topology updated
- `docs/architecture-detailed.md` — Module table and package layout updated
- `docs/system-routing-map.md` — Routing tables, event bus, startup sequence updated
- `_archived/modules/README.md` — Triage rationale

## Commit History

1. `docs: add archived modules directory with triage rationale`
2. `refactor: archive online_learner module (lean audit)`
3. `refactor: archive transfer_engine module and engine transfer files (lean audit)`
4. `refactor: archive organic_discovery module (lean audit)`
5. `refactor: archive activity_labeler module (lean audit)`
6. `refactor: merge data_quality into discovery module (lean audit)`
7. `refactor: rename pattern_recognition to trajectory_classifier (lean audit)`
8. `fix: update pipeline Sankey for archived modules (lean audit)`

## Next Steps (Phase 2)

- **Known-answer test harness** — vertical integration tests that trace one real input through the full pipeline
- **Dashboard greyed-out modules** — Tier-gated modules (trajectory_classifier) show as greyed out when inactive, not hidden
- **CLAUDE.md update** — Update ha-aria CLAUDE.md to reflect leaner module count and test commands
