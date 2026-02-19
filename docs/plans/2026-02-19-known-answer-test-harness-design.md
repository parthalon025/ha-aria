# Known-Answer Test Harness — Phase 2 Design

**Date:** 2026-02-19
**Status:** Approved
**Author:** Justin McFarland + Claude
**Parent:** `docs/plans/2026-02-19-lean-audit-restructure-design.md` (Phase 2)

## Problem

After Phase 1, ARIA has 10 surviving modules and 1,416 tests — but intelligence quality is still unvalidated. The existing test suite proves modules initialize and don't crash, but never asks: "given this input, does ARIA produce the correct recommendation or detect the correct anomaly?"

## Goal

Build a known-answer test harness that feeds deterministic inputs through each module and the full pipeline, then asserts that outputs match expected behavior. Discrepancies = intelligence bugs.

## Approach: Three-Layer Testing

### Layer 1: Per-Module Known-Answer Tests (10 modules)

Each surviving module gets an isolated known-answer test:
- Feed a deterministic fixture (from simulator or hand-crafted)
- Assert **behavioral properties** (hard pass/fail): "finds >= 2 patterns", "flags anomaly", "produces recommendation"
- Compare against **golden snapshot** (warn on drift, not fail): full output diffed against reference JSON

| Module | Fixture Source | Key Behavioral Assertion |
|--------|---------------|------------------------|
| `discovery` | Mocked HA subprocess | Finds N entities, classifies tiers correctly |
| `activity_monitor` | Simulated `state_changed` events | Produces activity windows with correct entity counts |
| `patterns` | Simulator logbook (stable_couple, 21d) | Finds >= 2 recurring patterns |
| `orchestrator` | Hand-crafted pattern list | Produces >= 1 automation recommendation |
| `shadow_engine` | Simulator state changes (vacation) | Flags vacation-day anomaly, accuracy within range |
| `trajectory_classifier` | Known shadow_resolved events | Classifies trajectory type correctly |
| `ml_engine` | Simulator snapshots (14d) | Trains models, accuracy > 0 |
| `intelligence` | Engine JSON output file | Reads and caches all metric paths correctly |
| `presence` | Simulated MQTT messages | Updates occupancy state correctly |
| `audit_logger` | Module action events | Writes tamper-evident log entries with correct fields |

### Layer 2: Full Pipeline Known-Answer Test

One end-to-end test tracing a scenario through the complete pipeline:

```
Simulator data (stable_couple, 21d)
  → engine snapshot processing
    → patterns module (finds recurring patterns)
      → orchestrator (generates recommendations)
    → shadow_engine (predicts + compares)
      → trajectory_classifier (classifies + explains anomalies)
        → Final outputs: recommendations[] + anomalies[]
```

Behavioral assertions on final outputs:
- Recommendations list is non-empty with valid YAML
- Anomalies list includes expected deviation signals
- Both outputs reference real entities from the fixture

### Layer 3: Golden Snapshot Infrastructure

Shared utility for optional golden file comparison:

- `golden_compare(actual, golden_path)` — diffs actual output against stored reference
- Drift produces a **warning** (printed in test output) not a **failure**
- `--update-golden` pytest CLI flag regenerates all golden files from current output
- Golden files are committed to git — they document "what ARIA currently produces"

## File Structure

```
tests/integration/known_answer/
├── conftest.py              # golden_compare(), --update-golden flag, shared fixtures
├── golden/                  # reference output JSON files (committed)
│   ├── discovery.json
│   ├── activity_monitor.json
│   ├── patterns.json
│   ├── orchestrator.json
│   ├── shadow_engine.json
│   ├── trajectory_classifier.json
│   ├── ml_engine.json
│   ├── intelligence.json
│   ├── presence.json
│   ├── audit_logger.json
│   └── full_pipeline.json
├── fixtures/                # hand-crafted edge-case inputs
│   ├── anomaly_spike.json
│   ├── empty_house.json
│   └── multi_pattern.json
├── test_discovery_ka.py
├── test_activity_monitor_ka.py
├── test_patterns_ka.py
├── test_orchestrator_ka.py
├── test_shadow_engine_ka.py
├── test_trajectory_classifier_ka.py
├── test_ml_engine_ka.py
├── test_intelligence_ka.py
├── test_presence_ka.py
├── test_audit_logger_ka.py
└── test_full_pipeline_ka.py
```

## Fixture Strategy

**Simulator-based:** Use `HouseholdSimulator` (seed=42) for modules that need realistic multi-day data (patterns, shadow_engine, ml_engine, full pipeline). Deterministic seed guarantees reproducible output.

**Hand-crafted:** Create minimal JSON fixtures for edge cases the simulator doesn't naturally produce:
- `anomaly_spike.json` — single entity with obvious anomaly (tests detection sensitivity)
- `empty_house.json` — no activity at all (tests boundary handling)
- `multi_pattern.json` — overlapping patterns (tests disambiguation)

**Mocked services:** For modules that call external services (discovery → HA subprocess, presence → MQTT), use deterministic mock responses.

## Quick Wins (same branch)

### Dashboard Greyed-Out Modules

Tier-gated modules (trajectory_classifier at Tier 3+) show as greyed out in the pipeline Sankey and module list, not hidden. Users see the full architecture but understand which modules are active.

### CLAUDE.md Update

Update `ha-aria/CLAUDE.md` to reflect:
- 10 surviving modules (not 14)
- Updated test commands (remove archived module keywords)
- Known-answer test run command
- Remove references to archived modules in gotchas/examples

## Success Criteria

- [ ] All 10 modules have known-answer tests with behavioral assertions passing
- [ ] Full pipeline known-answer test passes end-to-end
- [ ] Golden snapshots established and committed for all modules
- [ ] `pytest tests/integration/known_answer/ -v` passes cleanly
- [ ] `--update-golden` flag works to re-baseline
- [ ] Dashboard shows greyed-out modules for tier-gated features
- [ ] CLAUDE.md reflects Phase 1 changes accurately
- [ ] No regressions in existing 1,416 tests
