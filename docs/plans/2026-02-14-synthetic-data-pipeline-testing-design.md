# Synthetic Data Pipeline Testing

**Date:** 2026-02-14
**Status:** Design approved, pending implementation

## In Plain English

This plan creates a virtual household that generates realistic smart home data so ARIA's entire brain can be tested end-to-end without waiting weeks for real data to accumulate. It is like a flight simulator for pilots -- you can practice every scenario, including emergencies, without risking a real aircraft.

## Why This Exists

ARIA has nearly 750 unit tests, but none of them verify that the full pipeline actually works when all the pieces connect. Each stage (data collection, pattern learning, prediction, scoring) is tested in isolation, but the handoffs between stages are where bugs hide. Real household data takes weeks to accumulate and cannot be controlled. A synthetic household simulator lets us generate months of realistic data in seconds, run the complete pipeline against it, and verify that models actually learn, predictions actually improve, and the dashboard renders correctly -- all deterministically and repeatably.

## Problem

ARIA has 747 unit tests across 38 files, but integration testing is shallow. No test validates the full pipeline flow (snapshot -> baseline -> feature engineering -> model training -> prediction -> scoring -> meta-learning). The hub-engine handoff (engine writes JSON, hub reads from disk) is untested. There is no way to verify ML models actually learn from data, that Ollama meta-learning improves accuracy, or that the dashboard renders correctly with realistic data.

## Goals

1. **Integration trust** — verify pipeline stages compose correctly at every handoff
2. **Regression safety** — future changes don't silently break downstream stages
3. **ML validation** — models learn real patterns, improve with more data, beat naive baselines
4. **Ollama validation** — meta-learning loop produces valid output that improves model accuracy
5. **Visual verification** — dashboard renders correctly against real pipeline output

## Approach: Hybrid (Factory + Agents)

Deterministic pytest suite for CI regression protection, plus Claude sub-agents for exploratory testing and edge case discovery. Demo mode runs the full pipeline with simulated data for visual dashboard verification.

---

## 1. Household Simulator

### Core Concept

A realistic household simulator that produces HA state-change events based on people, devices, schedules, and environmental conditions. Data represents a real household — not tailored for specific test outcomes. Models must discover patterns on their own.

### Components

**HouseholdSimulator** — top-level orchestrator. Accepts a scenario configuration, seed for determinism, and number of days. Produces a stream of HA entity state-change events.

**Person** — models a resident with wake/sleep times, work schedule, room-to-room transitions. Drives occupancy, light usage, motion sensor triggers, door/lock events.

**DeviceRoster** — defines the entity set (lights, sensors, switches, climate, locks, media, EV, etc.). Each device has a domain, device class, state transitions, and watt rating. Template `typical_home` provides ~60 entities matching ARIA's real entity profile.

**WeatherProfile** — generates temperature, humidity, wind based on region and month. Influences climate setpoints and daylight hours.

**EntityStreamGenerator** — converts person movements + device states into HA-format state-change events with realistic timestamps, jitter, and noise. Supports configurable sensor gaps and unavailability periods.

**SnapshotAssembler** — feeds entity state events through ARIA's real `CollectorRegistry` (all 15 collectors) to produce snapshot dicts. This ensures synthetic snapshots match production format exactly and tests the collectors as part of the pipeline.

### Scenarios

Scenarios are household configurations, not outcome specifications. Each produces naturally occurring patterns that models should be able to learn.

| Scenario | Configuration | Natural result |
|----------|--------------|----------------|
| `stable_couple` | 2 residents, consistent schedules, 30 days | Predictable patterns — models should converge |
| `new_roommate` | 2 residents for 14 days, 3rd joins at day 15 | Power/occupancy shift — drift detector should notice |
| `vacation` | Both residents away days 10-17 | Dramatic activity drop — anomaly or regime shift |
| `work_from_home` | 1 resident switches to WFH at day 8 | Daytime patterns change — models must adapt |
| `sensor_degradation` | Battery sensors go unavailable at day 20 | Missing data — pipeline should degrade gracefully |
| `holiday_week` | Normal schedule with holiday flags | Weekend-like behavior on weekdays |

### Design Decisions

- **Real collectors, not synthetic snapshots.** The existing `make_synthetic_snapshots()` builds snapshot dicts directly, bypassing collectors. The simulator generates raw entity states and runs them through real collectors. This catches collector bugs in integration tests.
- **Deterministic via seed.** Same seed + same scenario = identical output. Tests are reproducible.
- **No embedded ground truth labels.** The data doesn't specify "this is anomalous" or "power correlates with occupancy at r=0.7." Models must discover patterns from realistic behavior.

---

## 2. ML Model Validation Suite

### Tier 1: Model Competence (sklearn)

Tests that models exhibit learner behavior against realistic simulated data. Assertions are relative (improvement, comparison) not absolute (hardcoded thresholds).

| Test | Validates | Assertion |
|------|-----------|-----------|
| `test_models_converge` | More data improves accuracy | R2 at day 25 > R2 at day 14 |
| `test_models_beat_naive` | ML adds value over baselines | After 21+ days, ML blend MAE < day-of-week average MAE |
| `test_models_generalize` | Not overfitting | Test accuracy within 20% of train accuracy |
| `test_models_degrade_gracefully` | Handles missing data | `sensor_degradation` scenario completes, predictions produced |
| `test_anomaly_detector_responds` | IsolationForest detects change | Anomaly scores increase after regime shift |
| `test_drift_detector_fires` | Page-Hinkley/ADWIN detect change | Drift flagged within days of pattern change |
| `test_cold_start_progression` | Pipeline advances stages | 7-day scenario hits collecting -> baselines -> ML training |

### Tier 2: Ollama Meta-Learning Loop

Validates that the deepseek-r1:8b meta-learning step produces valid output and improves model accuracy.

| Test | Validates | Assertion |
|------|-----------|-----------|
| `test_meta_learning_produces_valid_output` | LLM returns parseable JSON | Schema valid, param values in range |
| `test_meta_learning_references_real_metrics` | Reasoning is grounded | Suggestions reference actual scores, not hallucinated |
| `test_meta_learning_improves_or_holds` | Loop has positive/neutral effect | Cycle 2 accuracy >= Cycle 1 |
| `test_meta_learning_multi_cycle` | Sustained improvement | 3 cycles — accuracy doesn't degrade |
| `test_automation_suggestions_valid` | LLM produces usable automations | Valid YAML, entity IDs exist, triggers reference real patterns |

**Ollama test strategy:**
- CI mode: recorded Ollama responses replayed via mock client. Validates parsing and pipeline integration.
- Local mode (`@pytest.mark.ollama`): hits real deepseek-r1:8b. Validates LLM output quality.
- Record with `--record-ollama` flag, replay in CI. Re-record when meta-learning prompts change.

### Tier 3: End-to-End Pipeline Flow

Full pipeline execution validating every handoff point.

| Test | Validates |
|------|-----------|
| `test_full_pipeline_completes` | All stages produce output, no crashes |
| `test_intermediate_formats_valid` | Each stage output matches next stage's expected schema |
| `test_hub_reads_engine_output` | IntelligenceModule loads engine-produced JSON |
| `test_shadow_engine_scores_predictions` | Shadow mode compares predictions to actuals |

---

## 3. Demo Mode (Full Pipeline Visual Testing)

### Concept

`aria demo` runs the full pipeline with simulated data and starts the hub, so the dashboard displays output that flowed through every real code path. The only fake input is the household simulation — every pipeline stage is real.

### Commands

```bash
# Generate + run full pipeline + start hub
aria demo --scenario stable_couple --days 30

# Load previously frozen checkpoint (skip pipeline, fast)
aria demo --checkpoint day_30
```

### Pipeline Flow (demo mode)

```
HouseholdSimulator
  -> EntityStreamGenerator (HA state events)
    -> Real Collectors (SnapshotAssembler)
      -> Storage (JSON to temp dir)
        -> Baseline computation
          -> Feature engineering (mRMR)
            -> Model training (9 models)
              -> Prediction generation (blending)
                -> Scoring (MAE/R2)
                  -> Meta-learning (Ollama or replay)
                    -> Hub starts with pipeline output
                      -> Dashboard renders everything
```

### Frozen Checkpoints

Generated once from the simulator, saved as static fixtures for fast UI iteration:

| Checkpoint | System state | UI testing focus |
|------------|-------------|-----------------|
| `day_07` | Cold-start: baselines only | Empty states, "insufficient data" messages |
| `day_14` | Early ML: first predictions | Partial data charts, wide confidence intervals |
| `day_30` | Mature: full system | All pages populated, SHAP, correlations |
| `day_45` | Post-drift: anomalies flagged | Anomaly highlights, drift indicators, accuracy dip |

### Implementation

Add `--demo` option to hub CLI:

```python
# aria/hub/cli.py
@click.option("--demo", type=str, help="Scenario name or checkpoint path")
@click.option("--demo-days", type=int, default=30)
@click.option("--checkpoint", type=str, help="Load frozen checkpoint")
```

When `--demo` is passed: run simulator + full pipeline to temp dir, then start hub pointing at that dir. When `--checkpoint` is passed: skip pipeline, load frozen fixtures.

---

## 4. Agent Architecture

### Generator Agent (`aria-generator`)

Explores the pipeline for weaknesses by designing and running new household scenarios.

- **Type:** general-purpose (needs Bash + Read + Write)
- **Access:** ha-aria project, tests/synthetic/ factory, temp directories
- **Constraints:** writes only to temp dirs and tests/ — never modifies aria/ source
- **Use case:** "stress test drift detection", "what happens with erratic occupancy?"

### Auditor Agent (`aria-auditor`)

Reviews pipeline outputs for correctness, validates ML behavior, audits test coverage.

- **Type:** general-purpose (needs Read + Grep + Bash for pytest)
- **Access:** full ha-aria read access, temp output dirs
- **Constraints:** read-only on source, can write new test files to tests/
- **Use case:** "review these outputs", "where are coverage gaps?"

### Feedback Loop

```
Agent discovers failure
  -> You verify it's real
    -> Add scenario to HouseholdSimulator
      -> Write deterministic pytest case
        -> Failure caught permanently in CI
```

Agents are disposable explorers. The pytest suite is the permanent record.

---

## 5. File Structure

```
tests/
  synthetic/
    __init__.py
    simulator.py          # HouseholdSimulator
    entities.py           # EntityStreamGenerator, DeviceRoster
    people.py             # Person, Schedule, room transitions
    weather.py            # WeatherProfile
    assembler.py          # SnapshotAssembler (uses real collectors)
    pipeline.py           # PipelineRunner (orchestrates full pipeline)
    scenarios/
      __init__.py
      household.py        # stable_couple, new_roommate, vacation, etc.
  integration/
    __init__.py
    test_model_competence.py    # Tier 1
    test_meta_learning.py       # Tier 2
    test_pipeline_flow.py       # Tier 3
    conftest.py                 # Shared fixtures, pipeline runner setup
  demo/
    generate.py                 # Freeze simulator outputs to checkpoints
    fixtures/
      day_07/
      day_14/
      day_30/
      day_45/
  fixtures/
    ollama_responses/           # Recorded Ollama outputs for CI replay
.claude/
  agents/
    aria-generator.md           # Scenario exploration agent
    aria-auditor.md             # Output validation + coverage agent
```

---

## 6. Implementation Phases

| Phase | What | Depends on |
|-------|------|-----------|
| 1 | HouseholdSimulator core — people, devices, entity streams | Nothing |
| 2 | SnapshotAssembler — entity streams through real collectors | Phase 1 |
| 3 | 3-4 household scenarios | Phase 1 |
| 4 | PipelineRunner — orchestrates full pipeline in temp dirs | Phase 2 |
| 5 | Tier 1 pytest — model competence tests | Phases 3 + 4 |
| 6 | Tier 3 pytest — E2E flow and handoff validation | Phase 4 |
| 7 | Ollama record/replay infrastructure | Phase 4 |
| 8 | Tier 2 pytest — meta-learning validation | Phase 7 |
| 9 | Demo mode — `aria demo` CLI + frozen checkpoints | Phases 3 + 4 |
| 10 | Agent definitions | Phases 3 + 4 + 5 |

Phases 1-6 deliver the core value. Phases 7-10 are incremental additions.

---

## Scope Boundaries

- Does NOT test real-time WebSocket hub (mocking HA WebSocket is a separate effort)
- Does NOT test Preact dashboard rendering (visual verification only via demo mode)
- Does NOT replace existing 747 unit tests (adds integration layer on top)
- Does NOT require changes to production ARIA code except the `--demo` CLI flag
