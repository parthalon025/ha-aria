# Metamorphic Assertions + Real-Data Backtest Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two complementary testing layers to the ARIA validation suite: metamorphic invariant assertions and real-data backtesting.

**Architecture:** Metamorphic tests extend `TestCrossScenarioComparisons` with invariant relationships between scenarios. Backtest layer loads real intraday snapshots from `~/ha-logs/intelligence/intraday/` and runs them through `PipelineRunner`, comparing predictions against actual outcomes.

**Tech Stack:** pytest, existing PipelineRunner, pathlib for real data loading, json/gzip for snapshot parsing

---

### Task 1: Metamorphic Assertions

**Files:**
- Modify: `tests/integration/test_validation_scenarios.py`

Add 5 metamorphic tests to `TestCrossScenarioComparisons`:

1. **`test_stable_more_accurate_than_vacation`** — stable_couple overall score >= vacation overall score (predictable patterns should be easier to model)
2. **`test_wfh_more_accurate_than_new_roommate`** — work_from_home >= new_roommate (consistent pattern > changing pattern). Allow 5pt variance.
3. **`test_scenario_ranking_deterministic`** — Run stable_couple twice with same seed, scores must be identical
4. **`test_higher_occupancy_predicts_higher_power`** — stable_couple predicted power > vacation predicted power (more people = more energy)
5. **`test_degradation_does_not_improve_accuracy`** — sensor_degradation overall <= stable_couple overall (bad data shouldn't produce better models)

These use the existing `all_scenario_results` fixture — no new infrastructure needed.

### Task 2: Real-Data Loader

**Files:**
- Create: `tests/synthetic/real_data.py`

Create `RealDataLoader` class:
- `load_intraday(base_dir, min_days=3) -> list[dict]` — reads `~/ha-logs/intelligence/intraday/YYYY-MM-DD/HH.json` files, returns list of snapshots sorted chronologically
- `load_daily(base_dir) -> list[dict]` — reads `~/ha-logs/intelligence/daily/YYYY-MM-DD.json{.gz}` files
- Handles both `.json` and `.json.gz` formats
- Adds `time_features.hour` if missing (parse from filename)
- Returns empty list if data dir doesn't exist (CI-safe)

### Task 3: Real-Data Backtest Tests

**Files:**
- Create: `tests/integration/test_validation_backtest.py`

Tests (all skip gracefully if `~/ha-logs/intelligence/intraday/` has < 3 days of data):

1. **`test_real_data_loads_successfully`** — RealDataLoader produces snapshots with expected keys
2. **`test_real_pipeline_completes`** — PipelineRunner.run_full() succeeds on real data without errors
3. **`test_real_predictions_are_numeric`** — All prediction values (power_watts, lights_on, devices_home) are numeric
4. **`test_real_accuracy_above_minimum`** — Overall score > 0% (pipeline isn't fundamentally broken on real data)
5. **`test_real_vs_synthetic_schema_compatible`** — Real snapshot keys are superset of synthetic snapshot keys (validates simulator fidelity)
6. **`test_real_accuracy_report`** — Print formatted report comparing real vs synthetic accuracy side by side

### Task 4: Golden Baseline Storage

**Files:**
- Create: `tests/integration/golden/` directory
- Modify: `tests/integration/test_validation_backtest.py`

Add golden baseline test:
1. **`test_golden_baseline_regression`** — If `tests/integration/golden/backtest_baseline.json` exists, compare current real-data scores against stored baseline. Fail if accuracy drops > 5 points. If file doesn't exist, create it (first run establishes baseline).

### Task 5: Commit and Verify

Run full validation suite, commit all changes.
