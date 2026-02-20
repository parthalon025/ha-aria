# ARIA Full-Stack Validation Suite — Design

**Date:** 2026-02-16
**Purpose:** Repeatable test suite that validates ARIA works end-to-end after any update, producing a single accuracy percentage as the KPI.

## Goal

One command: `pytest tests/integration/test_validation_*.py -v`
One number: **ARIA Prediction Accuracy: X%**

The suite must be:
- **Repeatable** — seeded RNG, no external dependencies, runs in tmp dirs
- **Deterministic** — same code = same results every time
- **Comprehensive** — tests engine pipeline, hub modules, WebSocket events, CLI commands, and API endpoints
- **Self-contained** — no HA connection, no Ollama, no MQTT needed

## Architecture

### Data Flow

```
HouseholdSimulator (6 scenarios, seed=42)
  → SnapshotAssembler (real collectors, synthetic entity states)
    → PipelineRunner (save → baselines → train → predict → score)
      → IntelligenceHub (async, mock HA connections, seed cache)
        → Module initialization + cache population
          → API routes via TestClient
            → Final accuracy calculation
```

### Test Files

| File | Tests | What It Proves |
|------|-------|---------------|
| `test_validation_engine.py` | ~18 | Engine pipeline produces correct, predictable results across all 6 scenarios |
| `test_validation_hub.py` | ~14 | Hub boots, modules init, cache populates, APIs respond, WebSocket events fire |
| `test_validation_cli.py` | ~4 | CLI entry points work against synthetic data |
| `test_validation_scenarios.py` | ~8 | Cross-scenario comparisons prove ARIA learns, plus final accuracy KPI |

### Fixture Strategy

Module-scoped fixtures avoid re-running expensive pipeline for each test:

```python
@pytest.fixture(scope="module")
def stable_30d(tmp_path_factory):
    """30-day stable household, full pipeline."""
    ...

@pytest.fixture(scope="module")
def all_scenarios(tmp_path_factory):
    """All 6 scenarios, full pipeline, shared for comparison tests."""
    ...
```

## Section 1: Engine Validation

### Stable Couple (baseline scenario)
- Pipeline completes, score > 0
- Power predictions in range (150W–3000W)
- Weekday baselines differ from weekend
- All target metrics produce trained models
- At least 2 metrics have R2 > 0.0
- Feature vector shape matches config

### Vacation
- Occupancy during days 10-17 < pre-vacation
- Power drops during vacancy
- Pipeline completes despite anomalous period

### Work From Home
- Daytime occupancy increases after day 8
- Daytime power shifts after WFH starts

### New Roommate
- Post-day-15 occupancy > pre-day-15
- Post-day-15 power > pre-day-15

### Sensor Degradation
- Pipeline completes despite degraded sensors
- Unavailable entity count increases over time

### Holiday Week
- Holiday flags present on days 24-26
- Pipeline completes normally

### Cold Start
- 7-day data produces baselines + predictions

### Schema
- Every scenario's snapshots have all required keys

## Section 2: Hub Validation

### Module Initialization
- Hub initializes without error
- IntelligenceModule reads engine output into cache
- MLEngine module trains from pipeline data
- ShadowEngine initializes
- DataQualityModule classifies entities
- PatternRecognitionModule finds patterns in logbook data
- Expected cache categories populated

### API Endpoints
- `/api/health` returns correct shape
- All `/api/cache/{category}` endpoints return data
- `/api/cache/intelligence` returns engine data

### WebSocket
- Cache update triggers WebSocket event
- WebSocket payload contains updated category
- Sequential updates deliver in order

### Event Bus
- publish/subscribe delivers events

## Section 3: CLI Validation

- `aria status` completes
- Engine snapshot processing via CLI entry
- `aria train` produces model files
- `aria predict` produces prediction JSON

## Section 4: Accuracy KPI

### Calculation Method

For each scenario:
1. Run full pipeline (save → baselines → train → predict → score)
2. For each predicted metric (power_watts, lights_on, occupancy_count, locks_locked):
   - `metric_accuracy = 1 - abs(predicted - actual) / max(actual, 1)`
   - Clamp to [0, 1]
3. `scenario_accuracy = mean(metric_accuracies)`
4. `overall_accuracy = mean(scenario_accuracies)` — weighted by scenario days

### Output

Tests print a formatted report visible in pytest output:

```
╔══════════════════════════════════════════════════╗
║           ARIA VALIDATION REPORT                 ║
╠══════════════════════════════════════════════════╣
║ Scenario            │ Accuracy │ Power │ Occ.   ║
║ stable_couple       │   78%    │  82%  │  74%   ║
║ vacation            │   65%    │  70%  │  60%   ║
║ work_from_home      │   71%    │  68%  │  74%   ║
║ new_roommate        │   69%    │  72%  │  66%   ║
║ sensor_degradation  │   58%    │  55%  │  61%   ║
║ holiday_week        │   75%    │  78%  │  72%   ║
╠══════════════════════════════════════════════════╣
║ OVERALL ACCURACY    │   69%    │       │        ║
╚══════════════════════════════════════════════════╝
```

### Thresholds
- **Pass**: Overall accuracy > 40% (synthetic data limits ceiling)
- **Warning**: Below 50% — investigate which scenario/metric degraded
- **Fail**: Below 40% — something is fundamentally broken

## Modules Under Test

All 10 hub modules will be exercised to varying degrees:

| Module | Testable Without HA? | How |
|--------|---------------------|-----|
| intelligence | Yes | Reads engine JSON files — fully testable |
| ml_engine | Yes | Trains from snapshots — fully testable |
| shadow_engine | Partial | Init testable, predictions need event stream |
| data_quality | Yes | Reads discovery cache — seed with synthetic entities |
| pattern_recognition | Yes | Reads logbook data — synthetic logbook_summary |
| orchestrator | Partial | Needs pattern_recognition output |
| discovery | No | Requires HA REST/WebSocket — mock only |
| organic_discovery | Partial | Clustering testable, Ollama naming mocked |
| activity_monitor | No | Requires HA WebSocket — mock only |
| presence | No | Requires MQTT + HA WebSocket — mock only |

## Not In Scope

- Live HA integration testing (requires running instance)
- Ollama/LLM naming tests (requires model server)
- MQTT/Frigate presence tests (requires running services)
- Dashboard rendering tests (Preact SPA, separate concern)
- Performance benchmarking (separate suite)
