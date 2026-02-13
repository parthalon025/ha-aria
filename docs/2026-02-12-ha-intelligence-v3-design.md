# HA Intelligence Engine v3: Modular Architecture + Advanced ML

**Date:** 2026-02-12
**Status:** Planning
**Author:** Justin McFarland + Claude
**Project:** `~/Documents/projects/ha-intelligence/`

---

## Executive Summary

Research spike (Feb 12, 2026) surveyed 20+ open-source HA ML projects, the broader smart home ML landscape, and modular ML architecture best practices. Key finding: **ha-intelligence has no direct competitor in the HA ecosystem** — it's the only general-purpose intelligence layer combining ML prediction + anomaly detection + LLM meta-learning across all HA domains. The closest projects (TheSillyHome, MyHomeSmart) are dead or low-activity.

This plan restructures the engine from a 2,373-line monolith into a modular package, then incrementally adds advanced ML capabilities identified in the research. Four phases, each independently valuable.

---

## Research Findings Summary

### Your Approach vs. the Field

| Decision | Field Consensus | Verdict |
|----------|-----------------|---------|
| GradientBoosting for tabular sensor data | Best performer across all studies (XGBoost, LightGBM, sklearn) | **Correct** |
| IsolationForest for anomaly detection | Standard for IoT — 22ms inference, unsupervised, low memory | **Correct** |
| Snapshot-based architecture | Used by EMHASS, most research systems | **Correct, hybrid is ideal** |
| LLM meta-learning loop (deepseek-r1:8b) | Only ai_automation_suggester does similar (shallower — one-shot, not adaptive) | **Ahead of field** |
| Phased ramp-up (Day 1-7 stats, Day 14+ ML) | Solar Forecast ML uses identical pattern (Day 0-30 progression) | **Validated** |
| Weekly retrain schedule | Works, but drift detection is more efficient | **Good, upgradeable** |

### Competitive Landscape (HA Ecosystem)

**Tier 1 — Production-ready, actively maintained:**
| Project | Domain | Stars | Approach |
|---------|--------|-------|----------|
| EMHASS | Energy optimization | 511 | sklearn forecasting + LP optimization (CVXPY/HiGHS) |
| Frigate | Computer vision | 30,178 | TensorFlow/YOLO real-time object detection |
| Home-LLM | Voice/chat control | 1,226 | Fine-tuned <5B LLMs for device control |
| Better Thermostat | Climate control | 1,333 | MPC, PID, TPI, auto-calibration |
| Versatile Thermostat | Climate control | 944 | Auto-TPI with learned coefficients (~48h) |
| AI Automation Suggester | Automation generation | 685 | LLM scans entities, suggests YAML automations |
| HA WashData | Appliance detection | 572 | NumPy correlation scoring on power profiles |

**Tier 2 — Promising, moderate maturity:**
| Project | Domain | Stars | Approach |
|---------|--------|-------|----------|
| Area Occupancy Detection | Presence | 220 | Bayesian inference with adaptive prior learning |
| SAT (Smart Autotune) | Climate | 235 | PID auto-tuning (relay feedback method) |
| Solar Forecast ML | Solar forecasting | 116 | Attention Transformer + adaptive ensemble |

**Key gap:** Anomaly detection is the biggest unserved need in the HA ecosystem. Almost nothing exists. HomeDetector (6 stars) does DNS anomaly detection only.

### ML Techniques Ranked for Home Sensor Data

| Technique | Suitability | Data Needed | You Have It? |
|-----------|-------------|-------------|--------------|
| Gradient Boosting (sklearn) | Excellent | 2-4 weeks | Yes |
| Isolation Forest | Excellent | 2 weeks normal data | Yes |
| Prophet (seasonal decomposition) | Good for daily/weekly energy | Ideally 1+ year, usable at 4-8 weeks | No — add in Phase 3 |
| LSTM | Good for sequences | 4-8 weeks | No — add in Phase 4 |
| TCN (Temporal Convolutional Network) | Good, underexplored | 2-4 weeks | No — future |
| LSTM-Autoencoder | Good for sequence anomalies | 4-8 weeks | No — add in Phase 4 |
| HMM (Hidden Markov Model) | Good for activity chains | 2-4 weeks | No — future |
| Bayesian inference | Good for occupancy fusion | 2-4 weeks | No — add in Phase 3 |
| Reinforcement Learning (SAC/PPO) | Niche — HVAC only | Simulation + 1-2 weeks | No — future |
| KAN (Kolmogorov-Arnold Networks) | Emerging — watch space | 2-4 weeks | No — future |

### Architecture Patterns

| Pattern | Description | Recommendation |
|---------|-------------|----------------|
| Snapshot-based | Periodic state collection, batch analysis | Keep — it works for daily/weekly patterns |
| Event-driven | WebSocket state_changed stream, real-time | Already have via ha-intelligence-hub |
| Hybrid | Snapshot for training + events for real-time | **Target state** — combine engine + hub |
| Registry pattern | Decorator-based extensibility for collectors/models | **Adopt in Phase 1** |
| Dataclass config | Type-safe config objects over module globals | **Adopt in Phase 1** |
| Data contracts | TypedDict/dataclass for snapshot schema | **Adopt in Phase 1** |
| DataStore abstraction | Unified I/O layer (enables future SQLite migration) | **Adopt in Phase 1** |

---

## Target Architecture

### Package Structure

```
ha-intelligence/
├── bin/
│   ├── ha-intelligence          # Thin shim: from ha_intelligence.cli import main; main()
│   └── ha-log-sync              # Stays as-is (independent script)
├── ha_intelligence/
│   ├── __init__.py              # Version, package metadata
│   ├── config.py                # Dataclass configs (HA, paths, models, Ollama, features)
│   ├── cli.py                   # argparse + command dispatch
│   ├── collectors/
│   │   ├── __init__.py          # CollectorRegistry.all() convenience
│   │   ├── registry.py          # CollectorRegistry + BaseCollector ABC
│   │   ├── ha_api.py            # fetch_ha_states(), fetch_weather(), fetch_calendar()
│   │   ├── extractors.py        # All 16 extract_* as registered collector classes
│   │   ├── logbook.py           # load_logbook(), summarize_logbook()
│   │   └── snapshot.py          # build_snapshot(), build_intraday_snapshot(), aggregate
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_features.py     # cyclical_encode(), build_time_features()
│   │   ├── vector_builder.py    # build_feature_vector(), get_feature_names()
│   │   └── feature_config.py    # DEFAULT_FEATURE_CONFIG, load/save
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py          # ModelRegistry + BaseModel ABC
│   │   ├── gradient_boosting.py # GradientBoosting wrapper (train + predict)
│   │   ├── random_forest.py     # RandomForest wrapper
│   │   ├── isolation_forest.py  # IsolationForest anomaly detector
│   │   ├── device_failure.py    # Device failure model (train + predict)
│   │   └── training.py          # train_all_models() orchestrator
│   ├── predictions/
│   │   ├── __init__.py
│   │   ├── predictor.py         # predict_with_ml(), blend_predictions(), generate_predictions()
│   │   └── scoring.py           # score_prediction(), score_all_predictions(), accuracy_trend()
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── baselines.py         # compute_baselines()
│   │   ├── correlations.py      # pearson_r(), cross_correlate()
│   │   ├── anomalies.py         # detect_anomalies(), detect_contextual_anomalies()
│   │   └── reliability.py       # compute_device_reliability()
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── data_store.py        # DataStore class — unified JSON I/O
│   │   └── model_io.py          # Pickle save/load for sklearn models
│   └── llm/
│       ├── __init__.py
│       ├── client.py            # ollama_chat(), strip_think_tags()
│       ├── reports.py           # generate_insight_report(), generate_brief_line()
│       └── meta_learning.py     # run_meta_learning(), parse/apply/validate suggestions
├── tests/
│   ├── conftest.py              # Shared fixtures (sample_states, sample_snapshot, sample_30d, tmp_data_dir, mock_ha_api)
│   ├── test_collectors.py       # Extractor unit tests
│   ├── test_features.py         # Feature engineering tests
│   ├── test_models.py           # ML model tests (with mocks)
│   ├── test_predictions.py      # Prediction + scoring tests
│   ├── test_analysis.py         # Baselines, correlations, anomaly tests
│   ├── test_storage.py          # DataStore I/O tests
│   ├── test_llm.py              # Ollama client + report tests
│   └── test_cli.py              # CLI integration tests
├── docs/
├── requirements.txt
├── pyproject.toml               # New — proper Python package metadata
└── CLAUDE.md
```

### Key Design Patterns

#### Registry Pattern (Collectors)

```python
# ha_intelligence/collectors/registry.py
from abc import ABC, abstractmethod

class CollectorRegistry:
    _collectors: dict[str, type] = {}

    @classmethod
    def register(cls, name: str = None):
        def decorator(collector_class):
            key = name or collector_class.__name__.lower()
            cls._collectors[key] = collector_class
            return collector_class
        return decorator

    @classmethod
    def all(cls) -> dict[str, type]:
        return dict(cls._collectors)

class BaseCollector(ABC):
    @abstractmethod
    def extract(self, snapshot: dict, states: list[dict]) -> None:
        """Extract data from HA states into snapshot dict (in-place)."""
        ...
```

Usage — adding a new device type:
```python
@CollectorRegistry.register("power")
class PowerCollector(BaseCollector):
    def extract(self, snapshot, states):
        # existing extract_power() logic
        ...
```

#### Registry Pattern (Models)

Same pattern for ML models. Each model self-registers. `train_all_models()` iterates the registry instead of hardcoding model names.

#### Dataclass Config

```python
# ha_intelligence/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HAConfig:
    url: str = "http://192.168.1.35:8123"
    token: str = ""

    @classmethod
    def from_env(cls):
        import os
        return cls(url=os.environ.get("HA_URL", cls.url),
                   token=os.environ.get("HA_TOKEN", ""))

@dataclass
class PathConfig:
    data_dir: Path = Path.home() / "ha-logs" / "intelligence"

    @property
    def daily_dir(self) -> Path: return self.data_dir / "daily"

    @property
    def intraday_dir(self) -> Path: return self.data_dir / "intraday"

    @property
    def models_dir(self) -> Path: return self.data_dir / "models"

    @property
    def baselines_path(self) -> Path: return self.data_dir / "baselines.json"
    # ... etc

@dataclass
class ModelConfig:
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1
    min_training_samples: int = 14

@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434/api/chat"
    model: str = "deepseek-r1:8b"
    timeout: int = 60
```

#### DataStore (Unified I/O)

```python
# ha_intelligence/storage/data_store.py
class DataStore:
    def __init__(self, paths: PathConfig):
        self.paths = paths

    def load_snapshot(self, date_str: str) -> dict | None: ...
    def save_snapshot(self, snapshot: dict): ...
    def load_recent_snapshots(self, days: int = 30) -> list[dict]: ...
    def load_baselines(self) -> dict: ...
    def save_baselines(self, baselines: dict): ...
    def load_predictions(self) -> dict | None: ...
    def save_predictions(self, predictions: dict): ...
    def load_correlations(self) -> dict: ...
    def save_correlations(self, correlations: dict): ...
    def load_accuracy_history(self) -> list[dict]: ...
    def update_accuracy_history(self, new_score: dict): ...
    # All JSON I/O in one place — enables future SQLite migration
```

#### Snapshot Type Contract

```python
# ha_intelligence/collectors/snapshot.py
from typing import TypedDict, Optional

class PowerData(TypedDict):
    total_watts: float
    outlets: dict[str, float]

class LightsData(TypedDict):
    on: int
    off: int
    unavailable: int
    total_brightness: int

class ClimateData(TypedDict):
    mode: str
    target_temp: float
    current_temp: float
    humidity: float

class Snapshot(TypedDict):
    date: str
    day_of_week: str
    is_weekend: bool
    is_holiday: bool
    holiday_name: Optional[str]
    weather: dict
    calendar_events: list
    power: PowerData
    lights: LightsData
    climate: ClimateData
    # ... all domains
```

---

## Implementation Phases

### Phase 1: Foundation (Modularize the Engine)

**Goal:** Restructure monolith into modular package. Zero functional changes.
**Prerequisite:** None
**Data requirement:** None (pure refactor)

#### Deliverables
1. Create `ha_intelligence/` package with 8 modules
2. Implement `config.py` with dataclass configs
3. Implement `CollectorRegistry` + migrate 16 extract_* functions
4. Implement `ModelRegistry` + migrate ML models
5. Implement `DataStore` + migrate all JSON I/O
6. Implement `Snapshot` TypedDict contract
7. Create `pyproject.toml` for proper package metadata
8. Migrate `bin/ha-intelligence` to thin shim
9. Split test suite into per-module files with shared `conftest.py`
10. Verify all existing CLI commands work identically

#### Test strategy
- Run existing 78 tests against modularized code — all must pass
- Add new tests for registry pattern, DataStore, config loading
- Integration test: `--full` pipeline produces identical output to monolith

#### Migration approach (incremental, not big-bang)
1. Create package structure, start with `config.py` and `storage/`
2. Move collectors to `collectors/` with registry
3. Move features to `features/`
4. Move models to `models/` with registry
5. Move analysis to `analysis/`
6. Move predictions to `predictions/`
7. Move LLM code to `llm/`
8. Wire up `cli.py`, update bin shim
9. Split tests
10. Final verification against production cron pipeline

Each step is independently testable. The original monolith script can be kept as a reference until Phase 1 is complete.

---

### Phase 2: Quick Wins (High Value, Low Effort)

**Goal:** Add three capabilities that leverage existing infrastructure.
**Prerequisite:** Phase 1 complete
**Data requirement:** 2+ weeks of snapshots (Day 14+)

#### 2a. Drift Detection

**Problem:** Weekly retrain burns compute when nothing has changed. Sudden behavioral shifts (vacation, new schedule) aren't caught until the next Monday retrain.

**Solution:** Monitor rolling prediction error (MAE). Trigger retrain when error exceeds 2x the rolling baseline. Skip scheduled retrain when error is stable.

**Implementation:**
- Add `analysis/drift.py` — `DriftDetector` class
- Track rolling MAE over 7-day window
- Threshold: retrain when MAE > 2 * median(recent_7d_MAE)
- Add `--check-drift` CLI command (called by cron, triggers `--retrain` if needed)
- Replace fixed Monday retrain cron with daily drift check

**Complexity:** Low — ~100 lines. Uses existing prediction scoring infrastructure.

**Operational impact:** Reduces unnecessary weekly retrains. Catches drift within 24h instead of waiting up to 7 days.

#### 2b. Entity Correlation Matrix

**Problem:** Cross-entity correlations are computed (Pearson r) but only used for statistical predictions. The most valuable use — discovering automation-worthy patterns — isn't implemented.

**Solution:** Build a co-occurrence matrix of entity state changes within time windows. Track conditional probabilities: P(lights.living_room ON | motion.hallway ON, hour=19-22).

**Implementation:**
- Add `analysis/entity_correlations.py` — beyond Pearson r
- Time-windowed co-occurrence tracking (5min, 15min, 1hr windows)
- Conditional probability computation for state transitions
- Store as enriched correlation data (extends existing `correlations.json`)
- Feed into meta-learning prompt as structured context

**Complexity:** Low-medium — ~200 lines. Builds on existing correlation infrastructure.

#### 2c. LLM Automation Suggestions

**Problem:** Meta-learning analyzes model performance but doesn't suggest HA automations.

**Solution:** Extend the meta-learning LLM prompt to include entity correlations and generate HA automation YAML suggestions.

**Implementation:**
- Extend `llm/meta_learning.py` with automation suggestion mode
- New prompt template that includes top entity correlations + behavioral patterns
- LLM generates HA automation YAML (validated before presenting)
- Add `--suggest-automations` CLI command
- Output to `insights/automation-suggestions/` as timestamped YAML files

**Complexity:** Low — ~150 lines. Leverages existing Ollama integration + new correlation data from 2b.

**Inspiration:** AI Automation Suggester (685 stars) does this with one-shot LLM calls. Our version is smarter because it's grounded in actual learned correlations, not just entity listing.

---

### Phase 3: New ML Capabilities

**Goal:** Add three new ML techniques that complement existing GradientBoosting + IsolationForest.
**Prerequisite:** Phase 2 complete
**Data requirement:** 4-8 weeks of snapshots minimum

#### 3a. Prophet for Seasonal Decomposition

**Problem:** GradientBoosting predicts well from features but doesn't decompose time series into trend + seasonal + residual components. Energy and climate patterns have strong daily/weekly/seasonal cycles.

**Solution:** Add Facebook Prophet models for energy consumption and climate metrics. Prophet handles holidays natively (you already track them).

**Implementation:**
- Add `prophet` to requirements.txt
- New model: `models/prophet_forecaster.py` — registered via ModelRegistry
- Train Prophet on power_total_watts, climate target/current temp, lights_on
- Use additive seasonality (daily, weekly) + holiday effects
- Prophet predictions blended with GradientBoosting predictions (weighted by data maturity)
- Prophet needs daily frequency data — perfect match for your daily snapshots

**Complexity:** Medium — ~200 lines. Prophet is well-documented and designed for non-experts.

**Data maturity:** Usable at 4-8 weeks. Best at 1+ year (captures annual seasonality). The engine's phased ramp-up already handles this — Prophet weight starts low and increases with data.

**Dependency:** `prophet` package (depends on cmdstanpy). Install via pip. ~200MB.

#### 3b. Bayesian Occupancy Fusion

**Problem:** Current occupancy tracking is binary (people_home count from device_tracker). Real occupancy is probabilistic and multi-signal.

**Solution:** Implement Bayesian occupancy estimation inspired by Area Occupancy Detection (220 stars). Combine motion, door, media, power, and climate sensors into per-area probability scores.

**Implementation:**
- New module: `analysis/occupancy.py` — `BayesianOccupancy` class
- Sensor fusion: motion sensors (high weight, fast decay), door sensors (medium weight), media players (low weight, slow decay), power draw (low weight), climate changes (low weight)
- Per-area probability with configurable sensor mapping
- Learned priors by day-of-week and hour-of-day (from historical snapshots)
- Output: probability 0-1 per area per time window
- Feed occupancy probability into feature vectors (replaces binary people_home)

**Complexity:** Medium — ~250 lines. The math is straightforward Bayesian updating.

**Key insight from research:** Area Occupancy Detection's "Wasp in Box" algorithm — for single-entry rooms, if motion stopped and door hasn't opened, person is still there. Simple but effective.

**Prerequisite sensors:** Motion sensors, door sensors. Works with whatever you have — degrades gracefully with fewer inputs.

#### 3c. Power Signal Correlation (Appliance Fingerprinting)

**Problem:** Smart plug wattage data is treated as a single number (total_watts). Individual appliance behavior patterns are invisible.

**Solution:** Inspired by HA WashData (572 stars). Learn power consumption profiles for individual appliances from smart plug data. Detect cycle start/stop, match against learned profiles, track appliance health.

**Implementation:**
- New module: `analysis/power_profiles.py` — `ApplianceProfiler` class
- Signal processing: sliding window on power time series from individual smart plugs
- Profile learning: NumPy correlation scoring against stored reference profiles
- Cycle detection: threshold-based start/stop with configurable parameters
- Health tracking: detect degradation in cycle duration, peak power, or profile shape
- Output: appliance state timeline + health scores fed into snapshot data

**Complexity:** Medium — ~300 lines. NumPy for signal processing (already a dependency).

**Data requirement:** Needs intraday snapshots with per-outlet power data. Current 4h intraday resolution may need to increase to 1h or 15min for appliance detection.

**Note:** Only useful if smart plugs report individual wattage. Check which entities report power draw.

---

### Phase 4: Advanced Capabilities

**Goal:** Add sequence-aware ML and integrate with the real-time Hub.
**Prerequisite:** Phase 3 complete, 8+ weeks of data
**Data requirement:** 8+ weeks for LSTM-AE training

#### 4a. LSTM-Autoencoder for Sequence Anomalies

**Problem:** IsolationForest catches point anomalies (single snapshot is unusual). It cannot catch sequence anomalies — unusual *ordering* of events (e.g., front door opens at 3am, motion in office, no lights turned on).

**Solution:** Train an LSTM-Autoencoder on normal daily event sequences. High reconstruction error = anomalous sequence.

**Implementation:**
- Add `torch` or `tensorflow-lite` to requirements (prefer torch for simplicity)
- New model: `models/lstm_autoencoder.py` — registered via ModelRegistry
- Input: hourly event-type sequences from intraday snapshots (e.g., [motion_kitchen, light_on_kitchen, motion_living, ...])
- Training: encode normal sequences, learn to reconstruct them
- Inference: high reconstruction error = anomalous sequence pattern
- Threshold: adaptive based on training distribution (e.g., 95th percentile)
- Alert integration: flag sequence anomalies in daily report + optional Telegram alert

**Complexity:** High — ~400 lines + new dependency. Needs careful sequence preprocessing.

**Trade-off:** PyTorch is ~2GB. For a lighter alternative, consider a simple Markov chain anomaly detector first (~100 lines, no new deps) that flags rare state transitions. Upgrade to LSTM-AE later if needed.

**Recommended approach:** Start with Markov chain (Phase 4a-lite), upgrade to LSTM-AE when data and need justify it.

#### 4b. Hub Integration (Real-Time + Batch)

**Problem:** ha-intelligence (batch, cron) and ha-intelligence-hub (real-time, WebSocket) are separate systems. Insights from one don't flow to the other.

**Solution:** Define a formal integration contract between the two systems. Hub feeds real-time activity data to the engine. Engine feeds predictions, baselines, and anomaly thresholds to the Hub.

**Implementation:**
- Shared data contract: JSON schema for predictions, baselines, anomaly thresholds
- Engine writes to well-known paths that Hub reads (already partially happening via `~/ha-logs/intelligence/`)
- Hub writes activity summaries that engine can consume for richer features
- Optional: Hub triggers engine retrain via subprocess when drift is detected in real-time
- Optional: Engine pushes prediction updates to Hub via file watch or simple IPC

**Complexity:** Medium — mostly contract definition and read/write coordination.

**Note:** This is where the hybrid architecture (snapshot + event-driven) becomes fully realized. The engine does the heavy thinking (training, prediction, scoring). The Hub does the real-time observation (activity detection, pattern matching, live anomaly flagging).

#### 4c. Future Capabilities (Backlog)

These emerged from research but don't have enough data/need to plan concretely yet:

| Capability | Technique | When to Consider |
|-----------|-----------|------------------|
| TCN for real-time event classification | Temporal Convolutional Network | When Hub needs sub-second event classification |
| HMM for activity chain prediction | Hidden Markov Model | When activity recognition is a priority |
| RL for HVAC optimization | SAC/PPO reinforcement learning | When smart thermostat is installed + simulation env exists |
| KAN for interpretable forecasting | Kolmogorov-Arnold Networks | When tooling matures (currently cutting-edge research) |
| Federated learning | Cross-household model sharing | When multiple HA instances exist |
| NILM energy disaggregation | Deep learning on mains power | When hardware (current transformers) is installed |
| Concept drift via FCA | Formal Concept Analysis | If weekly retrain + drift detection proves insufficient |

---

## Dependencies

### New Dependencies by Phase

| Phase | Package | Size | Purpose |
|-------|---------|------|---------|
| 1 | None | — | Pure refactor of existing code |
| 2 | None | — | Uses existing stdlib + sklearn |
| 3a | `prophet` (+ `cmdstanpy`) | ~200MB | Seasonal time series decomposition |
| 3b | None | — | Bayesian math with existing numpy |
| 3c | None | — | Signal processing with existing numpy |
| 4a | `torch` OR simple Markov chain | ~2GB or 0 | Sequence anomaly detection |
| 4b | None | — | File-based integration contract |

### Existing Dependencies (unchanged)
- `scikit-learn` 1.8.0
- `numpy` 2.4.2
- `holidays`
- Python 3.12 (system, NOT Homebrew 3.14)

---

## Data Maturity Timeline

| Day | Available Capability |
|-----|---------------------|
| 1-7 | Statistical baselines, simple anomaly thresholds |
| 7-14 | Correlations stabilize, intraday patterns visible |
| 14+ | sklearn models activate (GradientBoosting, RandomForest) |
| 14+ | Meta-learning starts, drift detection meaningful |
| 30+ | Prophet usable for weekly seasonality |
| 60+ | ML predictions weighted higher, IsolationForest well-calibrated |
| 90+ | LSTM-AE trainable (if implemented), full seasonal patterns visible |
| 365+ | Prophet captures annual seasonality, full behavioral model |

Current status (Feb 12, 2026): **Day 2.** Engine collecting data since Feb 11.

---

## Cron Pipeline (Updated)

### Current
```
*/15 * * * *    ha-log-sync
0 0 * * *       ha-log-sync --rotate
0 23 * * *      ha-intelligence --snapshot
30 23 * * *     ha-intelligence --full
0 0,4,8,12,16,20  ha-intelligence --snapshot-intraday
30 1 * * 1      ha-intelligence --retrain && ha-intelligence --meta-learn
```

### After Phase 2 (drift-based retraining)
```
*/15 * * * *    ha-log-sync
0 0 * * *       ha-log-sync --rotate
0 23 * * *      ha-intelligence --snapshot
30 23 * * *     ha-intelligence --full
0 0,4,8,12,16,20  ha-intelligence --snapshot-intraday
0 2 * * *       ha-intelligence --check-drift    # Daily drift check (replaces fixed Monday retrain)
30 1 * * 1      ha-intelligence --meta-learn      # Weekly meta-learning (kept — LLM analysis is always valuable)
```

### After Phase 2c (automation suggestions)
```
# Add weekly automation suggestion generation
0 3 * * 0       ha-intelligence --suggest-automations  # Sunday 3am
```

---

## Testing Strategy

### Phase 1 Test Structure
```
tests/
├── conftest.py              # Shared fixtures
│   ├── sample_states()      # Realistic HA state list (session scope)
│   ├── sample_snapshot()    # Complete snapshot (session scope)
│   ├── sample_30d()         # 30 days of varied snapshots (session scope)
│   ├── mock_ha_api()        # Monkeypatched API (function scope)
│   └── tmp_data_dir()       # Isolated PathConfig with tmp_path
├── test_collectors.py       # Each extractor produces expected keys/types
├── test_features.py         # Feature vectors correct shape/values
├── test_models.py           # Mock models, verify predict interface
├── test_predictions.py      # Blending weights, scoring accuracy
├── test_analysis.py         # Baselines, correlations, anomaly thresholds
├── test_storage.py          # DataStore round-trip (save → load)
├── test_llm.py              # Mocked Ollama, report generation
└── test_cli.py              # End-to-end CLI commands
```

### Test Types
| Type | What It Tests | Example |
|------|---------------|---------|
| Unit | Individual functions in isolation | `cyclical_encode(12, 24)` returns expected sin/cos |
| Data validation | Snapshot schema correctness | Snapshot has all required TypedDict keys |
| Behavioral/directional | Predictions make directional sense | More people → more lights predicted |
| Integration | Full pipeline | `--full` produces valid snapshot + predictions + report |
| Regression | Outputs don't change unexpectedly | Golden snapshot comparison |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Prophet dependency is large (~200MB) | Disk space, install complexity | Defer until Phase 3. Prophet is optional — engine works without it |
| PyTorch is very large (~2GB) | Disk, memory | Start with Markov chain (Phase 4a-lite, zero deps). Only add PyTorch if Markov chain proves insufficient |
| Modularization breaks cron pipeline | Data collection gap | Incremental migration. Keep monolith as reference. Verify each step against cron output |
| Not enough data for advanced ML | Models underfit | Phased ramp-up already handles this. Each technique has documented minimum data requirement |
| Over-engineering beyond actual need | Wasted effort | Each phase is independently valuable. Stop after any phase if value plateaus |
| Registry pattern adds indirection | Code harder to follow | Keep registry simple (no metaclasses, no magic). Each collector/model is a plain class with one method |

---

## Success Criteria

### Phase 1
- [ ] All 78 existing tests pass against modularized code
- [ ] `--full` pipeline produces identical output
- [ ] Cron commands work without modification
- [ ] Adding a new collector requires only: write class, add decorator
- [ ] Adding a new ML model requires only: write class, add decorator

### Phase 2
- [ ] Drift detection triggers retrain within 24h of behavioral shift
- [ ] Entity correlation matrix identifies top 10 strongest co-occurrence patterns
- [ ] LLM generates valid HA automation YAML suggestions
- [ ] Unnecessary weekly retrains eliminated (retrain only on drift)

### Phase 3
- [ ] Prophet captures weekly energy/climate seasonality
- [ ] Bayesian occupancy outperforms binary people_home in prediction accuracy
- [ ] At least one appliance power profile learned and tracked

### Phase 4
- [ ] Sequence anomaly detection catches at least one event-order anomaly per month
- [ ] Hub and engine share a documented integration contract
- [ ] Real-time anomaly flags from Hub match batch anomaly detections from engine

---

## References

### HA ML Projects
- [EMHASS](https://github.com/davidusb-geek/emhass) — Energy optimization (sklearn + LP)
- [Frigate](https://github.com/blakeblackshear/frigate) — NVR with real-time object detection
- [Home-LLM](https://github.com/acon96/home-llm) — Local fine-tuned LLMs for HA control
- [Better Thermostat](https://github.com/KartoffelToby/better_thermostat) — Smart TRV controller
- [Versatile Thermostat](https://github.com/jmcollin78/versatile_thermostat) — Auto-TPI thermostat
- [AI Automation Suggester](https://github.com/ITSpecialist111/ai_automation_suggester) — LLM-based automation suggestions
- [HA WashData](https://github.com/3dg1luk43/ha_washdata) — Appliance cycle detection from power
- [Area Occupancy Detection](https://github.com/Hankanman/Area-Occupancy-Detection) — Bayesian occupancy
- [Solar Forecast ML](https://github.com/Zara-Toorox/ha-solar-forecast-ml) — Self-learning solar prediction
- [TheSillyHome](https://github.com/lcmchris/thesillyhome-container) — General ML for HA (inactive)
- [HA-MCP](https://github.com/homeassistant-ai/ha-mcp) — MCP server for LLM-HA integration

### ML Techniques
- [Personalized Smart Home Automation Using ML (MDPI Sensors 2025)](https://www.mdpi.com/1424-8220/25/19/6082)
- [Edge AI for Real-Time Anomaly Detection (MDPI Future Internet 2025)](https://www.mdpi.com/1999-5903/17/4/179)
- [Occupancy Prediction in IoT-Enabled Smart Buildings (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174554/)
- [Dynamic Appliance Scheduling Using Adaptive RL (Nature)](https://www.nature.com/articles/s41598-025-08125-9)
- [Smart Home Energy Prediction with TKAT (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0378778825002592)
- [LSTM-Autoencoder Anomaly Detection (Journal of Big Data)](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00746-z)
- [KANs for Time Series (arXiv)](https://arxiv.org/abs/2405.08790)
- [Prophet: Forecasting at Scale](https://facebook.github.io/prophet/)
- [CASAS Smart Home Datasets](https://casas.wsu.edu/datasets/)
- [Incremental Learning via FCA for Smart Environments (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1574119219302081)

### Architecture
- [Model Registry Pattern](https://www.abhik.ai/articles/registry-pattern)
- [Made With ML: Testing ML Systems](https://madewithml.com/courses/mlops/testing/)
- [Kedro: Modular Data Science Pipelines](https://kedro.org/)
- [sktime: Modular Time Series Forecasting](https://www.sktime.net/en/latest/)
- [EMHASS ML Forecaster Architecture](https://emhass.readthedocs.io/en/latest/mlforecaster.html)
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Home Assistant Statistics Component](https://github.com/home-assistant/core/blob/dev/homeassistant/components/statistics/sensor.py)
- [Home Assistant AI Blog (Sep 2025)](https://www.home-assistant.io/blog/2025/09/11/ai-in-home-assistant/)
