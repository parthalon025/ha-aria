# ha-intelligence

Home Assistant intelligence engine with ML-based prediction and anomaly detection.

## Structure

```
ha_intelligence/              # Python package (8 modules)
  __init__.py
  config.py                   # 7 dataclass configs + AppConfig composer
  cli.py                      # CLI entry point — wires all modules together
  storage/
    data_store.py             # Unified JSON I/O layer
  collectors/
    registry.py               # Decorator-based collector registry
    extractors.py             # 15 registered HA entity collectors
    ha_api.py                 # HA REST API + weather + calendar
    logbook.py                # Logbook event summarization
    snapshot.py               # Snapshot construction + intraday aggregation
  analysis/
    baselines.py              # Day-of-week baseline computation
    anomalies.py              # Statistical anomaly detection
    correlations.py           # Cross-correlation (Pearson r)
    drift.py                  # Concept drift detection + retrain triggers
    entity_correlations.py    # Entity co-occurrence + conditional probability
    occupancy.py              # Bayesian multi-sensor occupancy estimation
    power_profiles.py         # Appliance power profiling + cycle detection
    reliability.py            # Device reliability scoring
    sequence_anomalies.py     # Markov chain sequence anomaly detection
  features/
    time_features.py          # Cyclical time encoding (sin/cos)
    feature_config.py         # ML feature toggle config
    vector_builder.py         # Feature vector construction + training data
  models/
    registry.py               # Model registry pattern
    gradient_boosting.py      # GradientBoosting regressor
    isolation_forest.py       # IsolationForest anomaly detector
    device_failure.py         # RandomForest device failure predictor
    prophet_forecaster.py     # Prophet seasonal time series forecaster
    training.py               # Training orchestration + blending
  predictions/
    predictor.py              # Prediction generation (statistical + ML blend)
    scoring.py                # Prediction scoring + accuracy tracking
  llm/
    client.py                 # Ollama API client + think tag stripping
    reports.py                # LLM insight report generation
    meta_learning.py          # Self-improving feature config via LLM
    automation_suggestions.py # LLM-powered HA automation YAML generation
bin/
  ha-intelligence             # Thin shim (adds package to sys.path, calls cli.main())
  ha-intelligence-v2-monolith # Backup of original 2,373-line monolith
  ha-log-sync                 # HA logbook API sync to local JSON
tests/                        # 177 tests across 15 files + conftest
  conftest.py                 # Shared fixtures, sample states, helpers
  test_collectors.py          # 16 tests (snapshot, extraction, weather, logbook)
  test_analysis.py            # 9 tests (baselines, correlations, reliability, anomalies)
  test_features.py            # 14 tests (time encoding, feature vectors, config)
  test_models.py              # 11 tests (sklearn training, blending, device failure)
  test_predictions.py         # 10 tests (predictions, scoring, ML-enhanced)
  test_llm.py                 # 15 tests (think tags, meta-learning, suggestions)
  test_storage.py             # 3 tests (intraday save/load, aggregation)
  test_drift.py               # 8 tests (drift detection, retrain triggers)
  test_entity_correlations.py # 14 tests (co-occurrence, conditional prob, hourly)
  test_automation_suggestions.py # 13 tests (YAML validation, LLM parsing)
  test_prophet.py             # 11 tests (Prophet train, predict, registry)
  test_occupancy.py           # 14 tests (Bayesian fusion, priors, features)
  test_power_profiles.py      # 15 tests (cycle detection, profiling, health)
  test_sequence_anomalies.py  # 11 tests (Markov chain training, detection, summary)
  test_cli.py                 # 9 tests (train-sequences, sequence-anomalies, dispatch)
docs/                         # Design documents
```

## Key Details

- **HA instance**: HAOS on Raspberry Pi at 192.168.1.35 (3,065 entities)
- **Data directory**: `~/ha-logs/intelligence/` (daily/, intraday/, models/, meta-learning/)
- **ML models**: GradientBoosting, IsolationForest, RandomForest (scikit-learn 1.8.0), Prophet (seasonal forecasting)
- **LLM**: deepseek-r1:8b via Ollama for reports and meta-learning (`<think>` tags auto-stripped)
- **Env vars**: HA_URL, HA_TOKEN from `~/.env`
- **15 collectors**: power, occupancy, climate, lights, locks, automations, motion, ev, entities_summary, doors_windows, batteries, network, media, sun, vacuum
- **Data outputs**: `entity_correlations.json`, `sequence_anomalies.json`, `models/sequence_model.json`, `insights/automation-suggestions/` (YAML), `insights/power-profiles.json`
- **Dependencies**: scikit-learn, numpy, prophet (+ cmdstanpy, pandas, matplotlib)

## CLI Commands

| Command | Description |
|---------|-------------|
| `--snapshot` | Collect current HA state snapshot |
| `--score` | Score yesterday's predictions against actuals |
| `--full` | Full daily pipeline (snapshot → predict → report) |
| `--retrain` | Retrain ML models from accumulated data |
| `--meta-learn` | LLM meta-learning to tune feature config |
| `--check-drift` | Detect concept drift; auto-triggers retrain if needed |
| `--entity-correlations` | Compute entity co-occurrence from logbook data |
| `--suggest-automations` | LLM-generated HA automation YAML from learned patterns |
| `--train-prophet` | Train Prophet seasonal forecasters (needs 14+ days) |
| `--occupancy` | Bayesian occupancy estimation from current snapshot |
| `--power-profiles` | Analyze per-outlet power consumption profiles |
| `--train-sequences` | Train Markov chain model from logbook event sequences |
| `--sequence-anomalies` | Detect anomalous event sequences using trained model |

## Running Tests

```bash
python3 -m pytest tests/ -v
```

**Important:** Use `python3` (Homebrew 3.14 on PATH), NOT `/usr/bin/python3` (system 3.12). numpy/sklearn are installed under Homebrew Python.

## Architecture Patterns

- **Registry pattern**: Collectors and models self-register via decorators (`@CollectorRegistry.register("name")`)
- **Dataclass configs**: `AppConfig` composed of 7 sub-configs (HAConfig, PathConfig, ModelConfig, OllamaConfig, WeatherConfig, SafetyConfig, HolidayConfig)
- **DataStore abstraction**: All JSON I/O goes through `DataStore(paths)` — no direct file access in business logic
- **Dependency injection**: Tests use `PathConfig(data_dir=tmpdir)` + `DataStore(paths)` — no monkey-patching globals
- **Lazy imports in CLI**: Each `cmd_*` function imports what it needs to avoid circular dependencies

## Gotchas

- HAOS doesn't write `home-assistant.log` — uses `/api/logbook` REST endpoint
- History API requires `filter_entity_id` — can't wildcard by domain
- 8,718 of ~11,000 daily logbook events are clock sensor updates — filtered via `CLOCK_SENSORS` set
- Python 3.12 deprecates `datetime.utcnow()` — use `datetime.now(tz=datetime.timezone.utc)`
- `bin/ha-intelligence` uses `os.path.realpath(__file__)` (not abspath) — required for symlink resolution from `~/.local/bin/`
- Mock patches in tests must target the *import site* (e.g., `ha_intelligence.collectors.snapshot.fetch_weather`), not the *definition site* (`ha_intelligence.collectors.ha_api.fetch_weather`)
- Conditional probabilities in entity correlations are capped at 1.0 — sliding window can inflate counts when multiple A events co-occur with the same B event
- `--suggest-automations` requires `--entity-correlations` to have run first (needs correlation data)
- DriftDetector needs >=5 days of accuracy history to produce meaningful results
- Prophet needs 14+ days of daily snapshots; yearly seasonality only activates at 60+ days
- Prophet imports trigger "Importing plotly failed" warning — harmless, plotly is optional for interactive plots
- BayesianOccupancy works without door sensors but loses "Wasp in Box" capability (no door sensors in current HA setup)
- Power profiles need intraday data with varying outlet usage to detect cycles; always-on outlets (NVR) show 0 cycles
- `--power-profiles` saves to `insights/power-profiles.json`; profiles improve as intraday snapshot count grows
- `--train-sequences` needs logbook data from `ha-log-sync`; model quality improves with more days of data
- `--sequence-anomalies` requires `--train-sequences` to have run first (needs trained Markov model)
- Markov chain uses same `_is_trackable()` filter as entity correlations — entities outside tracked domains are excluded
- Unknown entities at detection time get minimum probability (1/total_transitions) rather than being silently skipped
