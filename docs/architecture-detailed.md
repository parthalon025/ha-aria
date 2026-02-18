# ARIA Architecture — Detailed Reference

Reference doc for CLAUDE.md.

## Package Layout

```
aria/
├── cli.py                  # Unified CLI entry point
├── capabilities.py         # Capability dataclass, CapabilityRegistry, validation engine
├── hub/                    # Real-time hub core
│   ├── core.py             # IntelligenceHub — module registry, task scheduler, event bus
│   ├── cache.py            # SQLite-backed cache (hub.db) with category-based get/set
│   ├── audit_logger.py     # AuditLogger — dedicated audit.db, async write-behind buffer, tamper-evident checksums
│   ├── constants.py        # Shared cache key constants
│   ├── api.py              # FastAPI routes + WebSocket server
│   ├── validation_runner.py # Subprocess pytest runner for on-demand validation
│   └── config_defaults.py  # Default config parameter seeding
├── modules/                # Hub modules (registered in order)
│   ├── discovery.py        # HA entity/device/area scanning via REST + WebSocket
│   ├── ml_engine.py        # Feature engineering, model training, periodic retraining
│   ├── patterns.py         # Recurring event sequence detection from logbook
│   ├── orchestrator.py     # Automation suggestions from detected patterns
│   ├── shadow_engine.py    # Predict-compare-score loop for shadow mode
│   ├── data_quality.py     # Entity classification pipeline (auto-exclude, edge, include)
│   ├── organic_discovery/  # HDBSCAN-based capability discovery (domain + behavioral clustering)
│   │   ├── module.py       # OrganicDiscoveryModule — full pipeline orchestration
│   │   ├── feature_vectors.py  # Entity attribute → numeric feature matrix
│   │   ├── clustering.py   # HDBSCAN clustering with silhouette scoring
│   │   ├── behavioral.py   # Layer 2: co-occurrence matrix + temporal patterns
│   │   ├── scoring.py      # 5-component weighted usefulness score (0-100)
│   │   ├── naming.py       # Heuristic + Ollama LLM naming backends
│   │   └── seed_validation.py  # Jaccard similarity validation against seed capabilities
│   ├── intelligence.py     # Unified cache assembly (snapshots, baselines, predictions, ML)
│   ├── activity_monitor.py # WebSocket state_changed listener, 15-min windows, analytics
│   └── presence.py         # Frigate MQTT + HA sensor presence tracking, BayesianOccupancy
├── engine/                 # Batch ML engine (formerly ha-intelligence)
│   ├── cli.py              # Engine CLI (called by aria CLI dispatcher)
│   ├── config.py           # Engine configuration
│   ├── collectors/         # HA API, logbook, snapshot, registry data collection
│   ├── features/           # Feature engineering (time features, vector builder, config)
│   ├── models/             # ML models (GradientBoosting, RandomForest, IsolationForest, Prophet, etc.)
│   ├── predictions/        # Prediction generation and scoring
│   ├── analysis/           # Baselines, correlations, drift, anomalies, power profiles, occupancy
│   ├── llm/                # LLM integration (meta-learning, automation suggestions, reports)
│   └── storage/            # Data store and model I/O
└── dashboard/              # Preact SPA (Jinja legacy removed)
    └── spa/                # Active SPA (esbuild-bundled Preact + Tailwind)
```

## Legacy Entry Points

| File | Status |
|------|--------|
| `bin/ha-hub.py` | Legacy wrapper — calls `aria serve` internally |
| `bin/discover.py` | Standalone discovery CLI (also used as subprocess) |
| `bin/ha-log-sync` | Log sync script (called by `aria sync-logs`) |

## Hub Modules (12, registered in order)

| Module | File | Purpose |
|--------|------|---------|
| `discovery` | `aria/modules/discovery.py` | Scans HA (REST + WebSocket), detects capabilities, caches entities/devices/areas |
| `ml_engine` | `aria/modules/ml_engine.py` | Model training (GradientBoosting, RandomForest, LightGBM), periodic retraining. Feature extraction delegates to engine's `vector_builder` (single source of truth). Snapshot validation applied before training. Stale training check on startup triggers retraining if >7 days since last train. |
| `pattern_recognition` | `aria/modules/pattern_recognition.py` | Trajectory classification (DTW + heuristic), pattern scale tagging (micro/meso/macro), anomaly explanation. Subscribes to shadow_resolved events. Tier 3+ only. |
| `patterns` | `aria/modules/patterns.py` | Detects recurring event sequences from logbook data |
| `orchestrator` | `aria/modules/orchestrator.py` | Generates automation suggestions from detected patterns |
| `shadow_engine` | `aria/modules/shadow_engine.py` | Predict-compare-score loop: captures context on state_changed, generates predictions (next_domain, room_activation, routine_trigger), scores against reality |
| `data_quality` | `aria/modules/data_quality.py` | Entity classification pipeline — auto-exclude (domain, stale, noise, vehicle, unavailable grace period), edge cases, default include. Reads discovery cache, writes to `entity_curation` table. Runs on startup and daily. |
| `organic_discovery` | `aria/modules/organic_discovery/module.py` | Two-layer HDBSCAN capability discovery: Layer 1 clusters entities by attributes, Layer 2 clusters by temporal co-occurrence. Usefulness scoring, seed validation, autonomy modes, heuristic/Ollama naming. Weekly via systemd timer. |
| `intelligence` | `aria/modules/intelligence.py` | Assembles daily/intraday snapshots, baselines, predictions, ML scores into unified cache. Reads engine outputs (entity correlations, sequence anomalies, power profiles, automation suggestions). Sends Telegram digest on new insights. |
| `activity_monitor` | `aria/modules/activity_monitor.py` | WebSocket listener for state_changed events, 15-min windowed activity log, adaptive snapshot triggering, prediction analytics. Emits filtered events to hub event bus for shadow engine. |
| `presence` | `aria/modules/presence.py` | Subscribes to Frigate MQTT (person/face detection) and HA WebSocket (motion sensors, lights, dimmers, device trackers, door sensors). Feeds signals into BayesianOccupancy for per-room presence estimation. |
| `audit_logger` | `aria/hub/audit_logger.py` | Dedicated `audit.db` separate from the cache DB. Async write-behind buffer minimizes latency on hot paths. Tamper-evident SHA-256 checksums on each event batch. Streams events to WebSocket subscribers via `/ws/audit`. Bridges to watchdog for alerting on high-severity events. |

Each module declares its capabilities via a `CAPABILITIES` class attribute (see `aria/capabilities.py` for the `Capability` dataclass). The `CapabilityRegistry.collect_from_modules()` method harvests all declarations for validation and CLI/API exposure.

### Engine Standalone Modules (Phase 3)

| Module | Path | Purpose |
|--------|------|---------|
| `anomaly_explainer` | `aria/engine/anomaly_explainer.py` | IsolationForest decision path tracing — identifies top-N features contributing to anomaly detection |
| `pattern_scale` | `aria/engine/pattern_scale.py` | StrEnum for micro (<5min), meso (5min-4h), macro (4h+) time-scale classification |
| `sequence` | `aria/engine/sequence.py` | DTW trajectory classifier (tslearn KNeighborsTimeSeriesClassifier) with heuristic fallback |

## Engine Subpackages (8)

| Package | Path | Purpose |
|---------|------|---------|
| `collectors` | `aria/engine/collectors/` | HA API, logbook, snapshot, registry data collection, presence (reads hub cache) |
| `features` | `aria/engine/features/` | Time features, vector builder (single source of truth for feature extraction — used by both engine and hub ML training), feature config (4 presence features + pattern_features) |
| `validation` | `aria/engine/validation.py` | Snapshot validation layer (MIN_ENTITY_COUNT, MAX_UNAVAILABLE_RATIO checks) |
| `models` | `aria/engine/models/` | ML models: GradientBoosting, RandomForest, IsolationForest, Prophet, device failure, registry |
| `predictions` | `aria/engine/predictions/` | Prediction generation and scoring |
| `analysis` | `aria/engine/analysis/` | Baselines, correlations, drift, anomalies, occupancy, power profiles, reliability |
| `llm` | `aria/engine/llm/` | Ollama/LLM integration: meta-learning, automation suggestions, reports |
| `storage` | `aria/engine/storage/` | Data store and model I/O |

## Cache Categories (8) + Shadow Tables (2)

**Category-based:** `activity_log`, `activity_summary`, `areas`, `capabilities`, `devices`, `discovery_metadata`, `entities`, `intelligence`, `presence`
**Shadow tables:** `predictions` (predict-compare-score records), `pipeline_state` (backtest→shadow→suggest→autonomous progression)
**Phase 2 tables:** `config` (editable engine parameters), `entity_curation` (tiered entity classification), `config_history` (change audit log)
**Organic discovery keys:** `discovery_history` (run history), `discovery_settings` (autonomy mode, naming backend, thresholds)
**ML data keys in intelligence cache:** `drift_status`, `feature_selection`, `reference_model`, `incremental_training`, `forecaster_backend`, `anomaly_alerts`, `autoencoder_status`, `isolation_forest_status`, `shap_attributions`
