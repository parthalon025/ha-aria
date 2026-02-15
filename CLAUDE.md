# ARIA — Adaptive Residence Intelligence Architecture

Unified intelligence platform for Home Assistant — batch ML engine, real-time activity monitoring, predictive analytics, and an interactive Preact dashboard in a single `aria.*` package.

**Repo:** https://github.com/parthalon025/ha-aria (private)

## Context

**Unified from:** `ha-intelligence` (batch ML engine) + `ha-intelligence-hub` (real-time dashboard), now in one repo.
**Design doc:** `~/Documents/docs/plans/2026-02-11-ha-intelligence-hub-design.md`
**Lean roadmap:** `~/Documents/docs/plans/2026-02-11-ha-hub-lean-roadmap.md`
**Activity monitor plan:** `~/.claude/plans/resilient-stargazing-catmull.md`
**Shadow mode design:** `~/Documents/docs/plans/2026-02-12-ha-hub-shadow-mode-design.md`
**Organic discovery design:** `docs/plans/2026-02-14-organic-capability-discovery-design.md`

## Running

**Service:** `aria-hub.service` (user systemd)
**CLI:** `aria` (installed via `pip install -e .`, see `pyproject.toml`)
**API:** `http://127.0.0.1:8001` (localhost only)
**Dashboard:** `http://127.0.0.1:8001/ui/`
**WebSocket:** `ws://127.0.0.1:8001/ws` (real-time cache updates to dashboard)
**Env vars:** HA_URL, HA_TOKEN from `~/.env` (bash wrapper pattern, not EnvironmentFile=)

```bash
# Start hub (preferred)
aria serve

# Start hub on custom port
aria serve --port 8002

# Restart systemd service
systemctl --user restart aria-hub

# Logs
journalctl --user -u aria-hub -f

# Cache inspection
curl -s http://127.0.0.1:8001/api/cache | python3 -m json.tool
curl -s http://127.0.0.1:8001/api/cache/activity_summary | /usr/bin/python3 -m json.tool
```

### ARIA CLI Commands

All commands route through the unified `aria` entry point (`aria/cli.py`).

| Command | What it does |
|---------|-------------|
| `aria serve` | Start real-time hub + dashboard (replaces `bin/ha-hub.py`) |
| `aria full` | Full daily pipeline: snapshot → predict → report |
| `aria snapshot` | Collect current HA state snapshot |
| `aria predict` | Generate predictions from latest snapshot |
| `aria score` | Score yesterday's predictions against actuals |
| `aria retrain` | Retrain ML models from accumulated data |
| `aria meta-learn` | LLM meta-learning to tune feature config |
| `aria check-drift` | Detect concept drift in predictions |
| `aria correlations` | Compute entity co-occurrence correlations |
| `aria suggest-automations` | Generate HA automation YAML via LLM |
| `aria prophet` | Train Prophet seasonal forecasters |
| `aria occupancy` | Bayesian occupancy estimation |
| `aria power-profiles` | Analyze per-outlet power consumption |
| `aria sequences train` | Train Markov chain model from logbook sequences |
| `aria sequences detect` | Detect anomalous event sequences |
| `aria snapshot-intraday` | Collect intraday snapshot (used internally by hub) |
| `aria sync-logs` | Sync HA logbook to local JSON |
| `aria discover-organic` | Run organic capability discovery (Layer 1 + Layer 2) |
| `aria capabilities list` | List all registered capabilities (--layer, --status, --verbose) |
| `aria capabilities verify` | Validate all capabilities against tests/config/deps |
| `aria capabilities export` | Export capability registry as JSON |

Engine commands delegate to `aria.engine.cli` with old-style flags internally.

## Architecture

### Package Layout

```
aria/
├── cli.py                  # Unified CLI entry point
├── capabilities.py         # Capability dataclass, CapabilityRegistry, validation engine
├── hub/                    # Real-time hub core
│   ├── core.py             # IntelligenceHub — module registry, task scheduler, event bus
│   ├── cache.py            # SQLite-backed cache (hub.db) with category-based get/set
│   ├── constants.py        # Shared cache key constants
│   ├── api.py              # FastAPI routes + WebSocket server
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
│   └── activity_monitor.py # WebSocket state_changed listener, 15-min windows, analytics
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

### Legacy Entry Points

| File | Status |
|------|--------|
| `bin/ha-hub.py` | Legacy wrapper — calls `aria serve` internally |
| `bin/discover.py` | Standalone discovery CLI (also used as subprocess) |
| `bin/ha-log-sync` | Log sync script (called by `aria sync-logs`) |

### Hub Modules (9, registered in order)

| Module | File | Purpose |
|--------|------|---------|
| `discovery` | `aria/modules/discovery.py` | Scans HA (REST + WebSocket), detects capabilities, caches entities/devices/areas |
| `ml_engine` | `aria/modules/ml_engine.py` | Feature engineering, model training (GradientBoosting, RandomForest, LightGBM), periodic retraining |
| `pattern_recognition` | `aria/modules/patterns.py` | Detects recurring event sequences from logbook data |
| `orchestrator` | `aria/modules/orchestrator.py` | Generates automation suggestions from detected patterns |
| `shadow_engine` | `aria/modules/shadow_engine.py` | Predict-compare-score loop: captures context on state_changed, generates predictions (next_domain, room_activation, routine_trigger), scores against reality |
| `data_quality` | `aria/modules/data_quality.py` | Entity classification pipeline — auto-exclude (domain, stale, noise, vehicle, unavailable grace period), edge cases, default include. Reads discovery cache, writes to `entity_curation` table. Runs on startup and daily. |
| `organic_discovery` | `aria/modules/organic_discovery/module.py` | Two-layer HDBSCAN capability discovery: Layer 1 clusters entities by attributes, Layer 2 clusters by temporal co-occurrence. Usefulness scoring, seed validation, autonomy modes, heuristic/Ollama naming. Weekly via systemd timer. |
| `intelligence` | `aria/modules/intelligence.py` | Assembles daily/intraday snapshots, baselines, predictions, ML scores into unified cache. Reads engine outputs (entity correlations, sequence anomalies, power profiles, automation suggestions). Sends Telegram digest on new insights. |
| `activity_monitor` | `aria/modules/activity_monitor.py` | WebSocket listener for state_changed events, 15-min windowed activity log, adaptive snapshot triggering, prediction analytics. Emits filtered events to hub event bus for shadow engine. |

Each module declares its capabilities via a `CAPABILITIES` class attribute (see `aria/capabilities.py` for the `Capability` dataclass). The `CapabilityRegistry.collect_from_modules()` method harvests all declarations for validation and CLI/API exposure.

### Engine Subpackages (7)

| Package | Path | Purpose |
|---------|------|---------|
| `collectors` | `aria/engine/collectors/` | HA API, logbook, snapshot, registry data collection |
| `features` | `aria/engine/features/` | Time features, vector builder, feature config |
| `models` | `aria/engine/models/` | ML models: GradientBoosting, RandomForest, IsolationForest, Prophet, device failure, registry |
| `predictions` | `aria/engine/predictions/` | Prediction generation and scoring |
| `analysis` | `aria/engine/analysis/` | Baselines, correlations, drift, anomalies, occupancy, power profiles, reliability |
| `llm` | `aria/engine/llm/` | Ollama/LLM integration: meta-learning, automation suggestions, reports |
| `storage` | `aria/engine/storage/` | Data store and model I/O |

### Cache Categories (8) + Shadow Tables (2)

**Category-based:** `activity_log`, `activity_summary`, `areas`, `capabilities`, `devices`, `discovery_metadata`, `entities`, `intelligence`
**Shadow tables:** `predictions` (predict-compare-score records), `pipeline_state` (backtest→shadow→suggest→autonomous progression)
**Phase 2 tables:** `config` (editable engine parameters), `entity_curation` (tiered entity classification), `config_history` (change audit log)
**Organic discovery keys:** `discovery_history` (run history), `discovery_settings` (autonomy mode, naming backend, thresholds)
**ML data keys in intelligence cache:** `drift_status`, `feature_selection`, `reference_model`, `incremental_training`, `forecaster_backend`, `anomaly_alerts`, `autoencoder_status`, `isolation_forest_status`, `shap_attributions`

### Dashboard (Preact SPA)

**Stack:** Preact 10 + @preact/signals + Tailwind CSS v4 + uPlot, bundled with esbuild
**Location:** `aria/dashboard/spa/`
**Design language:** `docs/design-language.md` — MUST READ before creating or modifying UI components
**Full component reference:** `docs/dashboard-components.md`
**Pages (13):** Home, Discovery, Capabilities, Data Curation, Intelligence, Predictions, Patterns, Shadow Mode, ML Engine, Automations, Settings, Guide

#### Build & CSS

```bash
# Rebuild SPA after JSX changes (REQUIRED — dist/bundle.js is gitignored)
cd aria/dashboard/spa && npm run build
```

**CSS rules:**
- All colors via CSS custom properties in `index.css` — NEVER hardcode hex values in JSX
- Use `.t-frame` with `data-label` for content cards (NOT `.t-card` — legacy)
- Use `class` attribute (Preact), NOT `className`
- Tailwind via pre-built `bundle.css` — arbitrary values may not exist. Use inline `style` for non-standard values.
- uPlot renders on `<canvas>` — CSS variables must be resolved via `getComputedStyle()` before passing to uPlot

### Activity Monitor — Prediction Analytics

Four analytical methods computed on each 15-min flush and cached in `activity_summary`:

| Method | What it does | Cold-start requirement |
|--------|-------------|----------------------|
| `_event_sequence_prediction` | Frequency-based next-domain model from 5-domain n-grams | 6+ events in window history |
| `_detect_activity_patterns` | Frequent 3-domain trigrams (3+ occurrences in 24h) | 3+ recurring sequences |
| `_predict_next_arrival` | Day-of-week historical occupancy transition averages | Occupancy transitions in history |
| `_detect_activity_anomalies` | Current event rate vs hourly historical average (flags >2x) | Hourly average data |

## Testing

### Pipeline Verification (after deployment or feature changes)

ARIA has the deepest pipeline — engine→JSON files→hub cache→API→WebSocket→dashboard. Unit tests cover each layer but not the flow between them. After any deployment or feature change, run dual-axis tests:

**Horizontal:** Hit every API endpoint (`/api/cache/{category}` for all 8 categories, `/api/shadow/accuracy`, `/api/pipeline`, `/api/ml/*`, `/api/capabilities/*`, `/api/config`, `/api/curation/summary`). Confirm each returns expected shape with real data.

**Vertical:** Trigger one engine command (e.g., `aria snapshot-intraday`), then trace:
```
aria snapshot-intraday →
  JSON file written to ~/ha-logs/intelligence/ →
    hub intelligence module reads it into cache →
      GET /api/cache/intelligence returns new data →
        WebSocket pushes update →
          Dashboard renders updated values
```

See: `~/Documents/docs/lessons/2026-02-15-horizontal-vertical-pipeline-testing.md`

### Unit Tests

**Memory warning:** The full suite (~984 tests) can consume 4-8G RAM. If concurrent agents or services are running, check `free -h` first. If available memory < 4G, run by suite instead of the full set. Shadow engine tests previously consumed 17G+ RAM due to mock objects returning None in tight loops — those are fixed, but watch for regressions.

```bash
# All tests (~984) — use timeout to catch hangs
.venv/bin/python -m pytest tests/ -v --timeout=120

# By suite (safer when memory-constrained)
.venv/bin/python -m pytest tests/hub/ -v         # Hub (~670 tests)
.venv/bin/python -m pytest tests/engine/ -v       # Engine (222 tests)
.venv/bin/python -m pytest tests/integration/ -v  # Integration (39 tests)

# By feature area (use -k for keyword filtering)
.venv/bin/python -m pytest tests/hub/ -k "organic" -v       # Organic discovery (148 tests)
.venv/bin/python -m pytest tests/hub/ -k "shadow" -v        # Shadow mode
.venv/bin/python -m pytest tests/hub/ -k "activity" -v      # Activity monitor
.venv/bin/python -m pytest tests/hub/ -k "data_quality" -v  # Data quality/curation
```

## Environment

- **HA instance:** 192.168.1.35:8123 (HAOS on Raspberry Pi, 3,065 entities, 10 capabilities)
- **Env vars:** HA_URL, HA_TOKEN from `~/.env`
- **Python:** 3.12 via `.venv/` (aiohttp, scikit-learn, numpy, uvicorn, fastapi)
- **Package config:** `pyproject.toml` (replaces old `requirements.txt`)
- **Node:** esbuild for SPA bundling (dev dependency only)
- **Cache DB:** `~/ha-logs/intelligence/cache/hub.db` (SQLite)
- **Snapshot log:** `~/ha-logs/intelligence/snapshot_log.jsonl` (append-only JSONL)

## WebSocket Connections

The hub maintains **two separate WebSocket connections** to HA:

1. **Discovery** (`aria/modules/discovery.py`) — listens for `entity_registry/list`, `device_registry/list`, `area_registry/list` (low-frequency registry events)
2. **Activity** (`aria/modules/activity_monitor.py`) — listens for `state_changed` (high-volume, ~22K events/day)

Separated by design: state_changed volume would drown registry events. Each has independent backoff (5s→60s max).

## Entity Filtering (Activity Monitor)

**Phase 2 curation layer:** Entity-level include/exclude from `entity_curation` SQLite table. Loaded on startup, reloaded dynamically via `curation_updated` event bus. Falls back to domain filtering when curation table is empty (first boot).

**Domain fallback (used when curation not loaded):**
**Tracked:** light, switch, binary_sensor, lock, media_player, cover, climate, vacuum, person, device_tracker, fan
**Conditional:** sensor (only if device_class == "power")
**Excluded:** update, tts, stt, scene, button, number, select, input_*, counter, script, zone, sun, weather, conversation, event, automation, camera, image, remote

**Noise suppression:** unavailable↔unknown transitions, same-state-to-same-state

### API Reference

Full curl examples for all endpoints: `docs/api-reference.md`

Key endpoints: `/api/cache/{category}`, `/api/shadow/accuracy`, `/api/pipeline`, `/api/ml/*`, `/api/capabilities/*`, `/api/capabilities/registry`, `/api/capabilities/registry/{id}`, `/api/capabilities/registry/graph`, `/api/capabilities/registry/health`, `/api/config`, `/api/curation/summary`

## HA Data Model

HA uses a three-tier hierarchy: **entity → device → area**. Only ~0.2% of entities have a direct `area_id`. The rest inherit area through their parent device. Any feature touching area assignments must resolve through the device layer: check `entity.area_id` first, then fall back to `devices[entity.device_id].area_id`. The discovery pipeline (`bin/discover.py`) backfills this automatically, but frontend code should also use `getEffectiveArea()` as defense-in-depth.

**Lessons learned:** `~/Documents/docs/lessons/2026-02-14-area-entity-resolution.md`, `~/Documents/docs/lessons/2026-02-14-organic-discovery-implementation.md`

## Gotchas

- **Entity area_id is usually inherited from device** — only 6/3,050 entities have direct area_id. Always resolve via device fallback. See "HA Data Model" above.
- **Collector registration requires extractors import** — `snapshot.py` imports `CollectorRegistry` from `registry.py` but collectors live in `extractors.py`. Without `import aria.engine.collectors.extractors`, the registry is empty and all snapshot metrics are 0. The `__init__.py` and `snapshot.py` both import it now.
- **Predictions fall back to overall average** — When the target day-of-week has no baseline (early data), `predictor.py` averages all available day baselines. Without this, predictions for missing weekdays are all 0.
- **All imports use `aria.*` namespace** — e.g. `from aria.hub.core import IntelligenceHub`, `from aria.engine.config import Config`
- `bin/ha-hub.py` is a legacy wrapper — use `aria serve` instead
- HA WebSocket requires `auth` message with token before subscribing
- Use `/usr/bin/python3` (3.12) not `python3` (3.14, no packages) for manual JSON piping
- SQLite cache persists across restarts — stale data shows until first flush cycle (15 min)
- `activity_summary.websocket` shows `null` until first summary flush after restart (expected)
- Prediction fields start empty on cold boot — need 24-48h of data to populate
- `dist/bundle.js` is gitignored — must rebuild SPA after JSX changes before restart
- `snapshot_log.jsonl` is append-only, never pruned — grows ~1KB/snapshot
- Snapshot subprocess (`aria snapshot-intraday`) uses `asyncio.get_running_loop().run_in_executor` — don't call from sync context
- Module registration order matters — discovery must run before ml_engine (needs capabilities cache)
- Intelligence module reads engine JSON files (entity_correlations, sequence_anomalies, power_profiles, automation_suggestions) — returns `None` gracefully if files don't exist yet
- Engine JSON schema changes require corresponding updates to `_read_intelligence_data()` in `aria/modules/intelligence.py`
- Shadow engine is non-fatal — hub starts without it if init fails (logged at ERROR)
- Shadow predictions need 24-48h of activity data before meaningful accuracy scores
- `hub.publish()` calls BOTH subscriber callbacks AND `module.on_event()` — shadow engine uses subscribe-only pattern with on_event as no-op to prevent double-handling
- Activity monitor emits events via fire-and-forget `create_task()` — never blocks state processing
- Engine commands in `aria` CLI delegate to `aria.engine.cli` internally — they translate subcommands to old-style `--flags`
- **Organic discovery needs 15+ entities per group** — HDBSCAN won't cluster small groups. If discovery finds no organic capabilities, the entity count may be too low or data too homogeneous.
- **Organic discovery Ollama contention** — If LLM naming is enabled, the Sunday 4:00 AM run (~45 min) overlaps with suggest-automations at 4:30 AM. Move one timer if both use Ollama.
- **Capabilities cache is extended, not replaced** — Organic discovery adds fields (`source`, `usefulness`, `layer`, `status`, etc.) to the existing capabilities cache. Existing consumers see the same key with optional new fields. Seed capabilities are always preserved.
