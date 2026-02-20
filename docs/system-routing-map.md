# ARIA System Routing Map

> **Purpose:** Dual-purpose reference — compact lookup tables for Claude Code (Part A), full system documentation for humans (Parts B-C). Read this before grepping the codebase.

---

## Part A — Quick Lookup Tables

### Topic → File(s) Index

| Topic | Primary File(s) | Test File(s) |
|-------|-----------------|--------------|
| Hub core / module registry | `aria/hub/core.py` | `tests/hub/test_hub.py` |
| Cache (SQLite) | `aria/hub/cache.py` | `tests/hub/test_cache.py` |
| API routes (FastAPI) | `aria/hub/api.py` | `tests/hub/test_api.py` |
| Audit logging | `aria/hub/audit.py` | `tests/hub/test_audit.py`, `tests/hub/test_audit_middleware.py`, `tests/integration/test_audit_pipeline.py` |
| Config defaults | `aria/hub/config_defaults.py` | `tests/hub/test_config_defaults.py` |
| Validation runner | `aria/hub/validation_runner.py` | `tests/hub/test_validation_runner.py` |
| Discovery (HA scan + entity classification) | `aria/modules/discovery.py`, `bin/discover.py` | `tests/hub/test_discovery.py` |
| ML training (hub) | `aria/modules/ml_engine.py` | `tests/hub/test_ml_engine.py` |
| Pattern detection | `aria/modules/patterns.py` | `tests/hub/test_patterns.py` |
| Trajectory classifier (Tier 3+) | `aria/modules/trajectory_classifier.py` | `tests/hub/test_trajectory_classifier.py` |
| Orchestrator (automation suggestions) | `aria/modules/orchestrator.py` | `tests/hub/test_orchestrator.py` |
| Shadow engine | `aria/modules/shadow_engine.py` | `tests/hub/test_shadow_engine.py` |
| Intelligence (cache assembly) | `aria/modules/intelligence.py` | `tests/hub/test_intelligence.py` |
| Activity monitor | `aria/modules/activity_monitor.py` | `tests/hub/test_activity_monitor.py` |
| Presence (MQTT + HA sensors) | `aria/modules/presence.py` | `tests/hub/test_presence.py` |
| Watchdog | `aria/watchdog.py` | `tests/hub/test_watchdog.py` |
| Capabilities registry | `aria/capabilities.py` | `tests/hub/test_capabilities.py` |
| Engine — snapshot collection | `aria/engine/collectors/snapshot.py` | `tests/engine/test_snapshot.py` |
| Engine — HA API collection | `aria/engine/collectors/ha_api.py` | `tests/engine/test_ha_api.py` |
| Engine — logbook collection | `aria/engine/collectors/logbook.py` | `tests/engine/test_logbook.py` |
| Engine — metric extractors | `aria/engine/collectors/extractors.py` | `tests/engine/test_extractors.py` |
| Engine — feature vectors | `aria/engine/features/vector_builder.py` | `tests/engine/test_vector_builder.py` |
| Engine — feature config | `aria/engine/features/feature_config.py` | `tests/engine/test_feature_config.py` |
| Engine — feature selection | `aria/engine/features/feature_selection.py` | `tests/engine/test_feature_selection.py` |
| Engine — ML models | `aria/engine/models/` | `tests/engine/test_models.py` |
| Engine — predictions | `aria/engine/predictions/predictor.py` | `tests/engine/test_predictor.py` |
| Engine — baselines | `aria/engine/analysis/baselines.py` | `tests/engine/test_baselines.py` |
| Engine — correlations | `aria/engine/analysis/correlations.py` | `tests/engine/test_correlations.py` |
| Engine — drift detection | `aria/engine/analysis/drift.py` | `tests/engine/test_drift.py` |
| Engine — anomaly detection | `aria/engine/analysis/anomalies.py` | `tests/engine/test_anomalies.py` |
| Engine — occupancy | `aria/engine/analysis/occupancy.py` | `tests/engine/test_occupancy.py` |
| Engine — power profiles | `aria/engine/analysis/power_profiles.py` | `tests/engine/test_power_profiles.py` |
| Engine — anomaly explainer | `aria/engine/anomaly_explainer.py` | `tests/engine/test_anomaly_explainer.py` |
| Engine — sequence classifier | `aria/engine/sequence.py` | `tests/engine/test_sequence.py` |
| Engine — LLM client | `aria/engine/llm/client.py` | `tests/engine/test_llm.py` |
| Engine — data store | `aria/engine/storage/data_store.py` | `tests/engine/test_data_store.py` |
| Engine — hardware tiers | `aria/engine/hardware.py` | `tests/engine/test_hardware.py` |
| CLI (unified) | `aria/cli.py` | (tested via integration) |
| Dashboard (Preact SPA) | `aria/dashboard/spa/src/` | (manual / visual) |
| Dashboard — pipeline Sankey | `aria/dashboard/spa/src/lib/pipelineGraph.js` | (must stay in sync with modules) |

### HTTP Route Table

**70+ routes** in `aria/hub/api.py`. Grouped by domain:

| Route | Method | Handler Purpose |
|-------|--------|----------------|
| `/health` | GET | Hub health check (uptime, module status, cache categories) |
| `/` | GET | JSON status check (`{"status": "ok", "service": "ARIA"}`) |
| **Cache** | | |
| `/api/cache` | GET | List all cache categories |
| `/api/cache/keys` | GET | List categories with timestamps |
| `/api/cache/{category}` | GET | Read cache category |
| `/api/cache/{category}` | POST | Write cache category |
| `/api/cache/{category}` | DELETE | Delete cache category |
| **System** | | |
| `/api/version` | GET | Package version |
| `/api/metrics` | GET | Hub metrics (request count, uptime) |
| `/api/events` | GET | Recent event log |
| `/api/modules` | GET | Registered modules list |
| `/api/modules/{module_id}` | GET | Module detail + status |
| **ML** | | |
| `/api/ml/drift` | GET | Drift status from intelligence cache |
| `/api/ml/features` | GET | Feature selection data |
| `/api/ml/models` | GET | Trained model metadata |
| `/api/ml/anomalies` | GET | Anomaly alerts |
| `/api/ml/shap` | GET | SHAP attribution data |
| `/api/ml/pipeline` | GET | ML pipeline status (training state, tier) |
| `/api/ml/hardware` | GET | Hardware tier detection |
| `/api/ml/online` | GET | Online learner status |
| **Patterns** | | |
| `/api/patterns` | GET | Detected event patterns |
| `/api/transfer` | GET | Transfer engine candidates |
| `/api/anomalies/explain` | GET | Anomaly explanation (IsolationForest paths) |
| **Capabilities** | | |
| `/api/capabilities/candidates` | GET | Organic discovery candidates |
| `/api/capabilities/history` | GET | Discovery run history |
| `/api/capabilities/{name}/promote` | PUT | Promote candidate to active |
| `/api/capabilities/{name}/archive` | PUT | Archive capability |
| `/api/capabilities/{name}/can-predict` | PUT | Toggle prediction target flag |
| `/api/capabilities/registry` | GET | Full capability registry |
| `/api/capabilities/registry/graph` | GET | Capability dependency graph |
| `/api/capabilities/registry/health` | GET | Registry health check |
| `/api/capabilities/registry/{id}` | GET | Single capability detail |
| `/api/capabilities/feedback/health` | GET | Feedback channel health |
| **Discovery Settings** | | |
| `/api/settings/discovery` | GET | Organic discovery settings |
| `/api/settings/discovery` | PUT | Update discovery settings |
| `/api/discovery/run` | POST | Trigger ad-hoc discovery |
| `/api/discovery/status` | GET | Discovery run status |
| **Shadow Engine** | | |
| `/api/shadow/predictions` | GET | Recent shadow predictions |
| `/api/shadow/accuracy` | GET | Shadow accuracy metrics |
| `/api/shadow/disagreements` | GET | Model disagreement cases |
| `/api/shadow/propagation` | GET | DemandSignal propagation trace |
| **Pipeline** | | |
| `/api/pipeline` | GET | Pipeline stage + gate status |
| `/api/pipeline/advance` | POST | Advance pipeline stage |
| `/api/pipeline/retreat` | POST | Retreat pipeline stage |
| `/api/pipeline/topology` | GET | Module topology for frontend Sankey (module IDs, statuses, layers) |
| **Config** | | |
| `/api/config` | GET | All editable config |
| `/api/config/{key}` | GET | Single config key |
| `/api/config/{key}` | PUT | Update config key |
| `/api/config/reset/{key}` | POST | Reset config key to default |
| `/api/config-history` | GET | Config change history |
| **Curation** | | |
| `/api/curation` | GET | Entity curation list |
| `/api/curation/summary` | GET | Curation summary stats |
| `/api/curation/{entity_id}` | PUT | Override entity curation |
| `/api/curation/bulk` | POST | Bulk curation update |
| **Activity** | | |
| `/api/activity/current` | GET | Current activity state |
| `/api/activity/label` | POST | Submit/correct activity label |
| `/api/activity/labels` | GET | All activity labels |
| `/api/activity/stats` | GET | Activity labeler stats |
| **Automation Feedback** | | |
| `/api/automations/feedback` | POST | Submit automation feedback |
| `/api/automations/feedback` | GET | Get automation feedback |
| **Validation** | | |
| `/api/validation/run` | POST | Trigger pytest validation suite |
| `/api/validation/latest` | GET | Latest validation results |
| **Frigate** | | |
| `/api/frigate/thumbnail/{event_id}` | GET | Proxy Frigate thumbnail |
| `/api/frigate/snapshot/{event_id}` | GET | Proxy Frigate snapshot |
| **Audit** | | |
| `/api/audit/events` | GET | Query audit events |
| `/api/audit/requests` | GET | Query audit request log |
| `/api/audit/timeline/{subject}` | GET | Timeline for a subject |
| `/api/audit/stats` | GET | Audit statistics |
| `/api/audit/startups` | GET | Hub startup history |
| `/api/audit/curation/{entity_id}` | GET | Curation audit trail |
| `/api/audit/integrity` | GET | Tamper-evident checksum verification |
| `/api/audit/export` | POST | Export audit data |
| **WebSocket** | | |
| `/ws` | WS | Cache update push (real-time dashboard) |
| `/ws/audit` | WS | Live audit event stream |
| **Static** | | |
| `/ui/{path}` | GET | Preact SPA — catch-all serves index.html for client-side routing; real files (JS/CSS) served directly |

### Event Bus Contract

Events propagate through **two channels**: explicit `hub.subscribe()` callbacks AND `module.on_event()` broadcast to ALL modules. Both fire on every `hub.publish()`.

#### Dual Dispatch Pattern

Every `hub.publish(event_type, data)` call triggers **both** paths unconditionally:

1. **Dispatch path 1 — explicit subscribers:** Only callbacks registered via `hub.subscribe(event_type, cb)` for the matching `event_type` are called.
2. **Dispatch path 2 — module broadcast:** `module.on_event(event_type, data)` is called on every registered module regardless of event type.

A module that also registers an explicit `subscribe()` callback will receive the event **twice** — once via the callback, once via `on_event()`. This is intentional: `subscribe()` is for targeted routing (e.g. wiring `shadow_engine` to `state_changed`); `on_event()` lets any module observe all events without upfront subscription.

#### Backpressure Monitoring (Issue #31 — mitigated)

`publish()` measures wall-clock time for each subscriber callback and each module's `on_event()` call. If any single handler exceeds **100 ms**, a `logger.warning()` is emitted. The total dispatch time is also measured and warned if it exceeds 100 ms. This is observability only — there is no queue drop or back-pressure mechanism; a slow handler still delays subsequent publish() calls inline.

Monitoring added in `aria/hub/core.py` (`publish()` method). Metrics tracked: `_event_count` (cumulative published events), per-callback timing, per-module timing, total dispatch timing.

Test coverage: `tests/hub/test_core.py` (slow-subscriber warning, fast-callback no-warn, dual dispatch firing both paths).

| Event | Emitter | `subscribe()` Listeners | `on_event()` Listeners |
|-------|---------|------------------------|----------------------|
| `config_updated` | `PUT /api/config/{key}` (api.py) | `IntelligenceHub.on_config_updated()` → all modules' `on_config_updated(config)` (core.py) | All modules receive via on_config_updated (no-op default in base class) |
| `cache_updated` | `hub.set_cache()` (core.py:348) | `broadcast_cache_update` → WebSocket `/ws` (api.py:1464) | `orchestrator.on_event()` — watches for `category == "patterns"` |
| `state_changed` | `activity_monitor` (activity_monitor.py:389) | `shadow_engine._on_state_changed` (shadow_engine.py:273) | All modules receive — most ignore |
| `shadow_resolved` | `shadow_engine` (shadow_engine.py:968, 1018) | `trajectory_classifier._on_shadow_resolved` (trajectory_classifier.py:119) | All modules receive via on_event |
| `discovery_complete` | `discovery` (discovery.py:287) | (none) | All modules receive |
| `pipeline_updated` | `api.py` route handlers (api.py:1029, 1060) | (none) | All modules receive |
| `curation_updated` | `api.py` route handlers (api.py:1212, 1227) | (none) | All modules receive |
| `presence_updated` | `presence` (presence.py:716) | (none) | All modules receive |
| `intelligence_digest` | `intelligence` (intelligence.py:204) | (none) | All modules receive |
| `automation_created` | `orchestrator` (orchestrator.py:370) | (none) | All modules receive |
| `automation_updated` | `orchestrator` (orchestrator.py:421) | (none) | All modules receive |
| `automation_approved` | `orchestrator` (orchestrator.py:371) | (none) | All modules receive — emitted when a suggestion is approved and an HA automation created |
| `automation_rejected` | `orchestrator` (orchestrator.py:422) | (none) | All modules receive — emitted when a suggestion is rejected |

### MQTT Topics

| Topic Pattern | Publisher | Subscriber | Data |
|---------------|----------|------------|------|
| `frigate/events` | Frigate NVR | `presence.py` (presence.py:366) | Person/face detection events (JSON) |
| `frigate/+/person` | Frigate NVR | `presence.py` (presence.py:367) | Per-camera person detection |

**Broker:** Mosquitto on `<mqtt-broker-ip>:1883` (core_mosquitto addon on HA Pi). Credentials in `config_defaults.py` presence section.

### Presence Domain Filtering (Domains in `_PRESENCE_DOMAINS`)

| Domain | Presence Signal | States Processed |
|--------|-----------------|------------------|
| `person.*` | Home/away state | (special: person-level, not room-level) |
| `light.*` | Motion, occupancy | All except unavailable/unknown |
| `binary_sensor.*` | Motion, occupancy, occupancy_status | All except unavailable/unknown |
| `media_player.*` | Media active indicator | playing/paused/idle/buffering→0.85; off/standby→0.15 |
| `event.*` | Device button events (Hue dimmer, etc.) | Custom handlers per event.* subclass |

**Cold-start:** Presence module seeds all four domains on startup from initial HA state fetch, not just person.* home/away. Prevents room probabilities from being zero at boot time.

### Systemd Timer Map

| Timer | Schedule | Command | Writes | Read By |
|-------|----------|---------|--------|---------|
| `aria-hub.service` | persistent daemon | `aria serve --port 8001` | hub.db, audit.db | dashboard, API clients |
| `aria-watchdog.timer` | every 5 min | `aria watchdog` | watchdog log | (monitoring) |
| `aria-snapshot.timer` | daily 23:00 | `aria snapshot` | `~/ha-logs/intelligence/daily/*.json` | intelligence module |
| `aria-intraday.timer` | every 4h (00,04,08,12,16,20) | `aria snapshot-intraday` | `~/ha-logs/intelligence/intraday/*.json` | intelligence module |
| `aria-full.timer` | daily 23:30 | `aria full` (via ollama-queue) | predictions, baselines, analyses JSON | intelligence module |
| `aria-check-drift.timer` | daily 02:00 | `aria check-drift` | drift status JSON | intelligence module |
| `aria-retrain.timer` | Sunday 02:00 | `aria retrain` | model `.joblib` files | ml_engine module |
| `aria-meta-learn.timer` | Monday 01:30 | `aria meta-learn` (via ollama-queue) | meta-learning JSON | intelligence module |
| `aria-correlations.timer` | Sunday 03:15 | `aria correlations` | entity correlations JSON | intelligence module |
| `aria-sequences.timer` | Sunday 03:45 | `aria sequences detect` | sequence anomalies JSON | intelligence module |
| `aria-suggest-automations.timer` | Sunday 04:30 | `aria suggest-automations` (via ollama-queue) | automation suggestions JSON | intelligence module |
| `aria-prophet.timer` | Monday 03:00 | `aria prophet` | prophet forecasts JSON | intelligence module |

**Via Ollama Queue:** `aria-full`, `aria-meta-learn`, `aria-suggest-automations` route through `ollama-queue` at port 7683 to serialize Ollama access.

### Cache Category Owners

| Category | Primary Writer(s) | Reader(s) |
|----------|-------------------|-----------|
| `entities` | discovery | presence, ml_engine |
| `devices` | discovery | presence |
| `areas` | discovery | (dashboard) |
| `capabilities` | discovery, ml_engine, shadow_engine | ml_engine, shadow_engine |
| `discovery_metadata` | discovery | (API/dashboard) |
| `intelligence` | intelligence | shadow_engine |
| `activity_log` | activity_monitor | intelligence, ml_engine, shadow_engine |
| `activity_summary` | activity_monitor | intelligence, shadow_engine |
| `presence` | presence | engine snapshot collector (read-back) |
| `patterns` | patterns | orchestrator (via on_event), shadow_engine |
| `automation_suggestions` | orchestrator | (API/dashboard) |
| `pending_automations` | orchestrator | (API/dashboard) |
| `created_automations` | orchestrator | (API/dashboard) |
| `automation_feedback` | API route handler | orchestrator |
| `ml_predictions` | ml_engine | Predictions.jsx (via useCache) |
| `ml_ensemble_weights` | ml_engine | (API/dashboard) |
| `feature_config` | ml_engine | ml_engine |
| `ml_training_metadata` | ml_engine | ml_engine |
| `latest_snapshot` | (engine via cache API) | ml_engine |
| `discovery` | (engine via cache API) | ml_engine |
| `pattern_trajectory` | trajectory_classifier | ml_engine |
| `config` | config_defaults, API | (all modules via hub) |
| `entity_curation` | discovery, API | (filtering layer) |

### Subprocess Calls

| Caller | Subprocess | Produces | Error Handling |
|--------|-----------|----------|---------------|
| `discovery.py` (module) | `bin/discover.py` as Python subprocess | JSON stdout (entities, devices, areas, capabilities) | TimeoutExpired caught, stderr logged |
| `activity_monitor.py` | `aria snapshot-intraday` | Intraday snapshot JSON file | TimeoutExpired (30s), stderr[:200] logged |
| `cli.py` (`sync-logs`) | `bin/ha-log-sync` script | Logbook sync to `~/ha-logs/logbook/` | `check=True` — raises on failure |
| `validation_runner.py` | `pytest` subprocess | JSON test results | TimeoutExpired caught |
| `ha_api.py` (collector) | `gog calendar list --today --all --plain` | Google Calendar events for today | stderr logged on failure |
| `watchdog.py` | `systemctl --user` commands | Timer status, journal output | Multiple subprocess.run calls |

---

## Part B — Data Flow Diagrams

### 1. Batch Pipeline (Engine → Hub)

```
┌─────────────────────┐
│ Home Assistant REST  │
│ GET /api/states      │
│ GET /api/logbook     │
│ GET /api/calendars   │
└────────┬────────────┘
         │
    ┌────▼────────────────────────┐
    │ aria/engine/collectors/     │
    │  ha_api.py → fetch states   │
    │  logbook.py → summarize     │
    │  snapshot.py → build JSON   │
    │                             │
    │  Also reads: hub API        │◄── presence cache read-back
    │  /api/cache/presence        │    (or direct hub.db fallback)
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │ ~/ha-logs/intelligence/     │
    │  daily/YYYY-MM-DD.json      │
    │  intraday/*.json            │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │ aria/engine/                │
    │  features/ → vector_builder │  ◄── single source of truth
    │  models/ → train/predict    │      (also imported by hub ml_engine)
    │  analysis/ → baselines etc  │
    │  llm/ → meta-learning       │
    └────────┬────────────────────┘
             │ writes JSON + .joblib
    ┌────────▼────────────────────┐
    │ ~/ha-logs/intelligence/     │
    │  models/*.joblib            │
    │  predictions, baselines,    │
    │  correlations, etc.         │
    └────────┬────────────────────┘
             │
    ┌────────▼────────────────────┐
    │ intelligence.py module      │
    │ _read_intelligence_data()   │  ◄── JSON schema coupling (seam risk)
    │  → hub.set_cache()          │
    └────────┬────────────────────┘
             │ publish("cache_updated")
    ┌────────▼──────────┐  ┌───────────────┐
    │ GET /api/cache/*   │  │ WebSocket /ws  │
    │ (REST response)    │  │ (push to SPA)  │
    └────────────────────┘  └───────┬───────┘
                                    │
                            ┌───────▼───────┐
                            │ Preact SPA    │
                            │ /ui/          │
                            └───────────────┘
```

### 2. Real-time Stream (HA → Shadow → Feedback)

```
┌──────────────────────────┐
│ Home Assistant WebSocket  │
│ subscribe_events:         │
│   state_changed           │
└────────┬─────────────────┘
         │ (activity_monitor connection)
    ┌────▼────────────────────────────┐
    │ ActivityMonitor                  │
    │  → filter by TRACKED_DOMAINS    │
    │  → 15-min window aggregation    │
    │  → set_cache(activity_log)      │
    │  → set_cache(activity_summary)  │
    │  → adaptive snapshot trigger ───┼──► subprocess: aria snapshot-intraday
    │  → hub.publish("state_changed") │
    └────────┬────────────────────────┘
             │ (event bus)
    ┌────────▼────────────────────────┐
    │ ShadowEngine                     │
    │  (subscribes to "state_changed") │
    │  → capture context snapshot      │
    │  → generate predictions          │
    │    (next_domain, room_activation,│
    │     routine_trigger)             │
    │  → store in predictions table    │
    │  → 10-min window → score actuals │
    │  → hub.publish("shadow_resolved")│
    └────────┬────────────────────────┘
             │ (event bus)
    ┌────────────────────────────┐
    │ Tier 3+ Subscribers:       │
    │                            │
    │ TrajectoryClassifier       │
    │  → DTW classification      │
    │  → scale tagging           │
    └────────────────────────────┘
```

### 3. Presence Signal Flow

```
┌─────────────────────┐    ┌──────────────────────────┐
│ Frigate NVR         │    │ Home Assistant WebSocket  │
│ (camera detection)  │    │ subscribe_events:         │
└────────┬────────────┘    │   state_changed           │
         │ MQTT            └────────┬─────────────────┘
         │ frigate/events           │ (presence connection)
         │ frigate/+/person         │
    ┌────▼──────────────────────────▼───┐
    │ PresenceModule                     │
    │  → MQTT: person/face events        │
    │  → WS: motion, lights, dimmers,    │
    │         device trackers, doors      │
    │  → feed BayesianOccupancy          │
    │  → flush every 30s                  │
    │  → set_cache("presence")            │
    │  → publish("presence_updated")      │
    └────────┬───────────────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │ Engine snapshot collector          │
    │  reads /api/cache/presence         │
    │  (or SQLite hub.db fallback)       │
    │  → embeds presence features        │
    │    in daily/intraday snapshots     │
    └───────────────────────────────────┘
```

### 4. Organic Discovery Pipeline

> **Archived** — lean audit 2026-02-19. The weekly HDBSCAN pipeline (feature_vectors, clustering, behavioral, scoring, seed_validation, naming) has been removed. See `_archived/modules/organic_discovery/`.

### 5. Closed-Loop Feedback

```
    ┌─────────────────┐
    │ Shadow Engine    │
    │ predictions →    │
    │ accuracy scores  │
    └────────┬────────┘
             │
    ┌────────▼────────────────────────────────────────────────────────┐
    │                    Feedback Channels                            │
    │                                                                │
    │  shadow accuracy → DemandSignal propagation                    │
    │  automation feedback → orchestrator adjustment                  │
    │  drift detection → retrain trigger                             │
    └────────────────────────────────────────────────────────────────┘
```

---

## Part B2 — Dashboard SPA → API Mapping

Which page components call which API endpoints. All pages also receive real-time updates via the WebSocket `/ws` connection (`store.js:connectWebSocket`).

### Cache-Driven Pages (via `useCache()` hook → `GET /api/cache/{category}` + WebSocket push)

| Page | File | Cache Categories | Additional API Calls |
|------|------|-----------------|---------------------|
| Home | `pages/Home.jsx` | `intelligence`, `activity_summary`, `entities` | `/api/shadow/accuracy`, `/api/pipeline`, `/api/curation/summary`, `/api/activity/current` |
| Intelligence | `pages/Intelligence.jsx` + 13 sub-files in `pages/intelligence/` (ActivitySection, AnomalyAlerts, Baselines, Configuration, Correlations, DailyInsight, DriftStatus, HomeRightNow, LearningProgress, PredictionsVsActuals, ShapAttributions, SystemStatus, TrendsOverTime) | `intelligence` | `/api/shadow/accuracy`, `/api/pipeline`, `/api/ml/drift`, `/api/ml/anomalies`, `/api/ml/shap` |
| Discovery | `pages/Discovery.jsx` | `entities`, `devices`, `areas`, `capabilities` | — |
| Capabilities | `pages/Capabilities.jsx` | `capabilities` | `/api/capabilities/registry`, `PUT /api/capabilities/{name}/can-predict` |
| Presence | `pages/Presence.jsx` | `presence` | `/api/frigate/thumbnail/{event_id}` |
| Automations | `pages/Automations.jsx` | `automation_suggestions` | `POST /api/cache/automation_suggestions` (delete) |
| Patterns | `pages/Patterns.jsx` | `patterns` | — |
| Predictions | `pages/Predictions.jsx` | `ml_predictions` | — |

### Direct-Fetch Pages (no `useCache()` — fetch on mount or user action)

| Page | File | API Calls | Write Actions |
|------|------|-----------|--------------|
| Shadow | `pages/Shadow.jsx` | `/api/pipeline`, `/api/shadow/accuracy`, `/api/shadow/predictions`, `/api/shadow/disagreements`, `/api/shadow/propagation` | `POST /api/pipeline/advance`, `POST /api/pipeline/retreat` |
| ML Engine | `pages/MLEngine.jsx` | `/api/ml/features`, `/api/ml/models`, `/api/ml/drift`, `/api/ml/pipeline` | — |
| Data Curation | `pages/DataCuration.jsx` | `/api/curation`, `/api/curation/summary` | `PUT /api/curation/{entity_id}`, `POST /api/curation/bulk` |
| Settings | `pages/Settings.jsx` | `/api/config` | `PUT /api/config/{key}`, `POST /api/config/reset/{key}` |
| Validation | `pages/Validation.jsx` | `/api/validation/latest` | `POST /api/validation/run` |
| Guide | `pages/Guide.jsx` | — | — (static content) |

### Shared Components with API Calls

| Component | File | API Calls |
|-----------|------|-----------|
| PresenceCard | `components/PresenceCard.jsx` | `useCache('presence')` |
| CapabilityDetail | `components/CapabilityDetail.jsx` | `PUT /api/capabilities/{name}/promote`, `PUT /api/capabilities/{name}/archive` |
| DiscoverySettings | `components/DiscoverySettings.jsx` | `/api/settings/discovery`, `PUT /api/settings/discovery`, `POST /api/discovery/run` |
| Sidebar | `components/Sidebar.jsx` | WebSocket status indicator only |

### Data Flow: Cache Update → Dashboard Render

```
Module writes cache (hub.set_cache)
    → hub.publish("cache_updated", {category, version})
    → api.py: broadcast_cache_update() [subscribe callback]
        → WebSocketManager.broadcast({type: "cache_updated", data: ...})
            → store.js: ws.onmessage handler
                → updates signal for that cache category
                    → useCache(name) re-renders subscribed components
```

**Latency:** Sub-second from cache write to dashboard re-render (WebSocket push, no polling).

**Gap:** Direct-fetch pages (Shadow, ML Engine, Data Curation, Settings, Validation) do NOT get real-time updates. They fetch on mount and require manual refresh or page navigation to see new data.

## Part B2b — Startup Sequence & Error Propagation

### Module Registration Order (`cli.py:_register_modules`)

Modules initialize **sequentially** in this exact order. Each module's `initialize()` runs before the next is registered. Modules in `try/except` blocks are **non-critical** — failure skips them but doesn't stop the hub.

| # | Module | Critical? | Init Starts | On Failure |
|---|--------|-----------|------------|------------|
| 1 | `discovery` | **Yes** | WebSocket listener, initial HA scan + entity classification | Hub starts but entities cache empty — downstream modules degraded |
| 2 | `ml_engine` | **Yes** | Loads models, schedules periodic training | No predictions available |
| 3 | `patterns` | **Yes** | Reads logbook data | No pattern detection |
| 4 | `orchestrator` | **Yes** | Reads patterns cache | No automation suggestions |
| 5 | `shadow_engine` | No | Subscribes to state_changed | No shadow predictions |
| 6 | `intelligence` | **Yes** | Reads engine JSON files, schedules 15min check | Intelligence cache stale, no Telegram digest |
| 7 | `trajectory_classifier` | No | Subscribes to shadow_resolved (Tier 3+) | No trajectory classification |
| 8 | `activity_monitor` | No | WebSocket listener, 15min summary | No activity data, shadow engine starved |
| 9 | `presence` | No | MQTT + WebSocket listeners | No presence data, snapshots get zero features |

**Registration structure:** `cli.py` uses 4 functions: `_register_modules()` (1–4), `_register_analysis_modules()` (5–6), `_register_ml_modules()` (7), `_register_monitor_modules()` (8–9). Analysis → ML → Monitors is the group order.

**Cold-start ordering risks:**
- `activity_monitor` (#8) fires `state_changed` events that `shadow_engine` (#5) subscribes to — ordering is correct here (shadow subscribes first).
- `intelligence` (#6) reads engine JSON files — if batch timers haven't run yet (fresh install), intelligence cache is empty for 24h.
- `discovery` (#1) entity classification runs `_reclassify_all` on startup — if the initial HA scan is still in progress, curation may process a stale cache. Self-corrects on next 24h cycle (risk partially mitigated by merging data_quality into discovery).

### Error Propagation Paths

When something fails, how does the failure surface?

| Failure | Detection Path | User Visibility | Latency |
|---------|---------------|----------------|---------|
| Module init failure | `hub.mark_module_failed()` → `/health` endpoint shows "failed" | Dashboard health indicator | Immediate |
| Module init failure | Watchdog checks module status every 5min | Telegram alert (if enabled) | Up to 5min |
| HA WebSocket disconnect | Module logs warning → exponential backoff reconnect | No user visibility unless watchdog detects stale data | Silent until next watchdog |
| MQTT disconnect | Presence module logs warning → backoff reconnect | Presence data goes stale → dashboard shows old presence | Silent |
| Engine batch timer failure | Watchdog checks `systemctl --user` timer status | Telegram alert | Up to 5min |
| Cache write failure | SQLite error propagates → module logs error | No user visibility (module continues) | Silent |
| Telegram send failure | `intelligence.py` / `watchdog.py` log error | **No visibility** — the alerting channel itself fails silently | Silent |
| Ollama queue timeout | Module logs warning, produces no output | Feature degrades (no LLM-enhanced batch output) | Silent |

**Key gap:** There is no "alert about alert failure" mechanism. If Telegram sending fails, the only evidence is in journalctl logs.

### Telegram Alerting Paths

Two independent Telegram senders with no shared state:

| Sender | Location | Triggers | Cooldown |
|--------|----------|----------|----------|
| Watchdog | `aria/watchdog.py:519` | Timer failures, hub health failures, recovery | Per-alert-key cooldown (watchdog internal) |
| Intelligence digest | `aria/modules/intelligence.py:531` | New daily intelligence data assembled | None (fires once per digest cycle) |

**Both** read `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from environment. Neither checks if the other has sent recently. On a bad day (multiple failures + daily digest), the user could receive a burst of messages.

### Config Change Propagation

When config is updated via `PUT /api/config/{key}`:

1. Value written to `config` cache category
2. `cache_updated` event published with `category="config"`
3. **No module explicitly subscribes to config changes**

Modules read config **on their own schedule** (e.g., shadow engine reads `min_confidence` on each prediction, ml_engine reads `feature_config` on each training run). Config changes take effect:

| Config Area | Effective After |
|-------------|----------------|
| Shadow engine thresholds | Next prediction (seconds) |
| ML training parameters | Next training run (hours to days) |
| Discovery intervals | Next scheduled run (hours) |
| Pattern recognition (`pattern.min_tier`, `pattern.sequence_window_size`) | Next hub restart — read once in `initialize()` (issue #29 fixed) |
| Presence flush interval | **Never** — hardcoded constant, not read from config |

**Gap:** No immediate "reload config" mechanism. Some config changes appear effective immediately (shadow engine reads per-prediction), others require a restart or next scheduled cycle.

## Part B3 — Fallback Routing

What happens when the primary path fails. Every module with external dependencies has a fallback strategy — or doesn't.

### WebSocket Reconnection (Exponential Backoff)

Three modules maintain persistent WebSocket/MQTT connections with identical retry patterns:

| Module | Connection | Backoff | Max Delay | Reset On |
|--------|-----------|---------|-----------|----------|
| `discovery` | HA WebSocket (entity registry) | 5s × 2^n | 60s | Successful auth |
| `activity_monitor` | HA WebSocket (state_changed) | 5s × 2^n | 60s | Successful auth |
| `presence` (MQTT) | Frigate MQTT broker | 5s × 2^n | 60s | Successful connect |
| `presence` (WS) | HA WebSocket (sensors) | 5s × 2^n | 60s | Successful auth |

**Mitigated (closes #20):** All three WS/MQTT modules (`discovery`, `activity_monitor`, `presence`) now apply ±25% jitter to each reconnect delay, preventing thundering-herd on simultaneous HA restarts. Coordinated stagger deferred as YAGNI — independent jitter is sufficient for the observed failure mode.

### Data Fallback Chains

| Module | Primary Source | Fallback | Fallback of Fallback |
|--------|---------------|----------|---------------------|
| Engine snapshot collector | Hub API `/api/cache/presence` | Direct SQLite read of `hub.db` | Zero-valued presence features (silent) |
| Activity monitor filtering | `entity_curation` table (discovery) | Domain-based filtering (`TRACKED_DOMAINS`) | Accept all state_changed (noisy) |
| Patterns module | Intraday snapshot JSON files | Logbook files from `~/ha-logs/logbook/` | No patterns detected (empty cache) |
| ML engine feature config | Hub cache `feature_config` | `DEFAULT_FEATURE_CONFIG` constant | (no further fallback) |
| Shadow engine confidence | Config store `min_confidence` key | Hardcoded constant | (no further fallback) |
| Shadow engine window | Config store `scoring_window_minutes` | Hardcoded constant | (no further fallback) |
| Activity monitor snapshot | `aria` CLI path from shutil | `python -m aria.cli` fallback | Log error, skip snapshot |

**Key pattern:** Most fallbacks are **silent** — they produce degraded output without errors. Only the WebSocket reconnects log warnings.

### No Fallback (Single Point of Failure)

| Module | Dependency | What Happens If Down |
|--------|-----------|---------------------|
| Intelligence module | Engine JSON files in `~/ha-logs/intelligence/` | Intelligence cache goes stale — dashboard shows old data |
| ML engine | Trained `.joblib` model files | Training fails, stale model used, stale training check triggers retrain on next startup |
| Watchdog | `systemctl --user` availability | Watchdog reports all timers as failed |
| Audit logger | `audit.db` SQLite | Audit events lost (async buffer, no retry) |

## Part B4 — Multi-Module Task Coordination

How multiple modules collaborate on shared workflows, and where synchronization exists (or is missing).

### Module Scheduled Tasks (hub.schedule_task)

Each module registers periodic tasks during `initialize()`. These run as `asyncio.Task` instances — **no mutual exclusion between tasks from different modules**.

| Module | Task | Interval | Runs Immediately | Dependencies |
|--------|------|----------|-----------------|-------------|
| hub (core) | `prune_events` | 24h | Yes | None |
| audit (cli.py) | `prune_audit` | 24h | — | audit.db |
| discovery | `_initial_discovery` | One-shot | Yes | HA connectivity |
| discovery | `_ws_registry_loop` | One-shot (internal loop) | Yes | HA WebSocket |
| discovery | `_periodic_discovery` | 6h (configurable) | — | HA REST API |
| discovery | `data_quality_reclassify` | 24h | Yes | entities cache (from discovery) |
| intelligence | `_periodic_check` | 15min | — | Engine JSON files |
| activity_monitor | `_ws_listen_loop` | One-shot (internal loop) | Yes | HA WebSocket |
| activity_monitor | `_periodic_summary` | 15min | — | activity_log cache |
| ml_engine | `_periodic_retraining_check` | 7d (configurable) | — | Latest snapshot, discovery cache |
| presence (MQTT) | `_mqtt_listen` | One-shot (internal loop) | Yes | MQTT broker |
| presence (WS) | `_ws_listen` | One-shot (internal loop) | Yes | HA WebSocket |
| presence (flush) | `_flush_presence` | 30s | — | BayesianOccupancy state |
| presence (stale) | `_check_stale_signals` | 5min | — | Presence signal timestamps |

### Multi-Module Pipelines (ordered by data dependency)

**Pipeline 1: Discovery → Curation → ML Training**
```
discovery (initial_discovery)
    → writes: entities, devices, areas, capabilities cache
    → triggers: discovery._reclassify_all (on startup, merged from data_quality)
        → writes: entity_curation table
        → triggers: activity_monitor loads curation rules (on startup)
    → triggers: ml_engine reads capabilities (when cache_updated fires)
        → trains models per capability
```
**Sync gap:** No explicit wait. `discovery._reclassify_all` and `ml_engine` both run on startup — if discovery's initial HA scan hasn't finished, they read stale/empty data. Self-corrects on next periodic run but **first-boot is degraded**.

**Pipeline 2: Activity → Shadow → Trajectory Classification**
```
activity_monitor (state_changed events)
    → publish("state_changed")
    → shadow_engine._on_state_changed() [subscribe]
        → predict → store → score
        → publish("shadow_resolved")
        → trajectory_classifier._on_shadow_resolved() [subscribe]
```
**Sync:** Fully event-driven, sequential within `publish()`. Shadow engine must finish before trajectory_classifier receives the event (because `publish()` awaits each callback in order). **Risk:** If shadow engine's callback is slow, it blocks delivery to all other subscribers.

**Pipeline 3: Snapshot → Intelligence → Dashboard**
```
aria-snapshot.timer (daily 23:00) OR activity_monitor (adaptive)
    → aria snapshot-intraday [subprocess]
    → writes: ~/ha-logs/intelligence/intraday/*.json
    → intelligence._periodic_check (every 15min) [polls files]
        → _read_intelligence_data()
        → set_cache("intelligence")
        → publish("cache_updated")
        → WebSocket broadcast → dashboard
```
**Sync gap:** File-based coupling. Intelligence module polls every 15min — up to 15min delay between snapshot write and dashboard update. No file-watch mechanism.


### Concurrency Hazards

| Hazard | Modules Involved | Mechanism | Mitigation |
|--------|-----------------|-----------|-----------|
| Simultaneous cache writes | Any two modules writing same category | SQLite WAL mode | WAL handles concurrent writes safely, but last write wins if both write same category |
| Validation runner concurrency | API `POST /api/validation/run` | `threading.Lock` (`_run_lock`) | Only lock in the codebase — prevents concurrent pytest runs |
| Timer + adaptive snapshot overlap | `aria-intraday.timer` + `activity_monitor` subprocess | None | Both can fire simultaneously (see RISK-03) |
| Event handler blocking | Any slow `subscribe()` callback | `hub.publish()` is sequential | Slow callback blocks delivery to remaining subscribers |
| Module init race | discovery vs ml_engine (discovery._reclassify_all vs discovery._initial_discovery) | Init order in `cli.py` | Modules init in registration order, but async tasks start immediately |

### What's NOT Synchronized (by design)

- **No distributed locks** — modules trust SQLite WAL for cache safety
- **No message queue** — event bus is in-process `asyncio` only (no persistence, no replay)
- **No transaction boundaries** — a module can read cache, compute, and write back without atomic guarantee that the source didn't change
- **No backpressure queue** — if shadow_engine produces events faster than trajectory_classifier can consume, events queue in memory (asyncio task queue); slow handlers block inline. Backpressure *monitoring* (100 ms warning threshold) was added in Issue #31 — see Event Bus Contract above.

## Part C — Seam Risk Catalog

Integration boundaries where bugs hide. Ordered by estimated risk (silent corruption > loud crash).

### RISK-01: Engine JSON → Hub `_read_intelligence_data()` — Schema Coupling ✓ MITIGATED

- **Boundary:** Engine writes JSON files to `~/ha-logs/intelligence/`; hub's `intelligence.py:_read_intelligence_data()` parses them
- **What crosses:** JSON with fields like `entity_correlations`, `sequence_anomalies`, `power_profiles`, `automation_suggestions`, `drift_status`
- **What can go wrong:** Engine adds/renames/removes a field; hub reads stale or missing keys
- **Test coverage:** **Covered** — `tests/integration/test_engine_hub_integration.py` contains four contract tests:
  - `test_snapshot_schema_round_trip` — verifies a minimal valid snapshot passes `validate_snapshot_schema()` with no errors
  - `test_required_keys_match_hub_reader` — asserts every `(section, key)` accessed by `METRIC_PATHS` is present in `REQUIRED_NESTED_KEYS`, closing the schema-reader gap
  - `test_schema_rejects_missing_required_keys` — incomplete snapshot must produce validation errors
  - `test_engine_output_consumable_by_hub` — writes a snapshot to disk and calls `_extract_trend_data()` via a real `IntelligenceModule` instance, verifying all metrics extract correctly
- **Failure mode:** **Silent** — hub shows empty/stale data in dashboard, no error logged
- **Mitigation applied (2026-02-18):** Shared `aria.engine.schema` module (`REQUIRED_NESTED_KEYS`, `validate_snapshot_schema`) imported by both engine and hub. Contract tests added in `tests/integration/test_engine_hub_integration.py` (closes GitHub issue #19). Any future schema change that breaks hub consumption will be caught at test time.

### RISK-02: Dual Event Propagation — subscribe() vs on_event()

- **Boundary:** `hub.publish()` calls BOTH `subscribe()` callbacks AND `module.on_event()` on ALL modules
- **What crosses:** Event type string + data dict
- **What can go wrong:** Developer adds a subscription but another module already handles the same event via `on_event()` — double processing. Or: module expects subscription-only delivery but `on_event()` fires first
- **Test coverage:** **Partial** — individual modules tested, but no test verifies that dual delivery doesn't cause side effects
- **Failure mode:** **Silent** — duplicate cache writes, double-counted metrics, or race conditions between subscribe callback and on_event handler
- **Suggested mitigation:** Document which modules use which channel (this table above). Lint rule: modules should use ONE channel per event, not both

### RISK-03: Activity Monitor ↔ Systemd Intraday Timer — Race Condition

- **Boundary:** `activity_monitor.py` triggers `aria snapshot-intraday` via subprocess; `aria-intraday.timer` also runs `aria snapshot-intraday` on a fixed schedule
- **What crosses:** Both invoke the same CLI command that writes to `~/ha-logs/intelligence/intraday/`
- **What can go wrong:** Both fire simultaneously → duplicate snapshot with same timestamp, or file write collision (though atomic write mitigates partial corruption)
- **Test coverage:** **None** — no test verifies concurrent invocation behavior
- **Failure mode:** **Mostly silent** — duplicate data wastes disk; if filenames collide, last write wins (atomic write prevents corruption but data could be lost)
- **Suggested mitigation:** File lock or deduplication check in snapshot builder (check if snapshot for current window already exists)

### RISK-04: Presence Cache Read-Back — Circular Dependency

- **Boundary:** Engine snapshot collector reads hub's presence cache via HTTP (`/api/cache/presence`) or direct SQLite fallback
- **What crosses:** Presence feature vector embedded in daily/intraday snapshots
- **What can go wrong:** Hub is down during batch run → fallback reads stale SQLite data; or hub just restarted and presence cache is empty (cold-start, Cluster C)
- **Test coverage:** **Partial** — fallback tested in isolation, but not tested with actual stale data or cold hub
- **Failure mode:** **Silent** — snapshot contains zero-valued presence features → ML models train on bad data
- **Suggested mitigation:** Snapshot validation layer should check for all-zero presence features and warn
- **Status: MITIGATED** — `PresenceModule._seed_presence_from_ha()` now queries `person.*` entity states from the HA REST API during `initialize()`, populating `_person_states` before listener loops start. Cold-start zeros eliminated. See `aria/modules/presence.py` and `tests/hub/test_presence.py` (3 seed tests). Remaining gap: hub-down fallback still reads potentially stale SQLite data.

### RISK-05: Feature Vector Source of Truth — Dual Import ✓ MITIGATED

- **Boundary:** `aria/engine/features/vector_builder.py` is imported by both `engine/cli.py` (batch training) and `modules/ml_engine.py` (hub real-time training)
- **What crosses:** Feature vector construction logic, feature names, feature ordering
- **Root cause (2026-02-18):** `ml_engine._get_feature_names()` was missing the `presence_features` section, so 4 presence features (`presence_probability`, `presence_occupied_rooms`, `presence_identified_persons`, `presence_camera_signals`) were built by `_engine_build_feature_vector()` but silently dropped from the name list used to assemble the numpy matrix. Predictions were trained on column N = one thing, scored on column N = another.
- **Fix applied (2026-02-18):** Added `presence_features` collection to `ml_engine._get_feature_names()` in the correct position (after `interaction_features`, before rolling window features), matching the engine's `get_feature_names()` ordering. See `aria/modules/ml_engine.py`.
- **Mitigation type:** **By test** (detection) — four contract tests added to `tests/integration/test_engine_hub_integration.py` (closes GitHub issue #26):
  - `test_feature_names_engine_hub_base_identical` — asserts shared base feature lists are identical
  - `test_feature_ordering_engine_hub_base_identical` — asserts column indices match (ordering, not just set equality)
  - `test_hub_rolling_window_features_extend_engine_base` — asserts hub list starts with engine base as strict prefix
  - `test_feature_vector_build_and_name_list_consistent` — asserts `build_feature_vector()` produces all keys named by `get_feature_names()` (excluding `trajectory_class` which is externally provided)
- **Remaining intentional divergence:** The hub appends hub-only features AFTER the shared base: 12 rolling window stats (`rolling_{1,3,6}h_{event_count,domain_entropy,dominant_domain_pct,trend}`) from the live activity log, and `trajectory_class` from pattern cache. The engine zero-fills `trajectory_class` in `build_training_data()`. This divergence is by design — the hub has live data the batch engine does not.
- **Failure mode:** **Silent** — model trained on features A, scored on features B → predictions are garbage but no error raised
- **Future protection:** Any future feature added to `vector_builder.get_feature_names()` or removed from `ml_engine._get_feature_names()` (or vice versa) will be caught by `test_feature_names_engine_hub_base_identical` at test time.

### RISK-06: Subprocess Exit Code Handling — Discovery

- **Boundary:** `discovery.py` calls `bin/discover.py` as subprocess, parses JSON from stdout
- **What crosses:** JSON blob with entities, devices, areas, capabilities
- **What can go wrong:** Subprocess exits with error but produces partial stdout; or stdout is empty/malformed
- **Test coverage:** **Mocked** — unit tests mock subprocess.run, never test actual subprocess execution
- **Failure mode:** **Partial** — TimeoutExpired is caught, but malformed JSON would raise and potentially leave cache stale
- **Suggested mitigation:** Integration test that runs actual `bin/discover.py` (against mock HA); validate JSON output schema

### RISK-07: Ollama Queue Availability

- **Boundary:** `aria-full`, `aria-meta-learn`, `aria-suggest-automations` all route through `ollama-queue` at port 7683
- **What crosses:** HTTP requests to enqueue Ollama inference tasks
- **What can go wrong:** Queue service down, queue full, Ollama itself down → requests timeout or fail silently
- **Test coverage:** **None** — all Ollama interactions mocked in tests
- **Failure mode:** **Silent timeout** — batch tasks produce no LLM-enhanced output; no retry logic
- **Suggested mitigation:** Health check endpoint on ollama-queue; watchdog should verify queue is responding

### RISK-08: Module Init Order — Event Subscription Timing

- **Boundary:** Modules register and subscribe during `initialize()`, which runs in registration order (9 modules as of lean audit 2026-02-19)
- **What crosses:** Events emitted during initialization of early modules may not be received by later modules
- **What can go wrong:** If `activity_monitor` emits `state_changed` during init (unlikely but possible on rapid reconnect), `shadow_engine` hasn't subscribed yet
- **Test coverage:** **None** — module init order not tested as a sequence
- **Failure mode:** **Silent** — missed events during startup window (self-corrects once all modules running)
- **Suggested mitigation:** Low risk since events during init are rare. Document registration order dependency. Consider deferred event delivery until all modules initialized

### RISK-09: SQLite WAL Concurrent Access

- **Boundary:** `hub.db` is read by engine snapshot collector while hub is writing; `audit.db` is separate
- **What crosses:** SQLite read/write queries
- **What can go wrong:** WAL mode handles concurrent reads well, but long-running engine reads during heavy hub writes could see transient lock contention
- **Test coverage:** **None** — concurrency not tested (all tests use in-memory or single-connection)
- **Failure mode:** **Unlikely** — WAL mode is designed for this, but under extreme load could see `database is locked` errors
- **Suggested mitigation:** Low risk. Engine's SQLite fallback read should use `timeout=5` on connection
- **Status: MITIGATED (closes #34):** `AuditLogger._batch_insert()` now retries up to 3 times on `sqlite3.OperationalError` (locked, busy, I/O) with exponential backoff (0.5s, 1s, 2s). On final failure, failed events are written to `~/ha-logs/intelligence/audit_dead_letter.jsonl` instead of being silently discarded. Non-OperationalError exceptions propagate immediately. See `aria/hub/audit.py` and `tests/hub/test_audit.py::TestBatchInsertRetry`.

### RISK-10: Pipeline Sankey Topology Drift

- **Boundary:** `aria/dashboard/spa/src/lib/pipelineGraph.js` defines `ALL_NODES`, `LINKS`, `NODE_DETAIL` — must match actual module topology
- **What crosses:** Module names, connections, data flow paths
- **What can go wrong:** New module added to hub but not to Sankey → dashboard shows incomplete pipeline
- **Test coverage:** **None** — no automated check that Sankey matches module registry
- **Failure mode:** **Visual** — dashboard shows outdated pipeline visualization
- **Suggested mitigation:** Automated test that compares Sankey node list against registered module IDs
- **Resolved (2026-02-20):** Lean audit removed 4 modules (online_learner, transfer_engine, organic_discovery, activity_labeler) and renamed pattern_recognition → trajectory_classifier. Sankey topology overhauled in pipeline-sankey-accuracy-design — all nodes, links, and output columns now match actual module registry.

### RISK-11: Telegram Alert Failure is Silent

- **Boundary:** `watchdog.py` and `intelligence.py` both send Telegram messages independently
- **What crosses:** HTTP POST to Telegram Bot API
- **What can go wrong:** Token revoked, rate limit hit, network down — the alerting channel itself fails
- **Test coverage:** **Mocked** — all tests mock HTTP calls
- **Failure mode:** **Silent** — logged to journalctl only, no secondary notification channel
- **Suggested mitigation:** Watchdog should verify Telegram connectivity on startup and log a prominent warning if unreachable. Consider a local fallback (write alert to a file that `ttyd` or dashboard can display)

### RISK-12: Config Changes Not Propagated Immediately — MITIGATED

- **Boundary:** `PUT /api/config/{key}` writes to cache, then publishes `config_updated` on the event bus
- **What crosses:** Config values (thresholds, intervals, feature flags)
- **Mitigation (Issue #24):** `PUT /api/config/{key}` now publishes `config_updated` immediately after saving. `IntelligenceHub.initialize()` subscribes a dispatcher that calls `module.on_config_updated(config)` on every registered module. `Module` base class provides a no-op default so modules that do not need live reloading require no changes. Modules override `on_config_updated()` to react.
- **Propagation path:** `PUT /api/config/{key}` → `hub.publish("config_updated", {key, value})` → `IntelligenceHub.on_config_updated()` → each `module.on_config_updated(config)`
- **Test coverage:** `tests/hub/test_config_propagation.py` (13 tests: base class contract, hub dispatch, event bus integration, API layer)
- **Remaining consideration:** Modules must explicitly override `on_config_updated()` to pick up the new value. The base no-op is intentional — not all config keys have runtime effect.

### RISK-13: Cold-Start Module Ordering — Entity Classification Reads Empty Discovery

- **Boundary:** `discovery` (#1) runs `_reclassify_all()` on startup via a scheduled task, reading `entities` cache populated by its own initial HA scan — which is also async
- **What crosses:** Entity list from cache (written by the same module's async WebSocket + REST scan)
- **What can go wrong:** Discovery's `_reclassify_all` task may fire before the initial HA scan completes, operating on an empty or stale cache
- **Test coverage:** **None** — tests provide pre-populated cache, never test empty-cache startup
- **Failure mode:** **Silent** — entity classification produces nothing on cold boot, activity_monitor falls back to domain filtering. Self-corrects on next 24h cycle.
- **Partially mitigated (2026-02-19):** Merging data_quality into discovery eliminates the inter-module race (one module now owns both the data and the classification). The intra-module race (task ordering within discovery) remains.
- **Suggested mitigation:** discovery could defer `_reclassify_all` until `_initial_discovery` task completes, e.g. by awaiting a `discovery_initial_complete` internal signal

---

## Cross-Project Dependencies

Data flows beyond the ARIA boundary:

| External System | Connection | ARIA Role | Direction |
|----------------|-----------|-----------|-----------|
| Home Assistant REST API | HTTP (`<ha-host>:8123`) | Consumer | Read |
| Home Assistant WebSocket | WS (`<ha-host>:8123/api/websocket`) | Consumer (2 connections) | Read |
| Frigate NVR | MQTT (`<mqtt-broker-ip>:1883`) | Consumer | Read |
| Ollama Queue | HTTP (`127.0.0.1:7683`) | Consumer | Read/Write |
| Telegram Bot API | HTTP | Producer (via watchdog, intelligence digest) | Write |
| ha-log-sync | Subprocess/timer | Producer → `~/ha-logs/logbook/` | Engine reads |
| Preact Dashboard | HTTP/WS (`127.0.0.1:8001`) | Producer | Serve |

### Shared File Directories

| Directory | Writer | Reader | Format |
|-----------|--------|--------|--------|
| `~/ha-logs/logbook/` | `aria sync-logs` (every 15min) | engine collectors | JSONL |
| `~/ha-logs/intelligence/daily/` | engine `aria snapshot` | intelligence module | JSON |
| `~/ha-logs/intelligence/intraday/` | engine `aria snapshot-intraday` | intelligence module | JSON |
| `~/ha-logs/intelligence/models/` | engine `aria retrain` | ml_engine module | .joblib |
| `~/ha-logs/intelligence/cache/hub.db` | hub (runtime) | engine (fallback read) | SQLite |
| `~/ha-logs/intelligence/cache/audit.db` | audit_logger | (query only) | SQLite |
| `~/ha-logs/watchdog/aria-watchdog.log` | watchdog | (monitoring) | text |
| `~/ha-logs/intelligence/snapshot_log.jsonl` | engine snapshot | (append-only log) | JSONL |
