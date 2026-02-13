# ARIA Architecture

## Package Structure

```
aria/
  engine/           Batch ML pipeline (runs via cron)
    collectors/     15 entity-type collectors (power, climate, etc.)
    analysis/       Baselines, anomalies, correlations, drift detection
    features/       Time encoding, feature vector construction
    models/         sklearn, Prophet, IsolationForest model wrappers
    predictions/    Predictor engine + scoring
    llm/            Ollama reports, meta-learning, automation YAML
    storage/        JSON I/O + data store abstraction

  hub/              Real-time service infrastructure
    core.py         IntelligenceHub — module registry, task scheduler, event bus
    cache.py        SQLite-backed category cache (hub.db)
    api.py          FastAPI routes + WebSocket for dashboard
    constants.py    Shared cache key constants
    config_defaults.py  Default config parameter seeding

  modules/          Hub runtime modules (registered in order)
    discovery.py    HA entity/device/area scanning
    ml_engine.py    Real-time ML (GradientBoosting, RF, LightGBM)
    patterns.py     Recurring event sequence detection
    orchestrator.py Automation suggestion generation
    shadow_engine.py Predict-compare-score validation loop
    data_quality.py Entity classification pipeline (auto-exclude, edge, include)
    intelligence.py Unified intelligence assembly
    activity_monitor.py WebSocket state_changed listener

  dashboard/        Preact SPA (esbuild-bundled)
    spa/src/        JSX source files
    static/         Built assets
    templates/      Jinja2 shell

  cli.py            Unified CLI entry point
```

## Runtime Modes

| Mode | Entry | What Runs |
|------|-------|-----------|
| Batch | `aria full`, `aria retrain`, etc. | Engine only. Runs, writes data, exits. |
| Service | `aria serve` | Hub + all modules. Stays running, WebSocket, dashboard. |

## Data Flow

1. **Engine** (cron) collects HA state, trains models, writes predictions to `~/ha-logs/intelligence/`
2. **Hub** (service) reads engine outputs, monitors real-time activity via WebSocket
3. **Intelligence module** assembles engine results + live data into unified cache
4. **Dashboard** reads cache via REST API + receives live updates via WebSocket
