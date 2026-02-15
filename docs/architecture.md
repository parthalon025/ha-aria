# ARIA Architecture

## In Plain English

ARIA watches your entire smart home — every light switch, door lock, thermostat, and motion sensor — and learns what "normal" looks like. Over time, it spots patterns ("the kitchen lights always come on at 6:30 AM on weekdays"), detects anomalies ("the garage door has been open for 3 hours, that's unusual"), and predicts what's about to happen next. It runs entirely on your own computer with no cloud dependency.

## Why This Exists

Home Assistant is great at controlling smart devices, but it doesn't *think* about them. It won't notice that your front door lock's battery is draining faster than usual, or that your energy usage spiked 40% this week, or that you always forget to close the garage when you leave on Tuesdays. ARIA adds a brain on top of Home Assistant — machine learning models that learn your household's rhythms, flag things that don't look right, and eventually suggest automations you didn't know you needed. It's the difference between a home that follows rules and a home that understands patterns.

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
