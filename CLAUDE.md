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
**Closed-loop feedback design:** `docs/plans/2026-02-15-closed-loop-feedback-design.md`
**Watchdog:** `aria/watchdog.py` — health monitoring, Telegram alerts, auto-restart

## Running

**Service:** `aria-hub.service` (user systemd)
**CLI:** `aria` (installed via `pip install -e .`, see `pyproject.toml`)
**API:** `http://127.0.0.1:8001` (localhost only)
**Dashboard:** `http://127.0.0.1:8001/ui/`
**WebSocket:** `ws://127.0.0.1:8001/ws` (real-time cache updates to dashboard)
**Watchdog:** `aria-watchdog.timer` (every 5 min) — logs to `~/ha-logs/watchdog/aria-watchdog.log`, Telegram alerts on failures
**Env vars:** HA_URL, HA_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from `~/.env` (bash wrapper pattern, not EnvironmentFile=)
**MQTT:** Mosquitto on 192.168.1.35:1883 (core_mosquitto addon on HA Pi) — credentials in `config_defaults.py` presence section. Required for camera-based presence signals via Frigate.

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

### CLI Commands

23 commands via `aria` entry point — full reference: `docs/cli-reference.md`

## Architecture

Single `aria.*` package: `hub/` (real-time core + 12 modules), `engine/` (batch ML, 7 subpackages), `modules/` (discovery through presence), `dashboard/` (Preact SPA). Full layout, module registry, cache categories: `docs/architecture-detailed.md`. Closed-loop feedback system connects ML accuracy, shadow hit rates, automation acceptance, drift signals, and activity label corrections back into the learning pipeline.

All imports use `aria.*` namespace — e.g. `from aria.hub.core import IntelligenceHub`

### Dashboard

Preact SPA at `aria/dashboard/spa/`. Must rebuild after JSX changes: `cd aria/dashboard/spa && npm run build`. Design language: `docs/design-language.md`. Build & CSS rules: `docs/dashboard-build.md`

### Activity Monitor

4 prediction analytics methods per 15-min window. Reference: `docs/activity-monitor-reference.md`

## Testing

### Pipeline Verification (after deployment or feature changes)

ARIA has the deepest pipeline — engine→JSON files→hub cache→API→WebSocket→dashboard. Unit tests cover each layer but not the flow between them. After any deployment or feature change, run dual-axis tests:

**Horizontal:** Hit every API endpoint (`/api/cache/{category}` for all 8 categories, `/api/cache/presence`, `/api/shadow/accuracy`, `/api/pipeline`, `/api/ml/*`, `/api/capabilities/*`, `/api/config`, `/api/curation/summary`, `/api/capabilities/feedback/health`, `/api/activity/current`, `/api/activity/labels`, `/api/activity/stats`, `/api/automations/feedback`). Confirm each returns expected shape with real data.

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

**Memory warning:** The full suite (~1111 tests) can consume 4-8G RAM. If concurrent agents or services are running, check `free -h` first. If available memory < 4G, run by suite instead of the full set. Shadow engine tests previously consumed 17G+ RAM due to mock objects returning None in tight loops — those are fixed, but watch for regressions.

```bash
# All tests (~1111) — use timeout to catch hangs
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
.venv/bin/python -m pytest tests/hub/ -k "feedback" -v      # Closed-loop feedback
.venv/bin/python -m pytest tests/hub/ -k "activity_labeler" -v  # Activity labeler
.venv/bin/python -m pytest tests/hub/test_presence.py -v       # Presence (51 tests)
.venv/bin/python -m pytest tests/hub/test_watchdog.py -v       # Watchdog (31 tests)
```

## Environment

- **HA instance:** 192.168.1.35:8123 (HAOS on Raspberry Pi, 3,065 entities, 10 capabilities)
- **Env vars:** HA_URL, HA_TOKEN from `~/.env`
- **Python:** 3.12 via `.venv/` (aiohttp, scikit-learn, numpy, uvicorn, fastapi)
- **Package config:** `pyproject.toml` (replaces old `requirements.txt`)
- **Node:** esbuild for SPA bundling (dev dependency only)
- **Cache DB:** `~/ha-logs/intelligence/cache/hub.db` (SQLite)
- **Snapshot log:** `~/ha-logs/intelligence/snapshot_log.jsonl` (append-only JSONL)

## WebSocket & Entity Filtering

Two HA WebSocket connections (discovery + activity), entity curation layer, domain filtering fallback. Reference: `docs/websocket-and-filtering.md`

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
- **Activity labeler Ollama dependency** — Uses ollama-queue (port 7683) for LLM-based activity predictions every 15 min. May queue-stack with organic discovery on Sundays (4:00 AM). Monitor queue depth if both run concurrently.
- **Activity labeler feature vector is 8 features** — If adding features, update `_context_to_features()`, adjust tests, and the classifier invalidation check in `initialize()` will auto-reset incompatible cached classifiers.
- **Intelligence features default to 0** — If intelligence cache is empty or engine hasn't run, correlated_entities_active, anomaly_nearby, active_appliance_count all default to 0. This is safe but means early predictions use only the base 5 sensor features.
- **Feedback data requires service restart** — New backend code writes to cache but the running service uses old code until `systemctl --user restart aria-hub`. Always restart + vertical trace after deploying feedback changes.
- **Engine conftest collision** — `from conftest import X` resolves to the wrong conftest in full-suite runs. Use `importlib.util.spec_from_file_location` for explicit file-based conftest imports. Tests that pass in isolation may break in full-suite runs due to implicit pytest conftest namespacing.
- **Presence cache access goes through hub** — Use `await self.hub.set_cache()` and `await self.hub.get_cache()`, NOT `self.hub.cache.set_cache()`. CacheManager doesn't expose set_cache directly; it goes through IntelligenceHub.
- **Frigate Docker required for camera presence** — Frigate container at `~/frigate/` must be running for camera-based presence signals. Without it, only HA sensor signals (motion, lights, dimmers) feed into presence.

## Reference Docs

- `docs/cli-reference.md` — All 22 CLI commands
- `docs/architecture-detailed.md` — Package layout, module registry, cache categories
- `docs/dashboard-build.md` — SPA build, CSS rules, design language
- `docs/activity-monitor-reference.md` — Prediction analytics methods
- `docs/websocket-and-filtering.md` — WebSocket connections, entity filtering, API endpoints
- `docs/api-reference.md` — Full API curl examples (pre-existing)
- `docs/design-language.md` — Dashboard design system (pre-existing)
- `docs/dashboard-components.md` — Component reference (pre-existing)
- `docs/plans/2026-02-15-closed-loop-feedback-design.md` — Feedback channels, activity labeler, DemandSignal
- `docs/plans/2026-02-15-closed-loop-feedback-implementation.md` — Implementation plan, wave dispatch
- `docs/plans/2026-02-15-presence-detection-design.md` — Presence detection design (Frigate + HA sensors + BayesianOccupancy)
