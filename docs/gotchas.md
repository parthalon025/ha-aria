# ARIA Gotchas — Extended Reference

Implementation details and less-frequently-encountered gotchas. The top-level CLAUDE.md has the most critical items; this file has the full catalog.

## Engine & Pipeline

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

## Organic Discovery

- **Organic discovery needs 15+ entities per group** — HDBSCAN won't cluster small groups. If discovery finds no organic capabilities, the entity count may be too low or data too homogeneous.
- **Organic discovery Ollama contention** — Sunday 4:00 AM run (~45 min) overlaps with suggest-automations at 4:30 AM. Move one timer if both use Ollama.
- **Capabilities cache is extended, not replaced** — Organic discovery adds fields (`source`, `usefulness`, `layer`, `status`, etc.) to existing capabilities cache. Seed capabilities always preserved.

## Activity Labeler

- **Activity labeler Ollama dependency** — Uses ollama-queue (port 7683) for LLM predictions every 15 min. May queue-stack with organic discovery on Sundays.
- **Activity labeler feature vector is 10 features** — presence_probability and occupied_room_count added to the original 8. Cached classifiers auto-invalidate on feature count mismatch. If adding features, update `_context_to_features()` in `aria/engine/activity_labeler/feature_config.py`.
- **Intelligence features default to 0** — If intelligence cache is empty or engine hasn't run, correlated_entities_active, anomaly_nearby, active_appliance_count default to 0. Safe but means early predictions use only base 5 sensor features.

## Presence

- **PresenceCollector reads from hub API first, falls back to SQLite** — Unlike other collectors, PresenceCollector calls `/api/cache/presence` (or hits `hub.db` directly if hub offline). Invoked separately from normal collector loop.
- **Frigate Docker required for camera presence** — Container at `~/frigate/` must be running for camera-based presence signals.
- **Camera-to-room mapping is discovery-driven** — No hard-coded camera list. Cameras found via `camera.*` entities, room resolved via device→area chain. Manual overrides via `presence.camera_rooms` config key.

## Pattern Recognition (Phase 3)

- **Pattern recognition is Tier 3+ only** — Self-gates on hardware tier via `scan_hardware()` / `recommend_tier()`. On Tier 2 or below, module initializes but sets `active = False` and ignores all events.
- **Trajectory classification uses heuristic until trained** — DTW classifier (tslearn) needs labeled training data. Before that, `label_window_heuristic()` classifies based on slope and coefficient of variation of the target column.
- **Anomaly explanations require feature_names** — `_run_anomaly_detection` returns empty explanations list if feature_names is None. The ML engine passes feature_names from `_get_feature_names()`.
- **Config values not yet wired** — The 4 `pattern.*` config entries in `config_defaults.py` (window_size, dtw_neighbors, anomaly_top_n, trajectory_change_threshold) are scaffolding — the module uses `DEFAULT_WINDOW_SIZE = 6` and hardcoded values. Wire these when the config UI lands.
- **tslearn optional dependency** — In `[project.optional-dependencies]` under `ml-extra`. Not installed by default `pip install -e .` — needs `pip install -e '.[ml-extra]'`.

## ML & Training

- **Snapshot validation rejects corrupt snapshots** — <100 entities or >50% unavailable → rejected before training.
- **ML training runs on startup if stale** — Hub checks last training timestamp; if >7 days, triggers retraining immediately (slower first startup).

## Deployment

- **Feedback data requires service restart** — New backend code writes to cache but running service uses old code until restart. Always restart + vertical trace after deploying.
- **Engine conftest collision** — `from conftest import X` resolves to wrong conftest in full-suite runs. Use `importlib.util.spec_from_file_location` for explicit imports.

## Dashboard

- **PageBanner expects `page=` not `title=`** — JSX silently drops unrecognized props. Always check component signature.
- **Entity discovery uses lifecycle merge** — Missing entities marked stale, archived after 72h. Auto-promote on rediscovery.
- **Pipeline Sankey topology must stay accurate** — `src/lib/pipelineGraph.js` contains `ALL_NODES`, `LINKS`, `NODE_DETAIL`. Update when adding/removing modules.
