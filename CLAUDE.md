# ARIA — Adaptive Residence Intelligence Architecture

Unified intelligence platform for Home Assistant — batch ML engine, real-time activity monitoring, predictive analytics, and an interactive Preact dashboard in a single `aria.*` package.

**Repo:** https://github.com/parthalon025/ha-aria (private)

## Context

**Unified from:** `ha-intelligence` (batch ML engine) + `ha-intelligence-hub` (real-time dashboard), now in one repo.
**All 52 design/implementation plans completed** — archived in `docs/plans/archive/`.
**Lean audit roadmap (Phase 5 complete):** `docs/plans/archive/2026-02-19-lean-audit-roadmap.md`
**Watchdog:** `aria/watchdog.py` — health monitoring, Telegram alerts, auto-restart

## Running

**Service:** `aria-hub.service` (user systemd)
**CLI:** `aria` (installed via `pip install -e .`, see `pyproject.toml`)
**API:** `http://127.0.0.1:8001` (localhost only)
**Dashboard:** `http://127.0.0.1:8001/ui/`
**WebSocket:** `ws://127.0.0.1:8001/ws` (real-time cache updates to dashboard)
**Watchdog:** `aria-watchdog.timer` (every 5 min) — logs to `~/ha-logs/watchdog/aria-watchdog.log`, Telegram alerts on failures
**Env vars:** HA_URL, HA_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from `~/.env` (bash wrapper pattern, not EnvironmentFile=)
**MQTT:** Mosquitto on `<mqtt-broker-ip>:1883` (core_mosquitto addon on HA Pi) — credentials in `config_defaults.py` presence section. Required for camera-based presence signals via Frigate.

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

21 commands via `aria` entry point — full reference: `docs/cli-reference.md`

**Pre-flight health check:** `bin/check-ha-health.sh` — validates HA connectivity + core stats before batch timers run. Used by all snapshot/training systemd timers.

## Quick Routing

**"How does X connect to Y?"** → Read `docs/system-routing-map.md` (lookup tables, data flow diagrams, seam risk catalog).

All imports use `aria.*` namespace — e.g. `from aria.hub.core import IntelligenceHub`. Dashboard: rebuild SPA after JSX changes (`cd aria/dashboard/spa && npm run build`).

## Testing

### Pipeline Verification (after deployment or feature changes)

ARIA has the deepest pipeline — engine→JSON→hub cache→API→WebSocket→dashboard. Full method: `projects/CLAUDE.md` § Pipeline Verification. Full endpoint list: `docs/system-routing-map.md` § HTTP Route Table.

**Vertical trace:** `aria snapshot-intraday` → JSON file → hub cache → GET /api/cache/intelligence → WebSocket push → dashboard render.

### Unit Tests

**Parallel execution:** Tests run across 6 workers via pytest-xdist (`-n 6` in pyproject.toml). Use `-n 0` to disable parallelism for debugging. Memory warning: 6 workers can consume 6-12G RAM total. If available memory < 8G, use `-n 2` or `-n 0` instead.

```bash
# All tests (~1584) — parallel across 6 workers
.venv/bin/python -m pytest tests/ -v --timeout=120

# Single-worker mode (for debugging test isolation issues)
.venv/bin/python -m pytest tests/ -v --timeout=120 -n 0

# By suite (safer when memory-constrained)
.venv/bin/python -m pytest tests/hub/ -v         # Hub (~1533 tests)
.venv/bin/python -m pytest tests/engine/ -v       # Engine (~485 tests)
.venv/bin/python -m pytest tests/integration/ -v  # Integration (~237 tests, includes known-answer)

# By feature area (use -k for keyword filtering)
.venv/bin/python -m pytest tests/hub/ -k "shadow" -v        # Shadow mode
.venv/bin/python -m pytest tests/hub/ -k "activity" -v      # Activity monitor
.venv/bin/python -m pytest tests/hub/ -k "feedback" -v      # Closed-loop feedback
.venv/bin/python -m pytest tests/hub/test_presence.py -v       # Presence (51 tests)
.venv/bin/python -m pytest tests/hub/test_watchdog.py -v       # Watchdog (31 tests)

# Known-answer tests (deterministic golden-snapshot regression suite)
.venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120  # 37 tests, all 10 modules
.venv/bin/python -m pytest tests/integration/known_answer/ --update-golden    # Re-baseline golden files
```

## Environment

- **HA instance:** `<ha-host>` (HAOS on Raspberry Pi, 3,065 entities, 10 capabilities)
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

**Lessons learned:** `lessons-db search "area entity resolution"` | `lessons-db search "organic discovery"`

## Gotchas

- **Entity area_id is device-inherited** — only 6/3,050 entities have direct area_id. Always resolve entity→device→area. See "HA Data Model" above.
- **Collector registration requires extractors import** — Without `import aria.engine.collectors.extractors`, the registry is empty and all snapshot metrics are 0.
- **patterns.py uses standard LLM client** — `ollama_chat(prompt, OllamaConfig(model="qwen2.5:7b"))` via `asyncio.to_thread`, not raw `ollama.generate()`. Routes through ollama-queue.
- **sequence_anomalies.py severity uses negative log-probabilities** — `threshold * 1.5` is MORE negative (stricter). See comment at line 171.
- **All imports use `aria.*` namespace** — e.g. `from aria.hub.core import IntelligenceHub`
- **Cache access goes through hub** — Use `await self.hub.set_cache()`/`get_cache()`, NOT `self.hub.cache.*`. CacheManager doesn't expose set_cache directly.
- **Engine↔Hub JSON schema coupling** — Changes to engine JSON output require matching updates in `_read_intelligence_data()` in `aria/modules/intelligence.py`.
- **Capabilities cache is extended, not replaced** — Organic discovery adds fields; seed capabilities always preserved.
- **Entity discovery uses lifecycle merge** — Missing entities marked stale, archived after 72h, auto-promote on rediscovery.
- **Feedback/code changes require restart** — `systemctl --user restart aria-hub` + vertical trace after deploying.
- **Cold boot: predictions empty for 24-48h** — Need data accumulation before meaningful scores.
- **Pipeline Sankey topology must stay accurate** — Update `ALL_NODES`/`LINKS`/`NODE_DETAIL` in `src/lib/pipelineGraph.js` when adding/removing modules. Sankey now lives on Home page, not its own route.
- **OODA nav is the canonical structure (Phase 5)** — Home/Observe/Understand/Decide are the 4 primary destinations. Old routes (intelligence, predictions, patterns, shadow, automations, presence) redirect. Do not add new top-level routes without an OODA fit — use the System section instead.
- **Presence module uses domain filter + entity room cache** — Filters state_changed events to `light.`, `binary_sensor.`, `media_player.`, `event.` domains upfront. O(1) local entity→room lookup prevents slow subscriber warnings. Falls back to hub cache for entities added since last cache build.
- **Presence module seeds on startup** — `_seed_presence_from_ha()` processes person.*, light.*, binary_sensor.*, media_player.*, event.* entities on boot, not just person.*, fixing cold-start room detection (e.g., TV on, lights on at startup now detected immediately).
- **Media player signals feed presence detection** — playing/paused/idle/buffering states contribute 0.85 media_active signal; off/standby contribute 0.15 media_inactive. Enables living room presence detection via Sonos/Apple TV state without requiring a room camera.
- **SUPERHOT mobile overrides use `!important`** — `index.css` forces all `superhot-ui` effects on phone/tablet. If `superhot-ui` renames classes, overrides silently break.
- **SPA catch-all route** — `/ui/{path}` returns `index.html` for non-file paths, enabling deep-linking and browser refresh. Real static files served directly.
- **hub.db events table is for hub management events only** — `hub.publish()` routes through `cache.log_event()`, but high-frequency HA telemetry (`state_changed`, ~6/sec × 3050 entities) must NOT be logged here — it already lives in events.db (EventStore). Writing it to hub.db accumulates 3.6M rows/week and causes a 540 MB/day RSS leak. See `_HIGH_FREQ_SKIP_EVENTS` in `aria/hub/cache.py`. (#140)
- **SQLite DELETE does not free memory — VACUUM does** — `prune_events()` must call `PRAGMA wal_checkpoint(TRUNCATE)` then `VACUUM` after deleting rows, with all cursors explicitly closed first. Without this, freed pages stay in the SQLite page cache and process RSS keeps growing even after retention pruning. (#140)

Full gotchas catalog: `docs/gotchas.md`

## ARIA Anti-Patterns (Lessons Derived)

- **Subscribe lifecycle:** `initialize()` not `__init__()`, store callback ref on `self`, match with `unsubscribe()` in `shutdown()` (#28, #37)
- **Event firehose:** Domain filter (`str.startswith()`) before any async work (#39)
- **Module IDs:** Grep for duplicates before adding a new module — last writer wins silently (#31)
- **Hub cache API:** `hub.set_cache()`/`get_cache()` only — never access `hub.cache.*` directly (#12)
- **Feature alignment:** Single source for ML feature lists; contract test between training and inference (#45)
- **Pipeline wiring:** Next batch must wire previous batch's components; explicit integration task in plans with 3+ batches (#53)
- **Dead config keys:** Every `register_config()` must have a matching `get_config_value()` — dead knobs lie to operators (#54)
- **HA automation keys:** Code reading HA automation dicts must check both singular and plural keys: `get("triggers") or get("trigger", [])` (#55)
- **A/B verification:** Run bottom-up + top-down verification after any 3+ batch implementation — finds non-overlapping bug classes (#56)
- **Entity resolution:** Always resolve entity→device→area chain — only 0.2% have direct area_id (#1)
- **Feature matrix:** Alphabetical column order — classifier reads column 0, not "first inserted" (#27)
- **SVG/Canvas:** Never nest Canvas in SVG — use native SVG elements (#19)

## Reference Docs

- **`docs/system-routing-map.md`** — **Start here for any routing/connection question.** Topic→file index, HTTP routes, event bus, MQTT, timers, cache owners, subprocess calls, data flow diagrams, seam risk catalog
- `docs/architecture-detailed.md` — Package layout, module registry, cache categories
- `docs/cli-reference.md` — All 21 CLI commands
- `docs/api-reference.md` — Full API curl examples
- `docs/websocket-and-filtering.md` — WebSocket connections, entity filtering
- `docs/dashboard-build.md` — SPA build, CSS rules
- `docs/design-language.md` — Dashboard design system
- `docs/activity-monitor-reference.md` — Prediction analytics methods
- `docs/dashboard-components.md` — Component reference

## Design System Usage

**Full guide:** `docs/llm-guide-design-system.md` (~750 lines) — LLM reference for applying the design system to the ARIA dashboard.

**Before building any UI:** Read `docs/llm-guide-design-system.md`. Follow §1.5 Strategy Stack (Outcome-Driven + Trust & Predictability + Context-Aware + Trust-Centered + Feedback-Rich). Behavioral target: confident monitoring with low anxiety.

Pipeline: ui-template (base) → expedition33-ui (theme) → ha-aria (consumer). Key mappings:
- **Entity health** → GlyphBadge (gustave=healthy, enemy=offline)
- **Occupancy/battery** → StatBar (HP=occupancy, AP=battery)
- **Power/metrics** → MonolithDisplay (gustave gold numeral)
- **Anomalies** → HUDFrame + CrossingOut + dread mood
- **Presence** → PortraitFrame with per-person chroma
- **OODA phases:** Observe=lune/wonder, Orient=lune/continent, Decide=gustave/dawn, Act=maelle/paint effects
- **Pages:** Home=lumiere/nostalgic, Understand=continent/wonder, Detail:anomaly=wasteland/dread, ML Engine=monolith/nostalgic

## Code Factory

## Scope Tags
language:python, framework:preact, framework:pytest, domain:ha-aria

Quality gates for agent-driven development (auto-triggered via superpowers integration in `~/Documents/CLAUDE.md`):
- **Quality checks**: `python3 -m pytest --timeout=120 -x -q`
- **PRD artifacts**: `tasks/prd.json`, `tasks/prd-<feature>.md`
- **Progress log**: `progress.txt` (append-only during execution)

## Code Quality
- Lint: `make lint`
- Format: `make format`

## Quality Gates
- Before committing: `/verify`
- Before PRs: `lessons-db scan --target . --baseline HEAD`

## Lessons
- Check before planning: `/check-lessons`
- Capture after bugs: `/capture-lesson`
- Lessons: `lessons-db search` to query, `lessons-db capture` to add. DB is authoritative — never write lesson .md files directly.

## Local AI Review
- Code review: `ollama-code-review .`

## Semantic Search
- Generate: `bash scripts/generate-embeddings.sh`
- Storage: `.embeddings/` (gitignored)
