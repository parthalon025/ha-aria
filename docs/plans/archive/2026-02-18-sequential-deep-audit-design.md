# Sequential Deep Audit — All 29 Open Issues

**Date:** 2026-02-18
**Scope:** Fix and audit all open GitHub issues comprehensively
**Approach:** Sequential deep audit — one-by-one in priority order

## Test Baseline

- **1620 passed, 5 failed, 16 skipped** (179s)
- 5 failures: promote/archive API tests (2), capabilities integration (1), golden baseline regression (1), capabilities validation (1)

## Execution Order

Each issue: **Read → Verify → Fix → Test → Audit → Close**

### Wave 1 — Critical (3 issues)

| # | Title | Key Files | Risk |
|---|-------|-----------|------|
| 19 | Engine→Hub JSON schema — no contract test | `aria/engine/schema.py`, `aria/modules/intelligence.py`, `tests/integration/test_engine_hub_integration.py` | RISK-01 |
| 20 | WebSocket/MQTT connections reconnect simultaneously after HA restart | `aria/modules/discovery.py`, `aria/modules/presence.py`, `aria/modules/activity_monitor.py` | Thundering herd |
| 21 | Telegram alert failure — no fallback notification channel | `aria/watchdog.py` | RISK-11 |

### Wave 2 — High (6 issues)

| # | Title | Key Files | Risk |
|---|-------|-----------|------|
| 22 | Activity monitor + systemd timer race on snapshot writes | `aria/modules/activity_monitor.py`, `aria/engine/storage/data_store.py` | RISK-03 |
| 23 | Presence cache cold-start → zero-valued ML features | `aria/modules/presence.py`, `aria/engine/collectors/snapshot.py` | RISK-04 |
| 24 | Config changes via API not propagated to modules | `aria/hub/api.py`, `aria/hub/cache.py`, all modules reading config | RISK-12 |
| 25 | Cold-start: data_quality reads empty discovery cache | `aria/modules/data_quality.py`, `aria/modules/discovery.py` | RISK-13 |
| 26 | Feature vector dual-import: engine/hub divergent features | `aria/engine/features/`, `aria/modules/intelligence.py` | RISK-05 |
| 27 | Capability promote/archive bypasses hub.set_cache | `aria/hub/api.py` (promote/archive routes) | Fixed in 8576a34, verify |

### Wave 3 — Medium (8 issues)

| # | Title | Key Files |
|---|-------|-----------|
| 28 | Direct-fetch dashboard pages miss WebSocket updates | `aria/dashboard/spa/` |
| 29 | Pattern recognition config — hardcoded constants | `aria/modules/pattern_recognition.py` |
| 31 | No backpressure on event bus | `aria/hub/core.py` |
| 32 | Dual event propagation (subscribe + on_event) | `aria/hub/core.py` |
| 34 | Audit logger async buffer — no retry on SQLite errors | `aria/hub/audit.py` |
| 35 | Ollama timer contention Sunday AM | systemd timers |
| 39 | CapabilityRegistry recreated on every request | `aria/hub/core.py`, `aria/capabilities.py` |
| 40 | Discovery subprocess JSON — no schema validation | `aria/modules/organic_discovery/` |

### Wave 4 — Low/Untagged (12 issues)

| # | Title | Key Files |
|---|-------|-----------|
| 9 | Venv Python 3.14 — ml-extra deps incompatible | `.venv/`, `pyproject.toml` |
| 10 | Pipeline Sankey needs Phase 4 modules | `aria/dashboard/spa/src/lib/pipelineGraph.js` |
| 15 | AuditLogger.prune() skips startups/curation tables | `aria/hub/audit.py` |
| 16 | AuditLogger direct-write — no uninitialized guard | `aria/hub/audit.py` |
| 17 | query_timeline() no limit parameter | `aria/hub/audit.py` |
| 18 | export_archive() sync I/O in async method | `aria/hub/audit.py` |
| 36 | datetime.now() without timezone throughout | Multiple files |
| 37 | snapshot_log.jsonl grows unboundedly | `aria/engine/storage/data_store.py` |
| 38 | Stale worktree .worktrees/phase2-config/ | Filesystem cleanup |
| 41 | Reference model training is a stub | `aria/engine/models/` |

## Per-Issue Protocol

1. **Read** — Issue description, relevant source files, existing tests
2. **Verify** — Check if recent commits already addressed it
3. **Fix** — Implement fix if not done or only partially done
4. **Test** — Write/update tests covering the fix, run targeted suite
5. **Audit** — Review surrounding code for:
   - Silent failures (Cluster A)
   - Integration boundary gaps (Cluster B)
   - Cold-start vulnerabilities (Cluster C)
6. **Close** — Close GitHub issue with summary

## Session Plan

- **Session 1:** Wave 1 (critical) + Wave 2 (high) — commit per wave
- **Session 2+:** Wave 3 + Wave 4 — commit per wave
- `/clear` between waves if context heavy

## Success Criteria

- All 29 issues closed on GitHub with evidence
- 0 test failures
- Each area audited for related problems
- New issues discovered during audit filed separately

## Already-Fixed Issues (Verify + Close)

Recent commits addressed many issues. These need verification, not reimplementation:
- `8576a34`: #27 (capability audit), #20 (jitter), #21 (fallback), #22 (locking), #24 (config event), #40 (schema validation), #16 (audit guard)
- `fcb04da`: #14, #17, #18
- `a35cc79`: #37, #38
- `a015d18`: #30, #33
- `d36940a`: #39
- `a8d8c0a`: Patterns module refactor (may affect #29)

Many issues are likely "verify fix is complete + tests pass + close" rather than "implement from scratch."
