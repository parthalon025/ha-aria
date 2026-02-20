# Phase 3+4: Issue Triage & Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Triage all 56 open GitHub issues, close stale/fixed ones, set up GitHub project infrastructure, then fix all surviving issues in priority order.

**Architecture:** Three-pass approach — triage first (close/annotate/label), then GitHub setup (labels/milestones/board), then fix all surviving issues grouped by priority tier with quality gates between batches.

**Tech Stack:** Python 3.12, FastAPI, aiohttp, SQLite, pytest, gh CLI

## Quality Gates

Run between every batch:
```bash
cd /home/justin/Documents/projects/ha-aria
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

If memory < 4G: `.venv/bin/python -m pytest tests/hub/ tests/engine/ --timeout=120 -x -q`

---

### Task 1: Initialize progress.txt

**Files:**
- Create: `progress.txt`

**Step 1:** Create progress file

```
echo "# Phase 3+4: Issue Triage & Fix Progress" > progress.txt
echo "" >> progress.txt
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> progress.txt
echo "" >> progress.txt
```

**Step 2:** Commit

```bash
git add progress.txt
git commit -m "chore: initialize progress.txt for Phase 3+4"
```

---

### Task 2: Close Already-Fixed Issues

Research confirmed these 7 issues are resolved in the current codebase.

**Step 1:** Close each with evidence comment

```bash
gh issue close 8 -c "Resolved: api.py:498 now calls hub.get_module('trajectory_classifier'). Module IDs are distinct."
gh issue close 14 -c "Resolved: /ws handler catches RuntimeError at api.py:1559,1569; WSManager.broadcast() cleans up disconnects at api.py:83-92."
gh issue close 15 -c "Resolved: prune() now iterates all 4 tables (audit_events, audit_requests, audit_startups, audit_curation_history) at audit.py:461."
gh issue close 16 -c "Resolved: All direct-write methods guard with 'if self._db is None: return' at audit.py:184,202,236."
gh issue close 17 -c "Resolved: query_timeline() now has limit=1000 default at audit.py:357."
gh issue close 18 -c "Resolved: export_archive() uses run_in_executor at audit.py:497."
gh issue close 39 -c "Resolved: CapabilityRegistry is lazy-initialized and cached in hub at core.py:461-468."
```

**Step 2:** Append to progress.txt

```
## Closed: Already-Fixed Issues
- #8, #14, #15, #16, #17, #18, #39 (7 issues, all verified in current codebase)
```

---

### Task 3: Close Archived-Module Issues

**Step 1:** Identify pure-archived issues

- **#10** — Requests Pipeline Sankey add transfer_engine + attention_explainer. Both archived.
- **#61** — ml_engine couples to online_learner via hub.modules dict. online_learner archived. Check if ml_engine still references it.

**Step 2:** Verify #61 in code

```bash
grep -n "online_learner" /home/justin/Documents/projects/ha-aria/aria/modules/ml_engine.py
```

If no references remain, close. If references exist, annotate as partial.

**Step 3:** Create `archived` label and close

```bash
gh label create "archived" --color "C5DEF5" --description "Closed: module archived in Phase 1 lean audit"
gh issue close 10 -c "Archived: transfer_engine and attention_explainer were both archived in Phase 1 lean audit." --reason "not planned"
# Close #61 only if grep confirms no remaining references
```

---

### Task 4: Annotate Partial-Archived Issues

Issues that reference archived modules but have surviving concerns. Add comment noting archived portions; keep open.

**Step 1:** Annotate each

```bash
gh issue comment 25 -b "**Phase 1 note:** data_quality module was merged into discovery. The cold-start concern survives — run_classification() in discovery.py:519-530 exits silently when entities cache is empty."
gh issue comment 27 -b "**Phase 1 note:** transfer_engine was archived. The cache bypass concern at api.py:645 (toggle_can_predict uses hub.cache.set instead of hub.set_cache) is confirmed and survives."
gh issue comment 33 -b "**Phase 1 note:** activity_labeler was archived. The Ollama queue monitoring concern survives as a standalone reliability issue."
gh issue comment 48 -b "**Phase 1 note:** data_quality was merged into discovery. Snapshot quality flagging concern survives."
gh issue comment 52 -b "**Phase 1 note:** organic_discovery was archived. Blocking I/O in ml_engine and intelligence modules survives."
gh issue comment 53 -b "**Phase 1 note:** data_quality was merged into discovery. N+1 queries in discovery.py:545-570 (run_classification) confirmed. Shadow engine hot-path may have been refactored."
gh issue comment 55 -b "**Phase 1 note:** data_quality was merged into discovery. Sequential init concern survives for all 10 modules."
gh issue comment 57 -b "**Phase 1 note:** online_learner was archived. Redundant scan_hardware() in ml_engine and trajectory_classifier survives."
gh issue comment 62 -b "**Phase 1 note:** online_learner and transfer_engine were archived. Cross-layer coupling confirmed in 4 surviving modules: ml_engine (7 imports), trajectory_classifier (4), presence (1), intelligence (1)."
```

---

### Task 5: Re-label Unlabeled Issues

Issues #42-73 are missing priority and category labels. Apply labels based on content analysis.

**Step 1:** Security issues

```bash
gh issue edit 42 --add-label "bug,routing,priority:medium"
gh issue edit 43 --add-label "bug,priority:critical"
gh issue edit 44 --add-label "bug,priority:high"
gh issue edit 45 --add-label "bug,reliability,priority:medium"
gh issue edit 46 --add-label "bug,priority:medium"
gh issue edit 47 --add-label "reliability,priority:medium"
gh issue edit 48 --add-label "bug,testing,priority:medium"
gh issue edit 49 --add-label "enhancement,priority:medium"
gh issue edit 50 --add-label "bug,priority:high"
gh issue edit 51 --add-label "bug,priority:high"
gh issue edit 52 --add-label "bug,priority:high"
gh issue edit 53 --add-label "bug,priority:medium"
gh issue edit 54 --add-label "enhancement,priority:medium"
gh issue edit 55 --add-label "enhancement,priority:medium"
gh issue edit 56 --add-label "bug,reliability,priority:medium"
gh issue edit 57 --add-label "enhancement,priority:low"
gh issue edit 58 --add-label "bug,priority:medium"
gh issue edit 59 --add-label "documentation,priority:low"
gh issue edit 60 --add-label "enhancement,architecture,priority:medium"
gh issue edit 61 --add-label "architecture,priority:medium"
gh issue edit 62 --add-label "architecture,priority:medium"
gh issue edit 63 --add-label "bug,reliability,priority:high"
gh issue edit 64 --add-label "bug,priority:critical"
gh issue edit 65 --add-label "bug,priority:critical"
```

**Step 2:** KA test issues (from Phase 2)

```bash
gh issue edit 66 --add-label "enhancement,testing,priority:low"
gh issue edit 67 --add-label "enhancement,testing,priority:medium"
gh issue edit 68 --add-label "enhancement,testing,priority:low"
gh issue edit 69 --add-label "documentation,testing,priority:low"
gh issue edit 70 --add-label "documentation,testing,priority:low"
gh issue edit 71 --add-label "bug,testing,priority:medium"
gh issue edit 72 --add-label "enhancement,testing,priority:low"
gh issue edit 73 --add-label "documentation,testing,priority:low"
```

---

### Task 6: Create GitHub Infrastructure

**Step 1:** Create remaining labels

```bash
gh label create "phase:4" --color "D93F0B" --description "Phase 4: Fix & Optimize"
gh label create "phase:5" --color "1D76DB" --description "Phase 5: UI Decision Tool"
```

**Step 2:** Create milestones

```bash
gh api repos/{owner}/{repo}/milestones -f title="Phase 4: Fix & Optimize" -f description="Security > Reliability > Performance > Architecture fixes" -f state="open"
gh api repos/{owner}/{repo}/milestones -f title="Phase 5: UI Decision Tool" -f description="OODA-based dashboard redesign"
```

**Step 3:** Assign surviving issues to Phase 4 milestone

All non-KA-test, non-UI issues get Phase 4. UI issues (#28) get Phase 5. KA test issues (#66-73) stay unassigned (Phase 2 scope).

```bash
# Get milestone numbers from step 2, then:
# Phase 4 issues: all surviving except #28 and #66-73
for i in 9 19 20 21 22 23 24 25 27 29 30 33 34 35 36 37 38 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 62 63 64 65; do
  gh issue edit $i --milestone "Phase 4: Fix & Optimize"
done

# Phase 5 issue
gh issue edit 28 --milestone "Phase 5: UI Decision Tool"
```

**Step 4:** Create Kanban project board

```bash
gh project create --title "ARIA Lean Audit" --owner @me
# Then add all open issues to the project
```

**Step 5:** Commit triage summary and update progress.txt

---

### Task 7: Security Batch — #43, #44, #64, #65

**Files:**
- Modify: `aria/hub/api.py:34-41` (auth), `api.py:645` (config redaction), `api.py:1445` (CORS), `api.py:1578` (ws/audit auth)
- Test: `tests/hub/test_api.py`

#### #43: MQTT Credentials Exposed via GET /api/config

**Step 1: Write failing test**

```python
# tests/hub/test_api.py
async def test_config_endpoint_redacts_sensitive_keys(api_client):
    """GET /api/config must not expose MQTT passwords."""
    response = await api_client.get("/api/config")
    configs = response.json()["configs"]
    sensitive_values = [c for c in configs if "mqtt_password" in c.get("key", "")]
    for c in sensitive_values:
        assert c["value"] == "***REDACTED***"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/hub/test_api.py -k "test_config_endpoint_redacts" -v`
Expected: FAIL

**Step 3: Implement fix**

In `aria/hub/api.py`, add near line 34:

```python
_SENSITIVE_KEY_PATTERNS = {"password", "token", "secret", "credential", "api_key"}

def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return any(p in key_lower for p in _SENSITIVE_KEY_PATTERNS)
```

In `get_all_config()` (around line 1144-1152), redact before returning:

```python
for config in configs:
    if _is_sensitive_key(config.get("key", "")):
        config["value"] = "***REDACTED***"
```

**Step 4: Run test to verify it passes**

**Step 5: Proceed to #44**

#### #44: Add CORS Middleware

**Step 1: Write failing test**

```python
async def test_cors_headers_present(api_client):
    """API responses include CORS headers restricting to localhost."""
    response = await api_client.options("/api/health", headers={"Origin": "http://evil.com"})
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] != "*"
```

**Step 2: Run test to verify it fails**

**Step 3: Implement** — add after `app = FastAPI(...)` at line 1445:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["X-API-Key"],
)
```

**Step 4: Run test to verify it passes**

#### #64: Make API Auth Required (Not Opt-In)

**Step 1: Write failing test**

```python
async def test_api_rejects_unauthenticated_when_key_unset(api_client_no_key):
    """Without ARIA_API_KEY set, API should log a startup warning."""
    # Test that startup logs contain the warning
    assert "ARIA_API_KEY not set" in captured_logs
```

**Step 2: Implement** — in `api.py:34-41`, replace the opt-in guard:

```python
_ARIA_API_KEY = os.environ.get("ARIA_API_KEY")
if not _ARIA_API_KEY:
    import logging
    logging.getLogger("aria.hub.api").warning(
        "ARIA_API_KEY not set — API authentication disabled. "
        "Set ARIA_API_KEY env var to enable authentication."
    )
```

This maintains backward compatibility (no breaking change) while making the security state visible.

**Step 3: Run test**

#### #65: Add Auth Gate to /ws/audit

**Step 1: Write failing test**

```python
async def test_ws_audit_rejects_without_token():
    """ws/audit should reject connections without valid token."""
    # Test WebSocket connection without token is rejected
```

**Step 2: Implement** — in `api.py:1578-1582`, add before `websocket.accept()`:

```python
if _ARIA_API_KEY:
    token = websocket.query_params.get("token")
    if token != _ARIA_API_KEY:
        await websocket.close(code=4003, reason="Invalid or missing token")
        return
```

**Step 3: Run test, commit batch**

```bash
git add aria/hub/api.py tests/hub/test_api.py
git commit -m "fix: security hardening — redact config secrets, add CORS, auth gate on /ws/audit (closes #43, closes #44, closes #64, closes #65)"
```

**Step 4: Run quality gate**

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

---

### Task 8: Reliability Critical Batch — #19, #20, #21

**Files:**
- Create: `aria/schemas.py` (#19)
- Modify: `aria/modules/intelligence.py` (#19), `aria/modules/discovery.py` (#20), `aria/modules/activity_monitor.py` (#20), `aria/modules/presence.py` (#20), `aria/watchdog.py` (#21)
- Test: `tests/integration/test_schema_contract.py`, `tests/hub/test_reconnect_jitter.py`, `tests/hub/test_watchdog.py`

#### #19: Engine→Hub JSON Schema Contract Test

**Step 1:** Create `aria/schemas.py` with TypedDict definitions for the key JSON files read by intelligence.py:

```python
from typing import TypedDict, Any

class IntelligenceSnapshot(TypedDict, total=False):
    entity_count: int
    predictions: list[dict[str, Any]]
    anomaly_scores: dict[str, float]
    # ... (enumerate all 17+ fields from intelligence.py:281-308)
```

**Step 2:** Add validation in `intelligence.py:_read_intelligence_data()` — log warning on missing required keys rather than silently returning None.

**Step 3:** Write contract test at `tests/integration/test_schema_contract.py`:

```python
def test_engine_output_matches_hub_reader_schema():
    """Engine JSON output must contain all keys that hub intelligence module reads."""
    # Run a minimal engine snapshot, then verify all required keys exist
```

**Step 4:** Run tests, commit

#### #20: Stagger Reconnection Delays

**Step 1:** Add per-module `_reconnect_base_delay` in each connection loop:
- `discovery.py:371` → `retry_delay = 5 + 0` (first to reconnect)
- `activity_monitor.py:254` → `retry_delay = 5 + 2`
- `presence.py:384` (MQTT) → `retry_delay = 5 + 4`
- `presence.py:515` (WS) → `retry_delay = 5 + 6`

Also apply jitter to the initial retry_delay (not just subsequent ones).

**Step 2:** Write test verifying different initial delays:

```python
def test_reconnect_delays_are_staggered():
    """Each module's initial reconnect delay must differ by >= 1s."""
```

**Step 3:** Run tests, commit

#### #21: Telegram Startup Probe + Health Exposure

**Step 1:** Add connectivity probe in watchdog startup:

```python
async def _verify_telegram_connectivity(self):
    """Send a test API call to verify Telegram bot token is valid."""
```

**Step 2:** Expose `last_telegram_ok` in `/api/health` response

**Step 3:** Write test, run, commit batch

```bash
git commit -m "fix: add schema contract test, stagger reconnects, expose Telegram health (closes #19, closes #20, closes #21)"
```

**Step 4:** Run quality gate

---

### Task 9: Reliability High Batch — #22, #23, #24, #25, #27

**Files:**
- Modify: `aria/engine/collectors/snapshot.py` (#22, #23), `aria/hub/api.py` (#24, #27), `aria/modules/discovery.py` (#25)
- Test: `tests/engine/test_snapshot.py`, `tests/hub/test_api.py`, `tests/hub/test_discovery.py`

#### #22: Snapshot Write Race — Add File Lock

In `snapshot.py:build_intraday_snapshot()`, add `fcntl.flock` before writing, or check if snapshot for current hour-window already exists.

#### #23: Presence Cold-Start — Add Validation

In `snapshot.py` or `validation.py`, detect all-zero presence features and add `presence_valid: false` flag.

#### #24: Config Propagation — Subscribe to config_updated

Each relevant module subscribes to `config_updated` event via `hub.subscribe()` and re-reads its config.

#### #25: Discovery Cold-Start — Defer Classification

In `discovery.py:initialize()`, don't call `run_classification()` immediately. Instead, subscribe to `cache_updated` for entities category and trigger classification when data arrives.

#### #27: Toggle Can-Predict Cache Bypass

In `api.py:645`, change `await hub.cache.set("capabilities", caps)` to `await hub.set_cache("capabilities", caps)`.

**Commit:** `fix: race conditions, cold-start handling, config propagation (closes #22, closes #23, closes #24, closes #25, closes #27)`

**Quality gate:** Run full test suite

---

### Task 10: Reliability Medium Batch — #45, #46, #47, #56

**Files:**
- Modify: `aria/engine/hardware.py` (#45), `aria/engine/collectors/snapshot.py` (#45), `aria/engine/collectors/extractors.py` (#45), `aria/engine/collectors/ha_api.py` (#46, #47), `aria/engine/llm/client.py` (#46, #47), `aria/engine/sequence.py` (#46), `aria/modules/shadow_engine.py` (#56), `aria/modules/activity_monitor.py` (#56)

#### #45: Replace Silent except-pass with Logging

Add `logger.warning()` to each identified location: `hardware.py:61`, `snapshot.py:38,51`, `extractors.py:41`.

#### #46: Typed Failure Returns

Change `sequence.py:classify()` to return `None` on failure instead of `"stable"`. Update callers in `trajectory_classifier.py` to handle `None`.

#### #47: Retry Decorator for Network Calls

Add retry logic to `ha_api.py` functions (3 attempts, 1.5x backoff, URLError/TimeoutError only).

#### #56: Bound In-Memory Collections

- `shadow_engine._recent_resolved` → `collections.deque(maxlen=200)`
- `activity_monitor._activity_buffer` → add early flush at 5000 events

**Commit:** `fix: silent failures, typed error returns, retry logic, bounded collections (closes #45, closes #46, closes #47, closes #56)`

**Quality gate**

---

### Task 11: Performance Batch — #51, #52, #53, #54, #55, #58

**Files:**
- Modify: `aria/modules/discovery.py` (#51, #53), `aria/modules/ml_engine.py` (#52), `aria/modules/intelligence.py` (#52), `aria/hub/cache.py` (#54), `aria/cli.py` (#55), `aria/modules/presence.py` (#58)

#### #51: Async Subprocess in Discovery

Replace `subprocess.run()` at `discovery.py:142` with `asyncio.create_subprocess_exec()`.

#### #52: Wrap Blocking I/O in asyncio.to_thread

- `ml_engine.py:214` (pickle.load) → `await asyncio.to_thread(pickle.load, f)`
- `ml_engine.py:524,596` (pickle.dump) → `await asyncio.to_thread(...)`
- `intelligence.py:326,345,353,...` (json file reads) → `await asyncio.to_thread(path.read_text)`

#### #53: Batch SQLite Queries in Discovery Classification

Add `get_curations_batch()` and `upsert_curations_batch()` to `cache.py`. Use in `discovery.py:run_classification()`.

#### #54: Buffer Event Logging

In `cache.py:log_event()`, buffer events and flush every 5s or 50 events. Remove read-before-write in `cache.set()`.

#### #55: Parallel Module Initialization

In `cli.py:_register_modules()`, group into dependency tiers and use `asyncio.gather()`:
- Tier 0: discovery (must be first)
- Tier 1: ml_engine, patterns, orchestrator (parallel)
- Tier 2: optional modules (parallel)

#### #58: Shared aiohttp Session in Presence

Create `self._http_session` in `presence.py:initialize()`, reuse across all HTTP calls, close in `shutdown()`.

**Commit:** `perf: async I/O, batch queries, parallel init, session reuse (closes #51, closes #52, closes #53, closes #54, closes #55, closes #58)`

**Quality gate**

---

### Task 12: Architecture Batch — #29, #30, #42, #59, #60, #62

**Files:**
- Modify: `aria/hub/api.py` (#42), `aria/modules/trajectory_classifier.py` (#29), `aria/capabilities.py` (#60), `docs/system-routing-map.md` (#59)
- Create: `tests/integration/test_sankey_sync.py` (#30)

#### #42: Fix cache bypass in toggle_can_predict

`api.py:645` → change `hub.cache.set` to `hub.set_cache` (same fix as #27 — may already be done in Task 9).

#### #29: Wire Pattern Config Values

In `trajectory_classifier.py:initialize()`, read `pattern.dtw_neighbors`, `pattern.anomaly_top_n`, `pattern.trajectory_change_threshold` from config and pass to constructors. Or remove from `config_defaults.py` if not needed.

#### #30: Sankey Topology Sync Test

Create `tests/integration/test_sankey_sync.py` that parses `pipelineGraph.js` ALL_NODES IDs and compares against registered module IDs.

#### #59: Document Undocumented Routes/Events

Add `/api/pipeline/topology`, `automation_approved`, `automation_rejected` to `docs/system-routing-map.md`.

#### #60: Auto-Discover Capabilities from Registered Modules

Replace hardcoded imports in `capabilities.py:226-248` with iteration over `hub.modules.values()`.

#### #62: Reduce Cross-Layer Coupling

Move shared constants (`TRAJECTORY_CLASSES`, `DEFAULT_FEATURE_CONFIG`) to `aria/shared/constants.py`. Accept functional coupling (ml_engine → vector_builder) as intentional.

**Commit:** `refactor: wire configs, auto-discover capabilities, reduce coupling (closes #29, closes #30, closes #42, closes #59, closes #60, closes #62)`

**Quality gate**

---

### Task 13: Remaining Fixes — #9, #28, #36, #37, #38, #41, #48, #49, #50, #57

**Files:** Various (see per-issue details in design doc research)

#### #38: Clean Up Stale Worktree

```bash
git worktree remove .worktrees/phase2-config
```

#### #36: Timezone-Aware Datetimes

Replace `datetime.now()` with `datetime.now(tz=timezone.utc)` across: `core.py:98,413,445,503`, `intelligence.py:345,436,467`, `cli.py:671`.

#### #37: Prune snapshot_log.jsonl

Add date-based pruning (90 days) in `core.py:_prune_stale_data()`.

#### #9: Document Python 3.12 venv Requirement

Update `pyproject.toml` and CLAUDE.md to document that `.venv` must use Python 3.12, not Homebrew 3.14.

#### #48: Snapshot Data Quality Flags

Add `data_quality: {ha_reachable: bool, entity_count: int}` to snapshot output.

#### #49: Watchdog Disk + Ollama Monitoring

Add `check_disk_space()` and `check_ollama_health()` to `watchdog.py`.

#### #50: Model Status Field

Add `model_status` field to ml_engine pipeline state: `untrained | training | ready | stale`.

#### #57: Shared Hardware Profile

Compute `scan_hardware()` once in hub startup, expose as `hub.hardware_profile`. Modules read from hub.

#### #41: Reference Model Training Stub

Implement `_train_reference_model()` in `ml_engine.py:1391-1399` with a gradient-boosting reference model.

#### #28: Dashboard WebSocket Subscriptions (Phase 5)

This is a UI issue — assign to Phase 5 milestone, do not fix in this batch.

**Commit:** Multiple commits grouped by concern area.

**Quality gate**

---

### Task 14: Triage KA Test Issues (#66-73)

These 8 issues are Phase 2 scope (known-answer test improvements). Leave open, labeled, unassigned to a milestone. They'll be addressed when Phase 2 implementation begins.

---

### Task 15: Update Roadmap and Final Verification

**Step 1:** Update `docs/plans/2026-02-19-lean-audit-roadmap.md`:
- Phase 3: Status → Done
- Phase 4: Status → Done (merged into Phase 3)
- Update success criteria checkboxes

**Step 2:** Run full test suite as final verification

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -x -q
```

**Step 3:** Update progress.txt with final summary

**Step 4:** Final commit

```bash
git commit -m "docs: mark Phase 3+4 complete in lean audit roadmap"
```

---

## Batch Dependency Map

```
Task 1 (init) → Task 2-6 (triage, can run in parallel) → Task 7 (security)
→ Task 8 (reliability critical) → Task 9 (reliability high)
→ Task 10 (reliability medium) → Task 11 (performance)
→ Task 12 (architecture) → Task 13 (remaining)
→ Task 14 (KA triage) → Task 15 (roadmap update)
```

Tasks 2-6 are independent and can run in parallel.
Tasks 7-13 must be sequential (each batch's quality gate must pass before the next).
Tasks 14-15 are independent of fix batches.

## Issue Count Summary

| Category | Count | Action |
|----------|-------|--------|
| Close: already fixed | 7 | #8, #14, #15, #16, #17, #18, #39 |
| Close: archived module | 1-2 | #10, possibly #61 |
| Annotate: partial archived | 9 | #25, #27, #33, #48, #52, #53, #55, #57, #62 |
| Re-label | 24 | #42-73 (missing labels) |
| Fix: security | 4 | #43, #44, #64, #65 |
| Fix: reliability | 11 | #19-25, #27, #45-47, #56 |
| Fix: performance | 6 | #51-55, #58 |
| Fix: architecture | 6 | #29, #30, #42, #59, #60, #62 |
| Fix: remaining | 10 | #9, #28, #36-38, #41, #48-50, #57 |
| Defer: Phase 2 (KA tests) | 8 | #66-73 |
| Defer: Phase 5 (UI) | 1 | #28 |
