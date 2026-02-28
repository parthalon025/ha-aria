# Agent Teams: Fix All GitHub Issues — Design
**Date:** 2026-02-28
**Scope:** All 182 open GitHub issues in ha-aria
**Predecessor:** `docs/plans/2026-02-25-fix-all-issues-plan.md` (100 issues, sequential agents)
**AAR:** `docs/plans/2026-02-26-fix-all-issues-aar.md`

---

## North Star

**ARIA's purpose:** Intelligent home automation through ML-driven predictions, real-time activity monitoring, presence detection, and automation generation.

Every agent decision in this pipeline must be anchored to this goal. When a fix choice is ambiguous, agents ask:

> "Does this decision preserve or improve ARIA's ability to learn from home data, make accurate predictions, and generate reliable automations?"

A bug that silently corrupts training snapshots outranks a UI cosmetic issue. A fix that trades ML accuracy for code cleanliness is wrong. A phantom issue that, on inspection, reveals a latent risk to the intelligence pipeline is promoted to confirmed.

---

## Context

| Metric | Value |
|--------|-------|
| Open issues | 182 (#80–#332) |
| Issue breakdown | 104 bugs, 28 tech-debt, 43 other (feature/phase tasks), 2 CI, 1 security, 1 perf, 3 test gaps |
| Test baseline | ~2,384 passing, 17 skipped, 2,401 collected |
| Previous run | Feb 25: ~72/100 fixed, +180 tests, sequential sub-agents |
| Conventions doc | `docs/conventions-fix-all-issues.md` (6 patterns, needs update) |
| Existing worktree | None (previous `fix/all-100-issues` branch already merged) |

---

## Architecture Overview

```
Stage 0: Parallel Triage (4 × Explore agents — read-only, simultaneous)
  → tasks/issues-engine.json
  → tasks/issues-hub-core.json
  → tasks/issues-hub-modules.json
  → tasks/issues-frontend.json
  → tasks/feature-backlog.json        ← feature/phase tasks parked here, not fixed
           ↓
Stage 1: Conventions Update (1 × general-purpose agent)
  → docs/conventions-fix-all-issues.md  (adds patterns G–K for new issue types)
           ↓
Stage 2: Four Parallel Fix Teams (each in isolated git worktree)
  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐
  │ Engine Team      │  │ Hub-Core Team     │  │ Hub-Modules Team   │  │ Frontend Team    │
  │ python-expert    │  │ python-expert     │  │ python-expert      │  │ general-purpose  │
  │ fix/engine       │  │ fix/hub-core      │  │ fix/hub-modules    │  │ fix/frontend     │
  │ ~50 issues       │  │ ~35 issues        │  │ ~40 issues         │  │ ~57 issues       │
  │ Internal batches │  │ Internal batches  │  │ Internal batches   │  │ bash-expert sub  │
  │ of ≤10, KA probe │  │ of ≤10, KA probe  │  │ of ≤10, KA probe   │  │ for CI/shell     │
  └─────────────────┘  └──────────────────┘  └────────────────────┘  └──────────────────┘
           ↓ (all four teams complete)
Stage 3: Judge Round (3 parallel specialized agents + 1 coordinator)
  ├─ pr-review-toolkit:silent-failure-hunter  → tasks/judge-silent-failures.md
  ├─ pr-review-toolkit:pr-test-analyzer       → tasks/judge-test-coverage.md
  ├─ superpowers:code-reviewer                → tasks/judge-completeness.md
  └─ general-purpose coordinator:
       runs full pytest (-n 2) + npm build
       collects 3 reports → tasks/judge-round-1.md (PASS or FAIL + rework list)
           ↓ PASS
Stage 4: Merge Sequence
  fix/engine → fix/hub-core → fix/hub-modules → fix/frontend → fix/all-issues-r1
  (check tasks/interface-changes.md at each step)
           ↓
Stage 5: Final Audit (integration-tester)
  Horizontal sweep (all API routes) + Vertical trace (snapshot → cache → WS → dashboard)
           ↓ PASS
Stage 6: Close Issues + PR
  Auto-closed via "closes #NNN" in commits; manual gh issue close for remainder
```

---

## Stage 0: Parallel Triage

**Agent type:** `Explore` (×4, simultaneous)
**Why:** Read-only (no Edit/Write tools) — cannot accidentally modify code during verification. Fast Glob/Grep/Read for codebase scanning.

### Each triage agent's protocol

```
For each issue in my domain:
  1. Read the issue title + body
  2. Locate the referenced file(s) in the codebase
  3. Search for the described bug pattern (grep for the specific code path)
  4. Classify:
     - confirmed: file exists, bug pattern found, not fixed in recent commits
     - pre-existing-fix: commit history shows fix already merged (grep recent commits)
     - phantom: file or component does not exist (renamed, merged, deleted)
     - feature-backlog: issue describes new functionality, not a defect
  5. For confirmed issues: assign priority
     - critical: data loss, silent corruption of ML training data, service crash
     - high: incorrect predictions, broken presence detection, automation generation failures
     - medium: error handling gaps, missing tests, API shape mismatches
     - low: code quality, dead config, cosmetic

NORTH STAR FILTER: When borderline between confirmed and phantom, ask:
"Does the described failure degrade ARIA's ability to learn from home data,
detect presence, or generate automations?" If yes, classify as confirmed.
```

### Output schema (tasks/issues-{domain}.json)

```json
{
  "domain": "engine",
  "generated": "2026-02-28T...",
  "baseline_tests": 2384,
  "issues": [
    {
      "number": 303,
      "title": "perf: ml_engine.py reads 60+ JSON files synchronously",
      "status": "confirmed",
      "priority": "high",
      "files": ["aria/modules/ml_engine.py"],
      "pattern": "asyncio.to_thread — Convention I",
      "aria_impact": "blocks event loop during training, delays real-time activity monitoring"
    }
  ],
  "feature_backlog": [88, 89, 90, 91, 92]
}
```

### Domain scopes for triage agents

| Agent | Issues to scan | Domain files |
|-------|---------------|-------------|
| Triage-Engine | All issues mentioning: engine, model, collector, snapshot, analysis, feature, prediction, training, autoencoder, prophet, gradient, neural, reliability, correlation, anomaly, baseline, sequence, explainability, llm, automation, ha-log-sync, extractors, data_store, storage, faces, face recognition | `aria/engine/`, `aria/automation/`, `aria/faces/` |
| Triage-Hub-Core | All issues mentioning: hub, api.py, websocket, cache, discovery, routes, cors, auth, api key, config PUT | `aria/hub/api.py`, `aria/hub/core.py`, `aria/hub/cache.py`, `aria/hub/routes_*.py` |
| Triage-Hub-Modules | All issues mentioning: module, presence, shadow, unifi, ml_engine (module), patterns, activity_monitor, orchestrator, shared, event_store, ha_automation_sync, day_classifier, watchdog, constants | `aria/modules/`, `aria/shared/`, `aria/watchdog.py` |
| Triage-Frontend | All issues mentioning: dashboard, spa, jsx, js:, pipelinegraph, timechart, sankey, npm, frontend, shellcheck, ci:, lint-fix, github actions, shell script, stabilization-smoke | `aria/dashboard/spa/`, `.github/workflows/`, `bin/*.sh` |

---

## Stage 1: Conventions Update

**Agent type:** `general-purpose`
**Input:** existing `docs/conventions-fix-all-issues.md` + all 4 `tasks/issues-*.json`
**Output:** updated `docs/conventions-fix-all-issues.md` with patterns G–K

### New conventions to add

**Convention G — asyncio.get_event_loop() replacement**
- Pattern: any `asyncio.get_event_loop()` call in async context or in code called from async context
- Fix: use `asyncio.get_running_loop()` if inside a coroutine; `asyncio.run()` for entry points
- Applies to: #306 and any similar patterns found during triage

**Convention H — Typed string enums (Literal)**
- Pattern: bare `str` type annotation where only a fixed set of values is valid
- Fix: `from typing import Literal; level: Literal["warning", "critical", "info"]`
- Applies to: #326 (WatchdogResult.level), #323 (event bus on_event types)

**Convention I — asyncio.to_thread() for blocking I/O**
- Pattern: `open()`, `json.load()`, `pickle.load()`, sqlite3 calls inside `async def` methods
- Fix: `result = await asyncio.to_thread(sync_function, *args)`
- Applies to: #303 (ml_engine JSON reads), #248 (faces pipeline SQLite), any similar

**Convention J — AbortController for Preact fetch cleanup**
- Pattern: fetch in useEffect without cleanup (causes state update on unmounted component)
- Fix: `const controller = new AbortController(); fetch(url, {signal: controller.signal}); return () => controller.abort();`
- Applies to: Frontend fetch races, double-fetch issues

**Convention K — Issue auto-close in commits**
- Every fix commit message must end with: `closes #NNN` (or `closes #NNN, closes #MMM`)
- Enables automatic GitHub issue closure on PR merge
- No exceptions — if a commit fixes a confirmed issue, the close reference is required

**Commit:**
```bash
git add docs/conventions-fix-all-issues.md
git commit -m "docs: update fix-all conventions with patterns G-K for new issue types"
```

---

## Stage 2: Four Parallel Fix Teams

### Team-level setup (each team)

```bash
# Create worktree (from main, after Stage 1 commit)
git worktree add ../ha-aria-fix-{domain} -b fix/{domain}
cd ../ha-aria-fix-{domain}

# Initialize progress log
echo "# Progress: {domain} team" > tasks/progress-{domain}.txt
echo "Started: $(date -Iseconds)" >> tasks/progress-{domain}.txt
echo "Baseline: $(cd ~/Documents/projects/ha-aria && .venv/bin/python -m pytest tests/ --co -q 2>/dev/null | tail -1)" >> tasks/progress-{domain}.txt

# Create interface changes log (shared across teams)
touch tasks/interface-changes.md
```

### Internal batch loop (all teams follow this exactly)

```
LOOP:
  1. Read tasks/issues-{domain}.json
  2. Filter: status=confirmed, not yet fixed
  3. Sort: critical → high → medium → low
  4. Take next ≤10 issues

  FOR EACH ISSUE:
    a. NORTH STAR CHECK: state ARIA impact before touching any file
       "This bug [description] affects ARIA by [specific impact on ML/presence/automations]"
    b. ECHO-BACK GATE (Lesson #83): state exact diagnosis
       "The bug is: [specific code path]. The fix is: [specific change]. Convention: [letter]."
    c. Write failing test first (TDD) — name: test_{description}_closes_{number}
    d. Verify test fails on current code
    e. Apply fix per conventions doc
    f. Verify test passes
    g. Commit: "fix({domain}): {description} closes #{number}"
    h. For phantom/pre-existing: document in progress log, do not commit a "fix"

  AFTER BATCH:
    - Run known-answer suite: .venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120
    - If KA fails: fix before next batch (do not proceed)
    - If KA passes: append batch summary to tasks/progress-{domain}.txt
    - Check tasks/interface-changes.md — if another team changed a contract this team depends on, assess impact
    - Append any cross-domain contract changes made in this batch to tasks/interface-changes.md

  REPEAT until all confirmed issues processed
  SIGNAL COMPLETION: echo "DONE: $(date -Iseconds)" >> tasks/progress-{domain}.txt
```

**Memory-safe testing:** All pytest runs use `-n 2` (not default `-n 6`) while multiple worktrees may be active.

### Team A: Engine (`python-expert`)

**Domain:** `aria/engine/`, `aria/automation/`, `aria/faces/`
**Agent type rationale:** `python-expert` specializes in async discipline, resource lifecycle, type safety, and the HA/aiohttp/asyncio ecosystem. It extends `lesson-scanner`, giving built-in anti-pattern detection. Engine issues are predominantly async I/O (#303), silent returns (#321), and ML training correctness — exactly this agent's specialty.

**Priority issues to tackle first:**
- #303 — ml_engine reads 60+ JSON files synchronously (blocks event loop → delays real-time monitoring)
- #321 — LLM JSON parse failures return [] silently (breaks automation suggestions)
- #313 — non-atomic snapshot log prune (data loss risk for ML training data)
- #318 — ha-log-sync exits 0 on HA failure (monitoring blind to HA downtime)
- Any issues tagged `critical` or affecting snapshot integrity

**Interface changes to watch:** Any change to snapshot JSON schema must be appended to `tasks/interface-changes.md` — Hub-Core reads these files.

### Team B: Hub-Core (`python-expert`)

**Domain:** `aria/hub/api.py`, `aria/hub/core.py`, `aria/hub/cache.py`, `aria/hub/routes_*.py`
**Agent type rationale:** Hub-Core is FastAPI + aiohttp + WebSocket async patterns. The new issues (#325 unguarded WebSocket send, #315 direct cache.get() bypass, #317 config PUT missing event bus, #316 phantom cache keys) are all async discipline and API correctness — `python-expert`'s core strength.

**Priority issues to tackle first:**
- #325 — unguarded WebSocket send_json on dirty connect (service crash)
- #307 — 4 discovery API endpoints permanently 503 (module ID mismatch)
- #317 — config PUT missing event bus publish (settings changes don't propagate)
- #316 — phantom cache keys in /api/ml/* (anomaly_alerts always empty)
- #315 — 15x hub.cache.get() direct access bypassing metadata + WS notifications
- #314 — security: unauthenticated write endpoints

**Interface changes to watch:** Any change to cache key names or API response shapes must be appended to `tasks/interface-changes.md` — Frontend reads these.

### Team C: Hub-Modules (`python-expert`)

**Domain:** `aria/modules/`, `aria/shared/`, `aria/watchdog.py`
**Agent type rationale:** Modules are long-running async subscribers with lifecycle patterns (initialize/shutdown), event bus coordination, and shared utilities. `python-expert` with `lesson-scanner` extension catches the subscriber lifecycle anti-patterns (#37 in CLAUDE.md lessons) that recur here.

**Priority issues to tackle first:**
- #305 — presence tracking silently stops after first Frigate event (aware/naive datetime)
- #329 — shadow_engine queries DB before domain filter (unnecessary I/O on hot path)
- #320 — watchdog subprocess.run() doesn't check returncode (health check lies)
- #326 — WatchdogResult.level untyped str (silent COOLDOWN_SECONDS lookup failure)
- #319 — dead config knobs in activity_monitor and ml_engine

**Interface changes to watch:** Any change to event bus payload shapes or MQTT message formats must be appended to `tasks/interface-changes.md`.

### Team D: Frontend (`general-purpose` + `bash-expert` sub-agent for CI/shell)

**Domain:** `aria/dashboard/spa/`, `.github/workflows/`, `bin/*.sh`
**Agent type rationale:** Frontend issues span JSX/Preact, CI YAML, and shell scripts — too heterogeneous for a single specialist. `general-purpose` handles JSX + build. For the 6 CI/shell issues (#330 ShellCheck, #332 lint-fix.yml, #310 stabilization-smoke.sh eval injection, #327 Actions version inconsistency), the Frontend agent spawns a `bash-expert` sub-agent — which knows ShellCheck patterns, YAML linting, and shell injection prevention.

**Priority issues to tackle first:**
- #310 — stabilization-smoke.sh eval injection (security: arbitrary code execution)
- #322 — pipelineGraph.js Sankey node has no input links (misleading topology display)
- #332 — lint-fix.yml || true masks ruff crash (bot can push broken code to main)
- #330 — no ShellCheck in CI (shell regressions undetected)
- #327 — GitHub Actions version inconsistency

**CI/shell sub-agent trigger:**
```
For issues #310, #330, #332, #327, and any other .sh or .yml issues:
  spawn bash-expert sub-agent with:
    "Fix {issue description}. File: {path}. Check against ShellCheck and POSIX portability.
     No eval, no hardcoded paths. Output must pass: shellcheck {file}"
```

**Build gate after each batch:**
```bash
cd aria/dashboard/spa && npm run build 2>&1 | tail -5
# Must exit 0
```

---

## Stage 3: Judge Round

**Trigger:** All four teams write "DONE" to their progress log.

**Three judge agents run in parallel:**

### Judge A: Silent Failure Hunter (`pr-review-toolkit:silent-failure-hunter`)

Scans all Python files changed across all four worktrees:
```bash
git diff main --name-only | grep '\.py$'
```
Reports: file:line, failure type, severity (HIGH/MEDIUM). Output: `tasks/judge-silent-failures.md`.

**NORTH STAR filter:** Judge A prioritizes HIGH findings that affect data integrity (training snapshots, cache writes, event bus publishes) over cosmetic silent returns.

### Judge B: Test Analyzer (`pr-review-toolkit:pr-test-analyzer`)

Reviews all new test files against the issues they claim to fix:
- Does each fix have a corresponding test?
- Does the test name reference the issue number?
- Would the test have caught the bug on pre-fix code?

Output: `tasks/judge-test-coverage.md`.

### Judge C: Completeness Reviewer (`superpowers:code-reviewer`)

Reviews against:
1. All four `tasks/issues-*.json` files — every `confirmed` issue must have a fix commit
2. `docs/conventions-fix-all-issues.md` — every fix must follow its convention letter
3. `tasks/interface-changes.md` — cross-domain contract changes are consistent

Output: `tasks/judge-completeness.md`.

### Judge Coordinator (`general-purpose`)

After A, B, C complete:
```bash
# Full test suite (memory-safe, single worktree)
cd ~/Documents/projects/ha-aria-fix-engine  # or merged branch
.venv/bin/python -m pytest tests/ --timeout=120 -q -n 2 2>&1 | tail -10

# Frontend build
cd aria/dashboard/spa && npm run build 2>&1 | tail -5

# Collect three judge reports
cat tasks/judge-silent-failures.md tasks/judge-test-coverage.md tasks/judge-completeness.md
```

Produces `tasks/judge-round-1.md`:
```
JUDGE ROUND 1: [PASS|FAIL]
- Silent failures: [PASS|FAIL] — N HIGH findings
- Test coverage: [PASS|FAIL] — N issues without tests
- Completeness: [PASS|FAIL] — N confirmed issues not addressed
- Test suite: [PASS|FAIL] — NNNN passed / N failed
- Build: [PASS|FAIL]

Rework required: [team: issue list]
```

---

## Stage 4: Rework (Conditional)

If judge returns FAIL:
- Only flagged teams rework their specific issues
- Rework uses the same internal batch loop
- After rework: re-run only the relevant judge sub-agent(s), not all three
- Coordinator re-runs test suite

Maximum 2 rework rounds before escalating to human review.

---

## Stage 5: Merge Sequence

```bash
# Merge order: Engine → Hub-Core → Hub-Modules → Frontend
git checkout -b fix/all-issues-r1

git merge fix/engine --no-ff -m "merge: engine team fixes (closes batch)"
# Check tasks/interface-changes.md for Engine→Hub-Core contracts
git merge fix/hub-core --no-ff -m "merge: hub-core team fixes"
# Check for Hub-Core→Hub-Modules contracts
git merge fix/hub-modules --no-ff -m "merge: hub-modules team fixes"
# Check for Python→Frontend contracts (API shapes, cache keys)
git merge fix/frontend --no-ff -m "merge: frontend team fixes"
```

Any merge conflict: resolve in favor of the fix that better serves ARIA's intelligence pipeline, per north star.

---

## Stage 6: Final Audit (`integration-tester`)

`integration-tester` verifies data flows across service seams — the Cluster B bugs that pass unit tests but fail at handoffs.

**Horizontal sweep:** Every route in `docs/system-routing-map.md` HTTP Route Table. Confirms surface exists and responds correctly.

**Vertical trace:**
1. `aria snapshot-intraday` → snapshot JSON written to `~/ha-logs/intelligence/`
2. Snapshot JSON contains `presence` key + `time_features` key
3. Hub cache at `GET /api/cache/intelligence` reflects new snapshot
4. WebSocket pushes `cache_updated` event within 5s
5. Dashboard pages render without crash: Predictions, Correlations, Presence, Timeline
6. `aria train` (if data available) — training logs surface convergence warnings

Output: `tasks/audit-final.md` with per-step PASS/FAIL.

---

## Issue Close Automation

Issues are auto-closed via `closes #NNN` in commit messages when PR merges.

For any confirmed issues not yet auto-closed after merge:
```bash
# Get list of confirmed issues from triage files
python3 -c "
import json, glob
for f in glob.glob('tasks/issues-*.json'):
    data = json.load(open(f))
    for i in data['issues']:
        if i['status'] == 'confirmed':
            print(i['number'])
" | while read num; do
  gh issue close $num --comment "Fixed in PR #XXX — closed automatically" 2>/dev/null || true
done
```

---

## Feature Backlog

Issues tagged `feature-backlog` during triage are written to `tasks/feature-backlog.json`. They are not fixed in this run. After the PR merges, create a single tracking issue:

```bash
gh issue create \
  --title "Feature backlog: N phase-task issues deferred from fix-all-issues run" \
  --body "$(cat tasks/feature-backlog.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(chr(10).join(f"- #{i}" for i in d[\"numbers\"]))')"
```

---

## Success Criteria

| Metric | Baseline | Target |
|--------|----------|--------|
| Tests passing | ~2,384 | ≥ 2,384 (net-positive from new tests) |
| Open bug/tech-debt/CI issues | ~139 | ≤ 15 (phantom remainder + deferred) |
| npm build | clean | clean |
| Judge silent-failure HIGH findings | unknown | 0 |
| Final H+V audit | — | PASS |
| Feature backlog filed | 0 | 1 tracking issue |

---

## Risk Register

| Risk | Mitigation |
|------|-----------|
| Hub-Modules team has most issues (~40) | Internal batches of ≤10; KA probe after each |
| Parallel worktrees exhaust RAM during pytest | `-n 2` per team (not `-n 6`); teams run internal tests sequentially within their worktree |
| Engine changes snapshot schema, Hub-Core breaks | `tasks/interface-changes.md` checked at every merge step |
| ~28% of issues may be phantom (AAR finding) | Triage stage eliminates these before any fix agent starts |
| Frontend bash-expert sub-agent diverges from conventions | Frontend agent reviews sub-agent output before committing |
| Lesson #83: static analysis "bugs" may be correct code | Echo-back gate required before every fix |
| North star violated (fix trades ML accuracy for cleanliness) | Encoded into every agent prompt; judge checks convention compliance |

---

## Agent Prompt Templates

### Triage Agent Prompt (Explore)

```
You are the ARIA triage agent for the {domain} domain.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
When classifying borderline issues, ask: does this failure degrade ARIA's ability
to learn from home data, detect presence, or generate automations?

Your task: scan all GitHub issues in your domain and classify each one.

1. Read `tasks/issues-{domain}.json` for your domain's issue list (or use `gh issue list`)
2. For each issue:
   - Locate the referenced file in the codebase (use Glob/Grep)
   - Search for the described bug pattern
   - Classify: confirmed | pre-existing-fix | phantom | feature-backlog
   - For confirmed: assign priority (critical/high/medium/low) based on ARIA impact
3. Write output to `tasks/issues-{domain}.json` using the schema in the design doc

Do NOT modify any source files. Read only.
```

### Fix Team Prompt (python-expert / general-purpose)

```
You are the ARIA {domain} fix agent.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
Every fix decision must preserve or improve ARIA's core intelligence pipeline.
When choosing between two valid approaches, prefer the one that better serves
ARIA's goal: accurate learning from home data, reliable presence signals,
trustworthy automation suggestions.

Read before starting:
1. `docs/conventions-fix-all-issues.md` — all fixes must follow the convention for their type
2. `tasks/issues-{domain}.json` — your confirmed issue list, sorted by priority
3. `tasks/interface-changes.md` — cross-domain contracts already changed by other teams
4. `docs/system-routing-map.md` — understand the full data flow before changing any interface

Internal batch loop:
- ≤10 issues per batch
- For each issue: NORTH STAR CHECK → ECHO-BACK GATE → failing test → fix → passing test → commit
- After each batch: run known-answer suite (tests/integration/known_answer/)
- Append batch summary to tasks/progress-{domain}.txt
- Append any cross-domain contract changes to tasks/interface-changes.md

Memory-safe testing: always use `pytest -n 2` (not -n 6)

Signal completion: echo "DONE: $(date -Iseconds)" >> tasks/progress-{domain}.txt
```

### Judge Coordinator Prompt (general-purpose)

```
You are the ARIA judge coordinator for round {N}.

NORTH STAR: ARIA is an intelligent home automation platform. Prioritize findings
that affect ML training data integrity, presence detection accuracy, or automation
generation correctness over cosmetic or low-impact issues.

1. Confirm all four teams have written "DONE" to their progress logs
2. Launch three judge sub-agents in parallel:
   - pr-review-toolkit:silent-failure-hunter → tasks/judge-silent-failures.md
   - pr-review-toolkit:pr-test-analyzer → tasks/judge-test-coverage.md
   - superpowers:code-reviewer → tasks/judge-completeness.md
3. Run full test suite: .venv/bin/python -m pytest tests/ --timeout=120 -q -n 2
4. Run frontend build: cd aria/dashboard/spa && npm run build
5. Collect all results → tasks/judge-round-{N}.md
6. Issue verdict: PASS (proceed to merge) or FAIL (rework list per team)
```
