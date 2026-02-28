# Agent Teams: Fix All GitHub Issues — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Fix all applicable GitHub issues in ha-aria using four parallel domain fix teams in isolated git worktrees, with a triage-first protocol, specialized agent types per task, and a judge round using silent-failure, test-coverage, and completeness reviewers before merge.

**Architecture:** Four parallel fix teams (`fix/engine`, `fix/hub-core`, `fix/hub-modules`, `fix/frontend`) each run in an isolated git worktree after a read-only triage pass classifies all 182 issues. A judge coordinator runs three specialized reviewer sub-agents after all teams complete, then orchestrates a dependency-ordered merge. Final audit uses `integration-tester` for horizontal + vertical verification.

**North Star:** Every agent decision is anchored to ARIA's purpose — intelligent home automation through ML-driven predictions, real-time activity monitoring, presence detection, and automation generation. When any decision is ambiguous, ask: "Does this preserve or improve ARIA's ability to learn from home data, detect presence, and generate reliable automations?"

**Tech Stack:** Python 3.12 (`.venv`), FastAPI, aiohttp, asyncio, SQLite, Preact/JSX (esbuild), pytest 2,401 tests, pytest-xdist (`-n 2` during parallel execution), GitHub CLI (`gh`), git worktrees

**Agent Types Used:**
- `Explore` — triage (read-only, fast Glob/Grep/Read)
- `python-expert` — Engine, Hub-Core, Hub-Modules fix teams (async discipline, HA ecosystem)
- `general-purpose` — Frontend fix team, judge coordinator, conventions update
- `bash-expert` — CI/shell issues (spawned as sub-agent by Frontend team)
- `pr-review-toolkit:silent-failure-hunter` — judge silent failure audit
- `pr-review-toolkit:pr-test-analyzer` — judge test coverage review
- `superpowers:code-reviewer` — judge completeness review
- `integration-tester` — final horizontal + vertical audit

---

## Quality Gates

| After Stage | Gate | Pass Criteria |
|-------------|------|--------------|
| Triage | Review `tasks/issues-*.json` | Every issue has status + priority + aria_impact |
| Conventions | Read updated doc | Patterns G–K present, doc committed |
| Each team internal batch | Known-answer suite | `pytest tests/integration/known_answer/ -v` all pass |
| Each team completion | Domain tests | `pytest tests/{domain}/ -n 2 -q` no regressions |
| Judge round | Full suite + build | ≥2,384 passed, `npm run build` clean, 0 HIGH silent failures |
| Final audit | H+V | All API routes respond, vertical trace PASS |

---

## PRD References

Design doc: `docs/plans/2026-02-28-agent-teams-github-issues-design.md`
Predecessor plan: `docs/plans/2026-02-25-fix-all-issues-plan.md`
AAR: `docs/plans/2026-02-26-fix-all-issues-aar.md`
Conventions: `docs/conventions-fix-all-issues.md`
Routing map: `docs/system-routing-map.md`

---

## Task 0: Initialize Infrastructure

**Files:**
- Create: `tasks/issues-engine.json` (schema seed)
- Create: `tasks/issues-hub-core.json` (schema seed)
- Create: `tasks/issues-hub-modules.json` (schema seed)
- Create: `tasks/issues-frontend.json` (schema seed)
- Create: `tasks/feature-backlog.json` (schema seed)
- Create: `tasks/interface-changes.md`
- Create: `progress.txt`

**Step 1: Verify you are in the project root**

```bash
pwd
# Expected: /home/justin/Documents/projects/ha-aria
```

**Step 2: Record baseline test count**

```bash
.venv/bin/python -m pytest tests/ --co -q 2>/dev/null | tail -1
# Expected: 2401 tests collected in N.XXs
```

**Step 3: Initialize progress log**

```bash
echo "# ARIA Fix-All-Issues Agent Teams Progress" > progress.txt
echo "Started: $(date -Iseconds)" >> progress.txt
echo "Baseline: 2401 collected, ~2384 passing" >> progress.txt
echo "Open issues: 182 (#80–#332)" >> progress.txt
```

**Step 4: Create tasks coordination files**

```bash
cat > tasks/interface-changes.md << 'EOF'
# Interface Changes Log
# Append here when your fix changes a cross-domain contract.
# Format: [team] [issue #] [file:change] [impact on other teams]
# Example: [engine] #313 snapshot JSON: added "time_features" key → Hub-Core must handle this key
EOF
```

**Step 5: Seed JSON schema files**

```bash
for domain in engine hub-core hub-modules frontend; do
cat > tasks/issues-${domain}.json << EOF
{
  "domain": "${domain}",
  "generated": null,
  "baseline_tests": 2384,
  "issues": [],
  "feature_backlog": []
}
EOF
done

cat > tasks/feature-backlog.json << 'EOF'
{
  "note": "Populated by triage agents. Issues tagged feature-backlog are NOT fixed in this run.",
  "numbers": [],
  "tracking_issue": null
}
EOF
```

**Step 6: Commit infrastructure**

```bash
git add tasks/interface-changes.md tasks/issues-*.json tasks/feature-backlog.json progress.txt
git commit -m "chore: initialize agent-teams infrastructure files"
```

---

## Task 1: Create Four Worktrees

**Step 1: Create Engine worktree**

```bash
git worktree add ../ha-aria-fix-engine -b fix/engine
echo "Worktree fix/engine created" >> progress.txt
```

**Step 2: Create Hub-Core worktree**

```bash
git worktree add ../ha-aria-fix-hub-core -b fix/hub-core
echo "Worktree fix/hub-core created" >> progress.txt
```

**Step 3: Create Hub-Modules worktree**

```bash
git worktree add ../ha-aria-fix-hub-modules -b fix/hub-modules
echo "Worktree fix/hub-modules created" >> progress.txt
```

**Step 4: Create Frontend worktree**

```bash
git worktree add ../ha-aria-fix-frontend -b fix/frontend
echo "Worktree fix/frontend created" >> progress.txt
```

**Step 5: Verify all four worktrees exist**

```bash
git worktree list
# Expected: 5 entries (main + 4 fix branches)
```

---

## Task 2: Stage 0 — Parallel Triage (4 × Explore agents)

Launch all four triage agents simultaneously. Each is `subagent_type: Explore`, read-only. They write their output JSON files to `tasks/` in the main worktree.

**Step 1: Launch Triage-Engine agent**

Launch an `Explore` agent with this prompt:

```
You are the ARIA triage agent for the ENGINE domain.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
When classifying borderline issues, ask: does this failure degrade ARIA's ability
to learn from home data, detect presence, or generate automations? If yes, classify
as confirmed even if the code path seems minor.

Working directory: /home/justin/Documents/projects/ha-aria

YOUR SCOPE — only classify issues that touch these paths:
  aria/engine/         (models, collectors, features, analysis, predictions, storage, llm)
  aria/automation/     (LLM-driven automation builder)
  aria/faces/          (face recognition pipeline)

STEP 1: Get all open issues
  Run: gh issue list --limit 200 --json number,title,body,labels,state

STEP 2: Filter to your domain
  Include issues whose title or body references: engine, model, collector, snapshot,
  analysis, feature, prediction, training, autoencoder, prophet, gradient, neural,
  reliability, correlation, anomaly, baseline, sequence, explainability, llm,
  automation (the aria/automation/ module), ha-log-sync, extractors, data_store,
  storage, faces, face recognition, ml_engine (only aria/modules/ml_engine.py
  training-side issues — NOT hub-side)

STEP 3: For each issue in your domain, classify it:
  a. Locate the referenced file using Glob (e.g., Glob("aria/engine/**/*.py"))
  b. Grep for the specific function/pattern described in the issue
  c. Check git log for recent commits that may have already fixed it:
     Run: git log --oneline -50 | grep -i "closes #NNN\|fix.*NNN\|#NNN"
  d. Assign status:
     - confirmed: file exists, bug pattern found, not fixed in recent commits
     - pre-existing-fix: git log shows this was already fixed
     - phantom: file or function doesn't exist (renamed/deleted/merged)
     - feature-backlog: issue describes new functionality, not a defect
  e. For confirmed: assign ARIA-anchored priority:
     - critical: data loss, silent corruption of ML training snapshots, service crash
     - high: incorrect predictions, broken presence detection, automation generation failures,
             silent exception loss in the intelligence pipeline
     - medium: error handling gaps, missing tests, API shape mismatches
     - low: code quality, dead config, cosmetic

STEP 4: Write output to tasks/issues-engine.json using EXACTLY this schema:
{
  "domain": "engine",
  "generated": "ISO timestamp",
  "baseline_tests": 2384,
  "issues": [
    {
      "number": 303,
      "title": "perf: ml_engine.py reads 60+ JSON files synchronously",
      "status": "confirmed",
      "priority": "high",
      "files": ["aria/modules/ml_engine.py"],
      "pattern": "asyncio.to_thread — Convention I",
      "aria_impact": "blocks event loop during snapshot reads, delays real-time activity monitoring"
    }
  ],
  "feature_backlog": [88, 89, 90]
}

Do NOT modify any source files. Read only.
Output ONLY to tasks/issues-engine.json.
```

**Step 2: Launch Triage-Hub-Core agent (parallel with Step 1)**

Launch an `Explore` agent with this prompt:

```
You are the ARIA triage agent for the HUB-CORE domain.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
When classifying borderline issues, ask: does this failure degrade ARIA's ability
to serve data to the dashboard, propagate config changes, or maintain WebSocket
connections that drive real-time monitoring?

Working directory: /home/justin/Documents/projects/ha-aria

YOUR SCOPE — only classify issues that touch these paths:
  aria/hub/api.py
  aria/hub/core.py
  aria/hub/cache.py
  aria/hub/routes_*.py      (all routes_* files)
  aria/hub/websocket.py     (if exists)

STEP 1: Get all open issues
  Run: gh issue list --limit 200 --json number,title,body,labels,state

STEP 2: Filter to your domain
  Include issues whose title or body references: hub/api, hub/core, hub/cache,
  websocket send, cache key, CORS, auth, API key, config PUT, event bus publish,
  discovery endpoint, /api/ endpoints (backend issues), phantom cache, direct
  cache.get() access, IntelligencePayload TypedDict

STEP 3: For each issue, classify it (same protocol as engine agent above)
  Check file existence, grep for bug pattern, check git log for prior fixes.
  Status: confirmed | pre-existing-fix | phantom | feature-backlog
  Priority: critical | high | medium | low (anchored to ARIA intelligence pipeline impact)

STEP 4: Write output to tasks/issues-hub-core.json using the same schema.
  Include "aria_impact" field explaining how this bug degrades ARIA's core purpose.

Do NOT modify any source files. Read only.
```

**Step 3: Launch Triage-Hub-Modules agent (parallel)**

Launch an `Explore` agent with this prompt:

```
You are the ARIA triage agent for the HUB-MODULES domain.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
Modules are the real-time heart of ARIA — presence.py detects who is home,
shadow_engine.py learns patterns, ml_engine.py serves predictions. Failures here
directly degrade ARIA's core intelligence.

Working directory: /home/justin/Documents/projects/ha-aria

YOUR SCOPE — only classify issues that touch these paths:
  aria/modules/            (all modules: presence, shadow_engine, ml_engine, patterns,
                            activity_monitor, orchestrator, discovery, intelligence,
                            unifi, etc.)
  aria/shared/             (event_store, ha_automation_sync, day_classifier, constants)
  aria/watchdog.py

STEP 1: Get all open issues
  Run: gh issue list --limit 200 --json number,title,body,labels,state

STEP 2: Filter to your domain
  Include issues mentioning: module, presence.py, shadow_engine, ml_engine (module side),
  patterns.py, activity_monitor, orchestrator, discovery.py, unifi, shared/,
  event_store, ha_automation_sync, day_classifier, constants, watchdog, WatchdogResult,
  asyncio.get_event_loop, subscriber lifecycle, domain filter

STEP 3: Classify each issue (confirmed | pre-existing-fix | phantom | feature-backlog)
  Priority anchored to ARIA impact on real-time monitoring and ML pipeline.
  critical = presence detection stops working, ML model stops updating
  high = silent data corruption in modules that feed the intelligence pipeline

STEP 4: Write output to tasks/issues-hub-modules.json using same schema.

Do NOT modify any source files. Read only.
```

**Step 4: Launch Triage-Frontend agent (parallel)**

Launch an `Explore` agent with this prompt:

```
You are the ARIA triage agent for the FRONTEND/CI domain.

NORTH STAR: ARIA is an intelligent home automation platform. Its purpose is
ML-driven predictions, real-time presence detection, and automation generation.
The frontend makes ARIA's intelligence visible and actionable — broken dashboard
pages hide ML insights from the user. CI failures allow regressions to reach
production silently.

Working directory: /home/justin/Documents/projects/ha-aria

YOUR SCOPE — only classify issues that touch these paths:
  aria/dashboard/spa/      (all JSX, JS, CSS)
  .github/workflows/       (CI YAML)
  bin/*.sh                 (shell scripts)

STEP 1: Get all open issues
  Run: gh issue list --limit 200 --json number,title,body,labels,state

STEP 2: Filter to your domain
  Include issues mentioning: dashboard, spa, jsx, .js file, pipelineGraph,
  TimeChart, Sankey, Presence.jsx, PresenceTimeline, Correlations, Settings,
  DataCuration, Predictions, Timeline, Anomalies, ActivityFeed, CapabilityCard,
  MetricCard, EntityGraph, PipelineStatusBar, npm, esbuild, ShellCheck, CI,
  lint-fix.yml, GitHub Actions, actions version, shell script, stabilization-smoke,
  safeFetch, store.js, useCache, useSearch, api.js, format.js

STEP 3: Classify each issue.
  Note: Many older issues (#80–#120) may be Phase 3/4 feature tasks — classify
  these as feature-backlog. Bugs and regressions are confirmed.
  Priority: critical/high = dashboard completely broken for key ARIA features
  (Predictions page empty, Presence not rendering), medium/low = cosmetic/minor

STEP 4: Write output to tasks/issues-frontend.json using same schema.

Do NOT modify any source files. Read only.
```

**Step 5: Wait for all four triage agents to complete**

All four agents write to `tasks/issues-*.json`. Verify all four files have content:

```bash
for f in tasks/issues-engine.json tasks/issues-hub-core.json tasks/issues-hub-modules.json tasks/issues-frontend.json; do
  count=$(python3 -c "import json; d=json.load(open('$f')); print(len(d['issues']))")
  echo "$f: $count issues classified"
done
```

Expected: Each file has > 0 issues with `status` and `priority` fields populated.

**Step 6: Review triage results and spot-check**

```bash
python3 << 'EOF'
import json, glob

total_confirmed = 0
total_phantom = 0
total_preexisting = 0
total_feature = 0

for f in sorted(glob.glob("tasks/issues-*.json")):
    if "feature-backlog" in f:
        continue
    d = json.load(open(f))
    confirmed = [i for i in d["issues"] if i["status"] == "confirmed"]
    phantom = [i for i in d["issues"] if i["status"] == "phantom"]
    preexisting = [i for i in d["issues"] if i["status"] == "pre-existing-fix"]
    feature = [i for i in d["issues"] if i["status"] == "feature-backlog"]
    print(f"\n{d['domain']}:")
    print(f"  confirmed: {len(confirmed)} | phantom: {len(phantom)} | pre-existing: {len(preexisting)} | feature: {len(feature)}")
    critical = [i for i in confirmed if i["priority"] == "critical"]
    if critical:
        print(f"  CRITICAL ({len(critical)}): {[i['number'] for i in critical]}")
    total_confirmed += len(confirmed)
    total_phantom += len(phantom)
    total_preexisting += len(preexisting)
    total_feature += len(feature)

print(f"\nTOTAL:")
print(f"  confirmed to fix: {total_confirmed}")
print(f"  phantom (skip): {total_phantom}")
print(f"  pre-existing (skip): {total_preexisting}")
print(f"  feature-backlog (park): {total_feature}")
EOF
```

**Step 7: Commit triage results**

```bash
git add tasks/issues-*.json tasks/feature-backlog.json
git commit -m "chore: add triage results — confirmed issues per domain"
echo "Stage 0 triage complete: $(date -Iseconds)" >> progress.txt
```

---

## Task 3: Stage 1 — Update Conventions (general-purpose agent)

**Agent type:** `general-purpose`

**Step 1: Launch conventions update agent**

Launch a `general-purpose` agent with this prompt:

```
You are the ARIA conventions updater.

Working directory: /home/justin/Documents/projects/ha-aria

Read these files first:
  1. docs/conventions-fix-all-issues.md      — existing 6 patterns (A–F)
  2. tasks/issues-engine.json
  3. tasks/issues-hub-core.json
  4. tasks/issues-hub-modules.json
  5. tasks/issues-frontend.json

Your job: add conventions G through K to docs/conventions-fix-all-issues.md.
These cover new issue patterns not present in the Feb 25 conventions doc.

CONVENTION G — asyncio.get_event_loop() replacement
Pattern name: event-loop-api
Applies to: any asyncio.get_event_loop() call in async context or code called from async

Wrong:
  loop = asyncio.get_event_loop()
  loop.run_until_complete(coro())

Correct (inside a coroutine):
  loop = asyncio.get_running_loop()

Correct (entry point / not in coroutine):
  asyncio.run(coro())

Rules:
- NEVER use asyncio.get_event_loop() — deprecated and raises RuntimeError in Python 3.12+
- In async def: use asyncio.get_running_loop()
- In sync context launching async: use asyncio.run()
- In tests: use pytest-asyncio or new_event_loop() + run_until_complete() pattern

CONVENTION H — Typed string enums (Literal)
Pattern name: typed-literal
Applies to: any str annotation where only a fixed set of values is valid

Wrong:
  level: str  # only "warning", "critical", "info" are valid

Correct:
  from typing import Literal
  level: Literal["warning", "critical", "info"]

Rules:
- Use Literal for ALL string fields with a fixed valid set
- If the set has >5 values, use an Enum instead
- Always include the Literal type in the class __init__ or TypedDict definition

CONVENTION I — asyncio.to_thread() for blocking I/O in async context
Pattern name: async-blocking-io
Applies to: open(), json.load(), pickle.load(), sqlite3 calls inside async def methods

Wrong:
  async def _load_models(self):
      with open(path) as f:
          data = json.load(f)

Correct:
  async def _load_models(self):
      def _read():
          with open(path) as f:
              return json.load(f)
      data = await asyncio.to_thread(_read)

Rules:
- ANY blocking I/O inside async def must use asyncio.to_thread()
- This includes: file reads >1KB, sqlite3 queries, pickle.load, numpy file ops
- Small config reads (<1KB, called infrequently) may be exempt with a comment explaining why
- Always log a WARNING if asyncio.to_thread raises — do not swallow

CONVENTION J — AbortController for Preact fetch cleanup
Pattern name: fetch-abort
Applies to: any fetch() call inside useEffect that may outlive component mount

Wrong:
  useEffect(() => {
    fetch(url).then(r => r.json()).then(setData)
  }, [])

Correct:
  useEffect(() => {
    const controller = new AbortController()
    fetch(url, { signal: controller.signal })
      .then(r => r.json())
      .then(setData)
      .catch(e => { if (e.name !== 'AbortError') console.error(e) })
    return () => controller.abort()
  }, [])

Rules:
- Every fetch in useEffect must have a cleanup function that calls controller.abort()
- Catch AbortError separately — it is not a real error, do not surface to user
- For polling intervals: clear the interval AND abort any in-flight fetch in cleanup

CONVENTION K — Issue auto-close in commits
Pattern name: issue-close
Applies to: ALL fix commits

Rule: Every commit that fixes a confirmed GitHub issue MUST include "closes #NNN" in the
commit message body. Format:

  fix(domain): brief description of fix

  closes #NNN

Or for multiple issues in one commit:

  fix(domain): brief description

  closes #NNN, closes #MMM

This causes GitHub to auto-close the issue when the PR merges. No exceptions.
If a commit partially addresses an issue but doesn't fully fix it, use "refs #NNN" instead.

---

After adding all 5 new conventions, commit:

  git add docs/conventions-fix-all-issues.md
  git commit -m "docs: add conventions G-K for new issue patterns (asyncio, Literal, to_thread, AbortController, issue-close)"

Then append to progress.txt:
  echo "Stage 1 conventions updated: $(date -Iseconds)" >> progress.txt
```

**Step 2: Verify conventions doc was updated**

```bash
grep -c "Convention [A-K]" docs/conventions-fix-all-issues.md
# Expected: 11 (A through K)
```

---

## Task 4: Stage 2 — Engine Fix Team (python-expert)

**Agent type:** `python-expert`
**Worktree:** `../ha-aria-fix-engine`

**Step 1: Launch Engine fix agent**

Launch a `python-expert` agent with this prompt:

```
You are the ARIA Engine fix agent.

NORTH STAR: ARIA is an intelligent home automation platform. Its core purpose is
ML-driven predictions, real-time activity monitoring, presence detection, and
automation generation. Every fix must preserve or improve ARIA's ability to learn
from home data and generate accurate predictions. When choosing between approaches,
pick the one that better serves the intelligence pipeline.

Working directory: /home/justin/Documents/projects/ha-aria-fix-engine

Before touching any code, read these files:
  1. docs/conventions-fix-all-issues.md    — follow the convention letter for each fix
  2. tasks/issues-engine.json              — your confirmed issue list
  3. docs/system-routing-map.md            — understand full data flow
  4. tasks/interface-changes.md           — cross-domain contracts already changed

Python environment: .venv/bin/python (NOT system python3)
Test runner: .venv/bin/python -m pytest tests/ --timeout=120 -n 2

INTERNAL BATCH LOOP — repeat until all confirmed issues are addressed:

  1. Read tasks/issues-engine.json, filter status=confirmed, sort by priority
  2. Take next ≤10 issues
  3. For each issue:
     a. NORTH STAR CHECK: state how this bug affects ARIA's intelligence pipeline
     b. ECHO-BACK GATE: state exact diagnosis before touching any file
        "The bug is at [file:line]. It causes [specific failure]. Fix: [specific change]. Convention: [letter]."
     c. Write failing test first — name it test_{description}_closes_{number}
        Place in the appropriate tests/ subdirectory
     d. Run test to verify it fails:
        .venv/bin/python -m pytest tests/[path]::[test_name] -v
     e. Apply minimal fix per convention
     f. Run test to verify it passes
     g. Commit with: "fix(engine): [description]" and body "closes #NNN"
     h. If phantom/pre-existing: log in progress file, skip — do NOT commit a false fix

  4. After each batch of ≤10:
     Run known-answer suite:
       .venv/bin/python -m pytest tests/integration/known_answer/ -v --timeout=120
     If ANY fail: stop and fix before next batch
     If all pass: append batch summary to tasks/progress-engine.txt

  5. Check tasks/interface-changes.md after each batch
     If Engine changed snapshot JSON schema or training output format:
       Append: "[engine] #NNN aria/engine/[file]: added/changed [field] → Hub-Core must [action]"

PRIORITY ISSUES (fix these first — critical to ARIA intelligence pipeline):
  - Any issue tagged "critical" in tasks/issues-engine.json
  - #303 — ml_engine reads 60+ JSON synchronously (blocks event loop, degrades real-time monitoring)
  - #321 — LLM JSON parse failures return [] silently (breaks automation suggestions)
  - #313 — non-atomic snapshot log prune (data loss risk for ML training data)
  - #318 — ha-log-sync exits 0 on HA failure (monitoring blind to HA downtime)

MEMORY-SAFE TESTING: always use -n 2, never -n 6

When all confirmed issues are addressed:
  echo "DONE: $(date -Iseconds)" >> tasks/progress-engine.txt
  Run final domain suite: .venv/bin/python -m pytest tests/engine/ tests/integration/ -n 2 -q --timeout=120
```

**Step 2: Verify Engine team completion**

```bash
tail -5 ../ha-aria-fix-engine/tasks/progress-engine.txt
# Expected: last line contains "DONE:"
```

---

## Task 5: Stage 2 — Hub-Core Fix Team (python-expert)

**Agent type:** `python-expert`
**Worktree:** `../ha-aria-fix-hub-core`

**Step 1: Launch Hub-Core fix agent (parallel with Task 4)**

Launch a `python-expert` agent with this prompt:

```
You are the ARIA Hub-Core fix agent.

NORTH STAR: ARIA is an intelligent home automation platform. The hub is the
central nervous system — it serves predictions to the dashboard, broadcasts
cache updates via WebSocket, and exposes the API that makes ARIA's intelligence
visible. Failures here make ARIA's ML insights invisible to the user.

Working directory: /home/justin/Documents/projects/ha-aria-fix-hub-core

Before touching any code, read:
  1. docs/conventions-fix-all-issues.md
  2. tasks/issues-hub-core.json
  3. docs/system-routing-map.md
  4. tasks/interface-changes.md

Python environment: .venv/bin/python

YOUR FILES: aria/hub/api.py, aria/hub/core.py, aria/hub/cache.py, aria/hub/routes_*.py
Do NOT touch aria/modules/ or aria/shared/ — those belong to Hub-Modules team.

INTERNAL BATCH LOOP (same protocol as Engine team):
  ≤10 issues per batch → NORTH STAR CHECK → ECHO-BACK GATE → failing test → fix → commit
  Known-answer suite after each batch
  Append cross-domain contract changes to tasks/interface-changes.md

PRIORITY ISSUES:
  - Any tagged "critical" in tasks/issues-hub-core.json
  - #325 — unguarded WebSocket send_json after accept (service crash on dirty connect)
  - #307 — 4 discovery API endpoints permanently 503 (module ID mismatch)
  - #317 — config PUT missing event bus publish (settings never propagate)
  - #316 — phantom cache keys in /api/ml/* (anomaly_alerts always empty)
  - #315 — 15x hub.cache.get() direct access (bypasses metadata + WS notifications)
  - #314 — unauthenticated write endpoints when ARIA_API_KEY unset

INTERFACE CHANGES: If you change any API response shape, cache key name, or
WebSocket event type, append to tasks/interface-changes.md:
  "[hub-core] #NNN aria/hub/[file]: [what changed] → Frontend must update [component]"

MEMORY-SAFE TESTING: -n 2 always

When done: echo "DONE: $(date -Iseconds)" >> tasks/progress-hub-core.txt
```

**Step 2: Verify Hub-Core team completion**

```bash
tail -5 ../ha-aria-fix-hub-core/tasks/progress-hub-core.txt
# Expected: last line contains "DONE:"
```

---

## Task 6: Stage 2 — Hub-Modules Fix Team (python-expert)

**Agent type:** `python-expert`
**Worktree:** `../ha-aria-fix-hub-modules`

**Step 1: Launch Hub-Modules fix agent (parallel with Tasks 4 and 5)**

Launch a `python-expert` agent with this prompt:

```
You are the ARIA Hub-Modules fix agent.

NORTH STAR: ARIA is an intelligent home automation platform. The modules are
ARIA's real-time sensors — presence.py detects who is home, shadow_engine.py
learns behavioral patterns, ml_engine.py (module side) serves predictions to
the dashboard. Failures in modules degrade ARIA's core intelligence in real time.

Working directory: /home/justin/Documents/projects/ha-aria-fix-hub-modules

Before touching any code, read:
  1. docs/conventions-fix-all-issues.md
  2. tasks/issues-hub-modules.json
  3. docs/system-routing-map.md
  4. tasks/interface-changes.md — especially Engine changes to snapshot format

Python environment: .venv/bin/python

YOUR FILES: aria/modules/*, aria/shared/*, aria/watchdog.py
Do NOT touch aria/hub/api.py or aria/hub/core.py — those belong to Hub-Core team.

INTERNAL BATCH LOOP (same protocol as other teams):
  ≤10 issues per batch → NORTH STAR CHECK → ECHO-BACK GATE → failing test → fix → commit
  Known-answer suite after each batch
  pytest tests/hub/ -n 2 -q after each batch for domain regression check

PRIORITY ISSUES:
  - Any tagged "critical" in tasks/issues-hub-modules.json
  - #305 — presence tracking silently stops after first Frigate event (critical: aware/naive datetime)
  - #329 — shadow_engine queries DB before domain filter (perf: unnecessary I/O on every HA event)
  - #320 — watchdog returncode unchecked (health check silently passes on failed systemctl)
  - #326 — WatchdogResult.level untyped (Convention H: use Literal)
  - #319 — dead config knobs (dead knobs lie to operators about ARIA behavior)

SUBSCRIBER LIFECYCLE REMINDER (Lesson #37 — most common hub-modules anti-pattern):
  - Store callback refs on self in initialize(), not __init__()
  - Always call hub.unsubscribe() in shutdown()
  - shutdown() must cancel self._task and close self._session

INTERFACE CHANGES: If you change event bus payload shapes, MQTT message formats,
or shared constants, append to tasks/interface-changes.md.

MEMORY-SAFE TESTING: -n 2 always

When done: echo "DONE: $(date -Iseconds)" >> tasks/progress-hub-modules.txt
```

**Step 2: Verify Hub-Modules team completion**

```bash
tail -5 ../ha-aria-fix-hub-modules/tasks/progress-hub-modules.txt
# Expected: last line contains "DONE:"
```

---

## Task 7: Stage 2 — Frontend Fix Team (general-purpose + bash-expert sub-agent)

**Agent type:** `general-purpose`
**Worktree:** `../ha-aria-fix-frontend`

**Step 1: Launch Frontend fix agent (parallel with Tasks 4–6)**

Launch a `general-purpose` agent with this prompt:

```
You are the ARIA Frontend fix agent.

NORTH STAR: ARIA is an intelligent home automation platform. The dashboard makes
ARIA's intelligence visible and actionable — broken Predictions pages hide ML
insights, broken Presence pages hide who is home, broken Correlations pages hide
behavioral patterns. Fix UI bugs in priority order based on which dashboard pages
carry the most intelligence value to the user.

Working directory: /home/justin/Documents/projects/ha-aria-fix-frontend

Before touching any code, read:
  1. docs/conventions-fix-all-issues.md      — especially Conventions C, E, J, K
  2. tasks/issues-frontend.json
  3. tasks/interface-changes.md              — API shape changes from Python teams
  4. docs/dashboard-build.md                 — build instructions

YOUR FILES: aria/dashboard/spa/*, .github/workflows/*, bin/*.sh
Do NOT touch any Python files.

BUILD GATE: After every fix, run:
  cd aria/dashboard/spa && npm run build 2>&1 | tail -5
  Must exit 0. Fix build errors before next issue.

NODE ENVIRONMENT: cd aria/dashboard/spa first for all npm commands.

INTERNAL BATCH LOOP:
  ≤10 issues per batch → NORTH STAR CHECK → ECHO-BACK GATE → fix → build gate → commit
  Commit message: "fix(frontend): [description]\n\ncloses #NNN"

CI/SHELL ISSUES (#310, #330, #332, #327, and any .sh or .yml issues):
  For these, spawn a bash-expert sub-agent with this prompt template:
  "Fix GitHub issue #NNN: [title]. File: [path].
   Requirements: no eval, no hardcoded absolute paths, must pass shellcheck [file].
   ShellCheck binary: shellcheck (install if missing: sudo apt-get install shellcheck).
   For GitHub Actions YAML: use actions/checkout@v4, actions/setup-python@v5 consistently.
   Commit with: git commit -m 'fix(ci): [description] closes #NNN'"

PRIORITY ISSUES (in order of ARIA intelligence value):
  - #322 — pipelineGraph.js Sankey node has no input links (misleading ARIA pipeline view)
  - #310 — stabilization-smoke.sh eval injection (security: arbitrary code execution)
  - #332 — lint-fix.yml || true masks ruff crash (CI blind to Python failures)
  - #330 — no ShellCheck in CI
  - #327 — GitHub Actions version inconsistency

Check tasks/interface-changes.md: if Python teams changed API response shapes or
cache key names, update the corresponding fetch calls in api.js/store.js to match.

When done:
  cd aria/dashboard/spa && npm run build 2>&1 | tail -3
  echo "DONE: $(date -Iseconds)" >> tasks/progress-frontend.txt
```

**Step 2: Verify Frontend team completion**

```bash
tail -5 ../ha-aria-fix-frontend/tasks/progress-frontend.txt
# Expected: last line contains "DONE:"
```

---

## Task 8: Stage 3 — Judge Round (coordinator + 3 parallel reviewers)

**Agent type:** `general-purpose` (coordinator), specialized sub-agents

**Step 1: Verify all four teams are done**

```bash
for domain in engine hub-core hub-modules frontend; do
  worktree="../ha-aria-fix-${domain}"
  progress="${worktree}/tasks/progress-${domain}.txt"
  if grep -q "DONE:" "$progress" 2>/dev/null; then
    echo "✓ ${domain}: DONE"
  else
    echo "✗ ${domain}: NOT DONE"
  fi
done
# All four must show ✓ before proceeding
```

**Step 2: Review interface-changes.md for cross-team conflicts**

```bash
cat tasks/interface-changes.md
# Manually review: are there any conflicts between what Engine changed
# and what Hub-Core or Frontend expects?
```

**Step 3: Launch Judge Silent-Failure Hunter (pr-review-toolkit:silent-failure-hunter)**

Launch a `pr-review-toolkit:silent-failure-hunter` agent with this prompt:

```
You are the ARIA judge — silent failure hunter.

Working directory: /home/justin/Documents/projects/ha-aria

Scan all Python files changed across the four fix branches:
  git diff main fix/engine fix/hub-core fix/hub-modules --name-only | grep '\.py$' | sort -u

For each changed Python file, check:
  1. Any except block that logs nothing before returning a fallback
  2. Any return None or return [] without a preceding logger.warning()
  3. Any bare json.load() without try/except json.JSONDecodeError
  4. Any asyncio.create_task() without a done-callback for error visibility (Lesson #43)
  5. Any datetime object passed to json.dumps() or hub.publish() without .isoformat()
  6. Any asyncio.get_event_loop() (should be get_running_loop() or asyncio.run())

NORTH STAR FILTER: Prioritize HIGH findings that affect data integrity
(training snapshots, cache writes, presence signals, event bus publishes)
over cosmetic silent returns in low-traffic paths.

Report format: file:line | issue type | severity (HIGH/MEDIUM) | ARIA pipeline impact
Output to: tasks/judge-silent-failures.md

Do NOT fix anything. Report only.
```

**Step 4: Launch Judge Test Analyzer (pr-review-toolkit:pr-test-analyzer) — parallel with Step 3**

Launch a `pr-review-toolkit:pr-test-analyzer` agent with this prompt:

```
You are the ARIA judge — test coverage analyzer.

Working directory: /home/justin/Documents/projects/ha-aria

Review the test coverage for all fixes made across the four fix branches.

Get the list of new test files added:
  git diff main fix/engine fix/hub-core fix/hub-modules fix/frontend --name-only | grep 'tests/'

For each confirmed issue in tasks/issues-*.json with status=confirmed:
  1. Does a corresponding test exist? (search by issue number in test file names and docstrings)
  2. Does the test name follow convention: test_{description}_closes_{number}?
  3. Would the test have caught the bug on pre-fix code?
     (A test that passes on both pre-fix and post-fix code is not a fix test)

Flag any confirmed issue that has NO corresponding test — these are coverage gaps.
Flag any test that tests a different thing than the issue describes.

Output to: tasks/judge-test-coverage.md

Format:
  COVERED: #NNN — test_name (file:line)
  MISSING: #NNN — no test found for this fix
  WEAK: #NNN — test exists but doesn't test the specific failure path
```

**Step 5: Launch Judge Completeness Reviewer (superpowers:code-reviewer) — parallel with Steps 3–4**

Launch a `superpowers:code-reviewer` agent with this prompt:

```
You are the ARIA judge — completeness and convention reviewer.

Working directory: /home/justin/Documents/projects/ha-aria

Review against the original plan and coding standards:

1. COMPLETENESS — For each confirmed issue in tasks/issues-*.json (status=confirmed):
   - Was the issue addressed? Check git log across all 4 fix branches:
     git log fix/engine fix/hub-core fix/hub-modules fix/frontend --oneline | grep "#NNN"
   - List any confirmed issue with no corresponding commit

2. CONVENTION COMPLIANCE — For each changed file:
   - Does each fix follow the convention letter stated in tasks/issues-*.json?
   - Check docs/conventions-fix-all-issues.md for the exact pattern
   - Flag deviations with: "Issue #NNN: expected Convention X pattern, found [what was done instead]"

3. INTERFACE CONTRACT — Read tasks/interface-changes.md:
   - Are cross-domain changes consistent? (e.g., if Engine added "time_features" to snapshot,
     does Hub-Core's fix account for this key?)
   - Flag any mismatch that would cause a runtime error at the integration seam

4. COMMIT MESSAGES — Verify every fix commit contains "closes #NNN" (Convention K)
   git log fix/engine fix/hub-core fix/hub-modules fix/frontend --oneline | grep -v "closes #"
   Flag any fix commit missing the close reference

Output to: tasks/judge-completeness.md
Verdict per category: PASS or FAIL with specific items
```

**Step 6: Wait for all three judge agents, then run tests**

After all three judge agents write their output files:

```bash
# Verify judge reports exist
ls -la tasks/judge-silent-failures.md tasks/judge-test-coverage.md tasks/judge-completeness.md
```

**Step 7: Run full test suite (memory-safe)**

This runs in the main worktree after temporarily merging all branches into a test branch:

```bash
# Create temporary merge branch for testing
git checkout -b fix/judge-test-merge
git merge fix/engine --no-ff --no-edit
git merge fix/hub-core --no-ff --no-edit
git merge fix/hub-modules --no-ff --no-edit
git merge fix/frontend --no-ff --no-edit

# Run full suite — memory-safe n=2
.venv/bin/python -m pytest tests/ --timeout=120 -q -n 2 2>&1 | tee tasks/judge-test-results.txt | tail -10
# Expected: ≥2384 passed, 0 errors (pre-existing synthetic errors acceptable)

# Frontend build
cd aria/dashboard/spa && npm run build 2>&1 | tee ../../tasks/judge-build-results.txt | tail -5
cd ../../../
# Expected: exit 0
```

**Step 8: Produce judge verdict**

Create `tasks/judge-round-1.md` summarizing all findings:

```bash
# Check for HIGH silent failures
high_count=$(grep -c "HIGH" tasks/judge-silent-failures.md 2>/dev/null || echo 0)
# Check test results
test_result=$(tail -1 tasks/judge-test-results.txt)
# Check build result
build_ok=$(grep -c "error" tasks/judge-build-results.txt 2>/dev/null && echo FAIL || echo PASS)

cat > tasks/judge-round-1.md << EOF
# Judge Round 1 Verdict: $(date -Iseconds)

## Summary
- Silent failures HIGH: ${high_count} (must be 0 to PASS)
- Test suite: ${test_result}
- npm build: ${build_ok}
- See: tasks/judge-silent-failures.md, tasks/judge-test-coverage.md, tasks/judge-completeness.md

## Verdict: [PASS or FAIL — fill in manually after reviewing all three reports]

## Rework Required
[List specific teams + issue numbers if FAIL]
EOF
```

**Step 9: If PASS — proceed to Task 10. If FAIL — proceed to Task 9.**

---

## Task 9: Stage 4 — Rework (Conditional, only if Judge FAIL)

**Step 1: Read the rework list from tasks/judge-round-1.md**

Identify which teams have rework items.

**Step 2: For each team with rework items, relaunch the appropriate fix agent**

Use the same agent prompt as Tasks 4–7, but add:

```
REWORK MODE: You are fixing specific issues flagged by the judge.

Rework items from tasks/judge-round-1.md:
  [paste specific items for this team]

For each rework item:
  1. Read the judge's specific finding
  2. Apply the correction (add missing test, fix convention violation, add close reference)
  3. Commit with: "fix(domain): rework per judge [specific item]"
  4. Re-run known-answer suite

When all rework items are addressed:
  echo "REWORK DONE: $(date -Iseconds)" >> tasks/progress-{domain}.txt
```

**Step 3: Re-run only the affected judge sub-agents**

If rework was for silent failures: re-run `pr-review-toolkit:silent-failure-hunter`
If rework was for test coverage: re-run `pr-review-toolkit:pr-test-analyzer`
If rework was for completeness: re-run `superpowers:code-reviewer`

**Step 4: Re-run test suite**

Same as Task 8 Step 7. Maximum 2 rework rounds before escalating to human review.

**Step 5: Update judge-round-1.md with new verdict**

---

## Task 10: Stage 5 — Merge Sequence

**Step 1: Clean up test merge branch**

```bash
git checkout main
git branch -D fix/judge-test-merge 2>/dev/null || true
```

**Step 2: Create the integration branch**

```bash
git checkout -b fix/all-issues-r1
echo "Integration branch created: $(date -Iseconds)" >> progress.txt
```

**Step 3: Merge Engine (defines interfaces Hub reads)**

```bash
git merge fix/engine --no-ff -m "merge: engine team fixes — closes engine issues"
# Check for conflicts
git status
# If conflicts: resolve in favor of the fix that better serves ARIA's intelligence pipeline
```

**Step 4: Check Engine→Hub-Core interface changes**

```bash
grep "\[engine\]" tasks/interface-changes.md
# Review: does Hub-Core need to handle any new snapshot fields or output formats?
```

**Step 5: Merge Hub-Core**

```bash
git merge fix/hub-core --no-ff -m "merge: hub-core team fixes — closes hub-core issues"
git status
# Resolve any conflicts; prefer fix that maintains cache/API contract integrity
```

**Step 6: Check Hub-Core→Hub-Modules interface changes**

```bash
grep "\[hub-core\]" tasks/interface-changes.md
# Review: any cache key or event bus changes Hub-Modules depends on?
```

**Step 7: Merge Hub-Modules**

```bash
git merge fix/hub-modules --no-ff -m "merge: hub-modules team fixes — closes hub-modules issues"
git status
```

**Step 8: Check Python→Frontend interface changes**

```bash
grep -E "\[engine\]|\[hub-core\]|\[hub-modules\]" tasks/interface-changes.md | grep -i "frontend\|api\|cache key\|response"
# Review: any API shape or cache key changes Frontend fetch calls depend on?
```

**Step 9: Merge Frontend**

```bash
git merge fix/frontend --no-ff -m "merge: frontend team fixes — closes frontend issues"
git status
```

**Step 10: Run full suite on merged branch**

```bash
.venv/bin/python -m pytest tests/ --timeout=120 -q -n 2 2>&1 | tail -10
cd aria/dashboard/spa && npm run build 2>&1 | tail -5
```

Expected: ≥2,384 passed, npm build exits 0.

**Step 11: Append to progress.txt**

```bash
echo "Stage 5 merge complete: $(date -Iseconds)" >> progress.txt
passed=$(python3 -m pytest tests/ --timeout=120 -q -n 2 2>&1 | grep "passed" | tail -1)
echo "Test count: ${passed}" >> progress.txt
```

---

## Task 11: Stage 6 — Final Audit (integration-tester)

**Agent type:** `integration-tester`

**Step 1: Start the hub in the background**

```bash
cd /home/justin/Documents/projects/ha-aria-fix-all-issues-r1 2>/dev/null || \
cd /home/justin/Documents/projects/ha-aria
aria serve &
HUB_PID=$!
sleep 4
curl -s http://127.0.0.1:8001/api/health | python3 -m json.tool
```

**Step 2: Launch integration-tester agent**

Launch an `integration-tester` agent with this prompt:

```
You are the ARIA final audit agent.

Working directory: /home/justin/Documents/projects/ha-aria
Hub is running at http://127.0.0.1:8001

NORTH STAR: ARIA is an intelligent home automation platform. Verify that the
intelligence pipeline works end-to-end — data flows from HA through the engine,
through the hub cache, out the API, over WebSocket, and renders correctly in
the dashboard.

HORIZONTAL SWEEP — hit every route group from docs/system-routing-map.md:
  Use docs/api-reference.md for exact curl commands.
  For each route: record status code, verify response shape, note any errors.
  Key routes to verify with specific fixes:
    GET  /api/health                  → should return {auth_enabled: bool, ...}
    GET  /api/cache                   → verify no phantom keys in /api/ml/*
    GET  /api/cache/activity_summary  → verify non-empty after activity data
    GET  /api/events?limit=50         → verify pagination works (#295)
    POST /api/models/retrain          → verify 429 on second call within 60s (#298)
    GET  /api/settings/discovery      → verify 500 (not 200 with error body) on failure (#293)
    All  /api/discover/*              → verify 4 previously-503 endpoints now return 200 (#307)
  Output: tasks/audit-horizontal.md with PASS/FAIL per route group

VERTICAL TRACE — one complete data path:
  1. aria snapshot-intraday
     → verify JSON file written to ~/ha-logs/intelligence/
     → verify snapshot contains "presence" key
     → verify snapshot contains "time_features" key
  2. GET /api/cache/intelligence
     → verify response reflects new snapshot
  3. WebSocket: connect to ws://127.0.0.1:8001/ws
     → wait 10s for cache_updated event
  4. Dashboard render check:
     → curl -s http://127.0.0.1:8001/ui/ | grep -c "aria"  (should be > 0)
     → curl -s http://127.0.0.1:8001/ui/ | grep -i "error"  (should be 0)
  Output: tasks/audit-vertical.md with per-step PASS/FAIL

Final verdict: tasks/audit-final.md
If all PASS: "FINAL AUDIT: PASS — ARIA intelligence pipeline verified end-to-end"
If any FAIL: list specific failures with root cause
```

**Step 3: Stop hub after audit**

```bash
kill $HUB_PID 2>/dev/null || pkill -f "aria serve" || true
```

**Step 4: Review audit results**

```bash
cat tasks/audit-final.md
grep "PASS\|FAIL" tasks/audit-horizontal.md | head -20
grep "PASS\|FAIL" tasks/audit-vertical.md
```

---

## Task 12: Close Issues and Create PR

**Step 1: Count auto-closed issues via commit messages**

```bash
git log fix/all-issues-r1 --oneline | grep "closes #" | wc -l
```

**Step 2: Manually close any confirmed issues not yet auto-closed**

```bash
python3 << 'EOF'
import json, glob, subprocess

# Get all confirmed issues from triage files
confirmed = set()
for f in glob.glob("tasks/issues-*.json"):
    if "feature-backlog" in f:
        continue
    d = json.load(open(f))
    for i in d["issues"]:
        if i["status"] == "confirmed":
            confirmed.add(i["number"])

# Get issues already referenced in commits
result = subprocess.run(
    ["git", "log", "fix/all-issues-r1", "--oneline"],
    capture_output=True, text=True
)
closed_in_commits = set()
for line in result.stdout.splitlines():
    import re
    for n in re.findall(r'closes #(\d+)', line, re.IGNORECASE):
        closed_in_commits.add(int(n))

not_closed = confirmed - closed_in_commits
print(f"Confirmed: {len(confirmed)}, in commits: {len(closed_in_commits)}, need manual close: {len(not_closed)}")
print("Manual close needed:", sorted(not_closed))
EOF
```

**Step 3: Close remaining issues manually**

```bash
# For each number in the "Manual close needed" list above:
gh issue close NNN --comment "Fixed in fix/all-issues-r1 — verified by integration-tester audit"
```

**Step 4: Create feature-backlog tracking issue**

```bash
python3 << 'EOF'
import json

d = json.load(open("tasks/feature-backlog.json"))
numbers = d.get("numbers", [])
body = "\n".join(f"- #{n}" for n in sorted(numbers))
print(f"Feature backlog count: {len(numbers)}")
print(body[:500])
EOF

# If feature backlog is non-empty:
gh issue create \
  --title "Feature backlog: phase-task issues deferred from fix-all-issues agent teams run" \
  --body "These issues were classified as feature requests (not defects) during triage and were not fixed in the agent-teams run. Address in a dedicated feature sprint.

$(python3 -c "import json; d=json.load(open('tasks/feature-backlog.json')); print(chr(10).join(f'- #' + str(n) for n in sorted(d.get('numbers',[]))))")"
```

**Step 5: Create the PR**

```bash
git push origin fix/all-issues-r1

gh pr create \
  --title "fix: agent teams — fix all applicable GitHub issues (engine + hub + frontend)" \
  --body "$(cat << 'EOF'
## Summary

Parallel agent team run fixing all applicable GitHub issues across all domains.

- **Engine team** (`python-expert`): aria/engine/, aria/automation/, aria/faces/
- **Hub-Core team** (`python-expert`): aria/hub/api.py, aria/hub/core.py, aria/hub/cache.py
- **Hub-Modules team** (`python-expert`): aria/modules/, aria/shared/, aria/watchdog.py
- **Frontend team** (`general-purpose` + `bash-expert`): aria/dashboard/spa/, CI, shell scripts

## Process

1. Parallel triage (4 × Explore agents) — classified all 182 issues
2. Conventions update — added patterns G–K to docs/conventions-fix-all-issues.md
3. Four parallel fix teams in isolated worktrees
4. Judge round: silent-failure-hunter + pr-test-analyzer + code-reviewer + full suite
5. Merge: Engine → Hub-Core → Hub-Modules → Frontend
6. Final audit: integration-tester horizontal sweep + vertical trace

## Quality Gates

- [ ] ≥2,384 tests passing
- [ ] npm build clean
- [ ] 0 HIGH silent-failure findings
- [ ] Final H+V audit: PASS
- [ ] Feature backlog tracking issue created

## Test plan

- Run `pytest tests/ -n 2 --timeout=120` — must show ≥2384 passed
- Run `cd aria/dashboard/spa && npm run build` — must exit 0
- Check `tasks/audit-final.md` — must show PASS

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 6: Final progress log entry**

```bash
echo "Stage 6 PR created: $(date -Iseconds)" >> progress.txt
echo "Final status: see tasks/audit-final.md + tasks/judge-round-1.md" >> progress.txt
git add progress.txt tasks/
git commit -m "chore: final progress log and judge artifacts"
git push origin fix/all-issues-r1
```

---

## Quick Reference: Worktree Paths

| Team | Worktree Path | Branch |
|------|--------------|--------|
| Engine | `../ha-aria-fix-engine` | `fix/engine` |
| Hub-Core | `../ha-aria-fix-hub-core` | `fix/hub-core` |
| Hub-Modules | `../ha-aria-fix-hub-modules` | `fix/hub-modules` |
| Frontend | `../ha-aria-fix-frontend` | `fix/frontend` |
| Integration | current repo | `fix/all-issues-r1` |

## Quick Reference: Agent Types

| Stage | Agent Type | Reason |
|-------|-----------|--------|
| Triage (×4) | `Explore` | Read-only, no accidental writes |
| Conventions | `general-purpose` | Needs read + write |
| Engine fixes | `python-expert` | Async discipline, HA ecosystem, lesson-scanner built-in |
| Hub-Core fixes | `python-expert` | FastAPI, aiohttp, WebSocket async patterns |
| Hub-Modules fixes | `python-expert` | Subscriber lifecycle, async modules |
| Frontend JSX | `general-purpose` | Preact + bash + YAML in one team |
| Frontend CI/shell | `bash-expert` (sub-agent) | ShellCheck, YAML, eval injection prevention |
| Judge: silent failures | `pr-review-toolkit:silent-failure-hunter` | Designed for this exact check |
| Judge: test coverage | `pr-review-toolkit:pr-test-analyzer` | Test gap detection |
| Judge: completeness | `superpowers:code-reviewer` | Reviews against plan + standards |
| Final audit | `integration-tester` | Cross-service seam verification (Cluster B bugs) |
