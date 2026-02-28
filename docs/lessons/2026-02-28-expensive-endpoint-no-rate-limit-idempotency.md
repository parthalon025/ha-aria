# Lesson: Expensive Mutation Endpoints Without Rate Limiting or In-Progress Guard

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** rate limiting, idempotency, retrain, concurrent, 409 Conflict, mutex, in-progress guard, resource exhaustion
**Files:** aria/hub/api.py (POST /api/models/retrain)

---

## Observation (What Happened)

`POST /api/models/retrain` starts a full ML retraining job with no guard against concurrent invocations. Clicking the button rapidly, a retry loop in the frontend, or concurrent API calls can queue multiple simultaneous retraining jobs, exhausting CPU and potentially corrupting model artifacts through concurrent writes to the same `.pkl` files.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The endpoint has no state tracking — it fires a new training task every time it's called regardless of what is already running.

**Why #2:** Expensive long-running operations are commonly protected in web apps with in-progress guards, but this pattern was not applied when the endpoint was created.

**Why #3:** Without a guard, the endpoint is idempotent-by-assumption: the developer assumed callers would wait, but the UI has retry logic and callers can be impatient.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Track retrain state in a module-level flag (`_training_in_progress`); return 409 Conflict if a retrain is running | proposed | Justin | issue #298 |
| 2 | Add minimum interval between re-triggered retrains (e.g. 60s cooldown) | proposed | Justin | issue #298 |
| 3 | Apply the same pattern to any endpoint that launches a long-running exclusive operation (e.g., discovery run, bootstrap) | proposed | Justin | issue #298 |

## Key Takeaway

Any endpoint that launches an exclusive long-running operation (training, reindexing, bootstrap) must return 409 Conflict if already in progress — an in-flight state flag costs one bool and prevents resource exhaustion and artifact corruption.
