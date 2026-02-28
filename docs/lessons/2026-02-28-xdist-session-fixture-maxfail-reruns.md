# Lesson: Session-Scoped Fixture Runs Multiple Times Under xdist --maxfail

**Date:** 2026-02-28
**System:** community (pytest-dev/pytest-xdist)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, xdist, session fixture, maxfail, parallel, fixture scope, over-execution
**Source:** https://github.com/pytest-dev/pytest-xdist/issues/1024

---

## Observation (What Happened)
A `session`-scoped fixture decorated with `autouse=True` was executed 3 times instead of once when running `pytest -n 2 --maxfail=1`. Without `--maxfail`, it ran twice (once per worker, as expected). With `--maxfail=1`, xdist spawned a third worker after the first worker failed and aborted.

## Analysis (Root Cause — 5 Whys)
**Why #1:** xdist workers each initialize their own copy of session-scoped fixtures because each worker is a separate process.
**Why #2:** When `--maxfail=1` triggers early shutdown, xdist dispatches remaining collected tests to a new worker before the abort signal fully propagates.
**Why #3:** The new worker spins up and initializes the session fixture again before receiving the stop signal.
**Why #4:** Session-scoped fixtures in xdist are per-worker-session, not per-suite — there is no shared fixture state across processes.
**Why #5:** No coordination mechanism prevents a new worker's session setup when early-exit is in flight.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Never assume session-scoped fixture runs exactly once under xdist — design it to be idempotent and safe to run N times where N = number of workers | proposed | community | issue #1024 |
| 2 | For resources that must be initialized exactly once (e.g., DB seed, external service setup), use a shared fixture with file-based lock or `pytest_configure` hook at the controller level | proposed | community | issue #1024 |
| 3 | Avoid side effects in session-scoped autouse fixtures that assume single execution | proposed | community | issue #1024 |

## Key Takeaway
Under xdist, `session`-scoped fixtures run once per worker process, not once per test suite — a 4-worker run will execute a session fixture 4 times, and `--maxfail` can trigger additional runs due to race conditions during abort.
