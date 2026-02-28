# Lesson: Unhandled Network Exception in Agent Tool Crashes Entire Agent Session

**Date:** 2026-02-28
**System:** community (Fosowl/agenticSeek)
**Tier:** lesson
**Category:** error-handling
**Keywords:** network timeout, ReadTimeoutError, exception propagation, agent crash, ASGI, browser agent, recovery, error boundary, exception type mismatch
**Source:** https://github.com/Fosowl/agenticSeek/issues/329

---

## Observation (What Happened)

When the agenticSeek browser agent attempted to load an unreliable website, `urllib3.exceptions.ReadTimeoutError` propagated up through `browser.go_to()` → `browser_agent.process()` → `planner_agent.process()` → `api.py`. The exception was not caught at any layer because exception handlers were written to catch `TimeoutError` (not `urllib3.exceptions.ReadTimeoutError`), and the ASGI application crashed with `ERROR: Exception in ASGI application`, terminating the entire agent session.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `browser.go_to()` re-raised exceptions it caught rather than normalizing them into a tool-layer error type.
**Why #2:** Upper layers caught `TimeoutError` but `urllib3.exceptions.ReadTimeoutError` is a different class (it inherits from `urllib3.exceptions.PoolError`, not from `TimeoutError`).
**Why #3:** The exception type hierarchy for urllib3 was not documented or checked; the assumption was all timeouts are `TimeoutError`.
**Why #4:** No error boundary existed at the agent task level — one tool failure propagated to crash the session instead of returning a failure result.
**Why #5:** The design treated tool exceptions as fatal rather than recoverable — the planner agent could have retried or skipped the failing URL.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | In `browser.go_to()`, catch both `TimeoutError` and `urllib3.exceptions.ReadTimeoutError` (and other urllib3 exceptions) and raise a normalized `BrowserToolError` | proposed | community | https://github.com/Fosowl/agenticSeek/issues/329 |
| 2 | Add an error boundary at the agent task level: catch all exceptions from tool calls, return a structured failure result, do not re-raise | proposed | community | issue |
| 3 | In `api.py`, add a top-level exception handler for `Exception` that returns a 500 with structured error JSON rather than crashing the ASGI app | proposed | community | issue |

## Key Takeaway

Third-party library exception types often do not inherit from Python stdlib base classes (e.g., urllib3's `ReadTimeoutError` is not a `TimeoutError`) — always check the actual exception class hierarchy when writing catch clauses for external libraries, and normalize external exceptions into domain types at the boundary layer.
