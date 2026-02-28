# Lesson: WebSocket Error-Path Calls `close()` on a `None` Client — Double-Fault Crash

**Date:** 2026-02-28
**System:** community (home-assistant/supervisor)
**Tier:** lesson
**Category:** error-handling
**Keywords:** WebSocket, NoneType, close, error handling, double fault, supervisor, shutdown, cleanup
**Source:** https://github.com/home-assistant/supervisor/issues/5629

---

## Observation (What Happened)

When the Home Assistant Supervisor's WebSocket connection to HA Core was dropped (e.g., due to a blocking operation stalling the event loop), the error-handling code called `await self._client.close()` as cleanup — but `self._client` was already `None` because the connection had already been torn down. This created a `AttributeError: 'NoneType' object has no attribute 'close'` — a double fault where the primary error (connection lost) triggered a secondary error in the cleanup path itself.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `'NoneType' object has no attribute 'close'` was raised in the exception handler.
**Why #2:** The error handler assumed `self._client` was still a valid object reference at cleanup time.
**Why #3:** A concurrent code path (or the same connection drop) had already set `self._client = None` before the cleanup ran.
**Why #4:** No null-guard on `self._client` before calling `close()` in the except block.
**Why #5:** Error paths are written to the happy path — cleanup calls in except blocks are rarely guarded against the state changes caused by the error itself.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always null-guard resource references in cleanup paths: `if self._client is not None: await self._client.close()` | proposed | community | issue #5629 |
| 2 | Apply `client, self._client = self._client, None` (swap-then-close) to prevent double-close races | proposed | community | issue #5629 |
| 3 | Error-path code must be treated as if the object is in any possible intermediate state — never assume the resource is still valid | proposed | community | issue #5629 |

## Key Takeaway

Exception handlers must null-guard every resource reference they close or release — the error that triggered the handler may have already invalidated those references, causing a secondary crash that buries the original error in noise.
