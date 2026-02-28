# Lesson: Per-Event HTTP Session Creation Exhausts File Descriptors Under Load

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** async
**Keywords:** aiohttp, ClientSession, per-event, file descriptor, fd exhaustion, reuse, Frigate, camera
**Files:** aria/modules/presence.py:644-647

---

## Observation (What Happened)

`presence.py:_process_face_async` creates a new `aiohttp.ClientSession()` for every Frigate person detection event using an inline `async with aiohttp.ClientSession() as session`. Under moderate camera load (multiple person events per second), this creates and tears down a new TCP connection per event, eventually exhausting the process's file descriptors.

`self._http_session` exists in the module specifically for session reuse and is used correctly at every other call site in the module — this is the sole exception.

## Analysis (Root Cause — 5 Whys)

**Why #1:** A new `aiohttp.ClientSession` was created inline at one call site instead of reusing the module-level session.

**Why #2:** The Frigate snapshot fetch was added at a different time than the other HTTP calls in the module, and the developer didn't notice the existing `self._http_session` pattern.

**Why #3:** Per-event session creation is a common mistake because the `async with aiohttp.ClientSession()` pattern looks clean and self-contained — the cost (FD + TCP overhead per event) is invisible until load.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace inline `aiohttp.ClientSession()` at presence.py:644 with `self._http_session` (guard with `if not self._http_session: return`) | proposed | Justin | issue #252 |
| 2 | In code review, flag any `async with aiohttp.ClientSession()` inside an event handler — sessions must be module-level | proposed | Justin | issue #252 |

## Key Takeaway

Never create a new `aiohttp.ClientSession()` inside an event handler — always reuse the module-level session; per-event session creation exhausts file descriptors under camera or sensor load.
