# Lesson: Library-Level Signal Handler Registration Silently Overwrites Application Handlers

**Date:** 2026-02-28
**System:** community (miguelgrinberg/python-socketio)
**Tier:** lesson
**Category:** async
**Keywords:** signal handler, asyncio, SIGINT, SIGTERM, loop.add_signal_handler, library side effect, graceful shutdown
**Source:** https://github.com/miguelgrinberg/python-socketio/issues/1390

---

## Observation (What Happened)

python-socketio's `connect()` call internally registered its own signal handlers via `asyncio.loop.add_signal_handler`, silently replacing any handlers the application had already registered for SIGINT/SIGTERM. Applications expecting to catch signals for graceful shutdown received no notification.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The application's signal handlers were never called on Ctrl+C.
**Why #2:** `python-socketio` called `loop.add_signal_handler(SIGINT, ...)` inside `connect()`, which replaces the previously registered handler entirely (not a chain — `add_signal_handler` overwrites).
**Why #3:** The library's connect path treated signal setup as a lifecycle concern, not an application concern.
**Why #4:** `asyncio.loop.add_signal_handler` has replace-not-append semantics — there is no built-in chaining; the last caller wins.
**Why #5:** Library authors did not audit for global side effects on shared OS-level resources during connection setup.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Libraries must never call `loop.add_signal_handler` unless the application explicitly delegates signal handling to them | proposed | community | issue link |
| 2 | Application code that uses third-party async clients: audit `connect()` / `start()` call chains for signal handler side effects; re-register after connect if needed | proposed | community | issue #1390 |
| 3 | In long-running services, register signal handlers AFTER all library initialization, or use `signal.signal()` (not asyncio's variant) as a fallback that libraries cannot overwrite | proposed | community | issue #1390 |

## Key Takeaway

Any library that calls `asyncio.loop.add_signal_handler` during its own initialization silently replaces application-registered signal handlers — always register your own handlers after all third-party library setup, or verify the library does not touch signal handling.
