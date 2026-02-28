# Lesson: Blocking JSON File I/O in Async Methods Stalls the Event Loop

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** async
**Keywords:** asyncio, blocking IO, json.load, asyncio.to_thread, event loop, training, file read
**Files:** aria/modules/ml_engine.py:455, aria/modules/ml_engine.py:1520, aria/modules/ml_engine.py:1543, aria/modules/activity_monitor.py:667

---

## Observation (What Happened)

`ml_engine.py` reads up to 60+ JSON snapshot files synchronously inside `async def` methods (`_load_training_data`, `_get_previous_snapshot`, `_compute_rolling_stats`). These block the entire asyncio event loop during every training run — incoming WebSocket events, presence signals, and API requests queue up for seconds.

The same file correctly wraps all pickle I/O with `asyncio.to_thread` (lines 254, 572, 648, 1650), so JSON reads were an inconsistent omission.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `open()` + `json.load()` is synchronous blocking I/O inside `async def` methods.

**Why #2:** Reading 60 files in a loop can take hundreds of milliseconds — enough to cause measurable latency on all concurrent event handlers.

**Why #3:** Pickle I/O was correctly wrapped in `asyncio.to_thread` when the ML engine was implemented, but JSON reads were added later without applying the same pattern consistently.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap all `open()` + `json.load()` in async methods with `asyncio.to_thread` | proposed | Justin | issue #303 |
| 2 | Audit all `async def` methods for bare `open()` calls; enforce in code review | proposed | Justin | issue #303 |

## Key Takeaway

Any `open()` / `json.load()` inside `async def` is blocking I/O — wrap it in `asyncio.to_thread()` just like pickle operations, or the event loop stalls during every file read.
