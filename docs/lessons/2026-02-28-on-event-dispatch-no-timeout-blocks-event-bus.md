# Lesson: Event Bus Dispatch Without Per-Module Timeout Lets One Slow Module Block All Others

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** async
**Keywords:** event bus, on_event, timeout, asyncio.wait_for, dispatch, slow module, blocking, hub, cache_updated
**Files:** aria/hub/core.py:402-416

---

## Observation (What Happened)

`hub/core.py:publish()` iterates all 9 modules and calls `await module.on_event(event_type, data)` sequentially with no timeout per module. A single module that blocks (slow HTTP call, deadlocked lock, infinite loop) delays every subsequent module and all high-frequency `state_changed` events. A 100ms warning threshold at line 410 is observability only — it does not interrupt the blocking module.

`cache_updated` events published inside `set_cache()` are directly affected — any module blocking on `on_event("cache_updated")` stalls the hub's internal poll loop.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `on_event` is awaited with no timeout, so a module that takes arbitrarily long blocks all subsequent dispatches.

**Why #2:** The observability warning (100ms threshold) was added but the corrective action (timeout + skip) was not.

**Why #3:** Sequential dispatch without isolation means modules are not truly independent — one misbehaving module degrades all others.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap each `on_event` call with `asyncio.wait_for(..., timeout=1.0)`; log ERROR on timeout but continue to next module | proposed | Justin | issue #261 |
| 2 | Consider parallel dispatch with `asyncio.gather` for non-ordering-sensitive event types | proposed | Justin | issue #261 |

## Key Takeaway

When dispatching to multiple independent modules sequentially, each dispatch must have an explicit per-module timeout — a single blocking module must not be able to delay or starve all others.
