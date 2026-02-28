# Lesson: Mixing Timezone-Aware and Naive Datetimes Produces Silent TypeError in Comparisons

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** datetime, aware, naive, UTC, TypeError, timezone, presence, Frigate, comparison, flush
**Files:** aria/modules/presence.py:404, presence.py:550, shadow_engine.py:337, time_features.py:36

---

## Observation (What Happened)

`presence.py` uses `datetime.now(UTC)` (aware) on the Frigate/MQTT path but `datetime.now()` (naive) on HA state change handlers. When `_flush_presence_state()` runs every 30 seconds, it compares timestamps from both paths: `ts >= cutoff` raises `TypeError: can't compare offset-naive and offset-aware datetimes`. `schedule_task`'s error handler swallows this — presence tracking silently stops on any Frigate-connected install after the first camera event.

The same naive datetime problem exists in `shadow_engine.py` (burst detection, cooldown) and `time_features.py` (sun feature fallback).

## Analysis (Root Cause — 5 Whys)

**Why #1:** Two code paths in the same module use different datetime conventions; comparison across paths raises `TypeError`.

**Why #2:** The error is swallowed by the task error handler (which logs a string, not a traceback) — there is no visible symptom until the user notices presence tracking has stopped producing output.

**Why #3:** Naive datetimes are the Python default; developers adding new code paths in modules where some paths already use `datetime.now(UTC)` can easily miss the convention.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace ALL `datetime.now()` calls with `datetime.now(UTC)` across presence.py, shadow_engine.py, time_features.py | proposed | Justin | issues #305, #253, #210 |
| 2 | Add a module-level convention comment: "All timestamps in this module use `datetime.now(UTC)` — never `datetime.now()`" | proposed | Justin | — |
| 3 | Add a lint rule or type check to catch `datetime.now()` without `tz=` argument in hub modules | proposed | Justin | — |

## Key Takeaway

In any module where multiple code paths produce timestamps that are later compared, all timestamps must use the same timezone convention — a single naive datetime in a pool of aware datetimes raises `TypeError` silently at comparison time.
