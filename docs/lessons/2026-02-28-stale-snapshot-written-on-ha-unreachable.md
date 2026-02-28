# Lesson: Unreachable-HA Snapshots Written to Disk Get Deduped as Valid on Next Run

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** data-model
**Keywords:** cold-start, snapshot, HA unreachable, deduplication, data quality, persist, poisoned data
**Files:** aria/engine/collectors/snapshot.py:157-195

---

## Observation (What Happened)

`build_intraday_snapshot()` writes a partial/empty snapshot to disk even when `ha_reachable=False`. On the next run for the same hour, the deduplication check finds the existing file and returns the stale partial snapshot without rebuilding. A single HA outage permanently poisons that hour's data for the day — the dedup guard that prevents redundant API calls also prevents recovery from bad data.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The snapshot write is unconditional — it writes to disk regardless of `ha_reachable`.

**Why #2:** The dedup check (does this hour's file exist?) was designed to prevent redundant collection runs, but it doesn't distinguish between "valid file" and "file written during an outage."

**Why #3:** A partial/empty snapshot has the same filename as a complete one — there is no quality signal in the file name or dedup key to prevent consuming stale data.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Do not write the snapshot file when `ha_reachable=False` — log a WARNING and return the in-memory snapshot without persisting | proposed | Justin | issue #209 |
| 2 | Add a `data_quality.ha_reachable` check in the dedup guard — skip dedup and force rebuild if the existing snapshot was marked unreachable | proposed | Justin | issue #209 |
| 3 | Consider writing to a `.partial` extension and only renaming to final on success | proposed | Justin | — |

## Key Takeaway

Never write a snapshot to disk when the data source was unreachable — the deduplication guard will prevent recovery on subsequent runs, permanently poisoning that time slot's data.
