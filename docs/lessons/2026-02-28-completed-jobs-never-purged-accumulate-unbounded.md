# Lesson: Completed Jobs Not Purged From Registry — Unbounded Accumulation With No Warning

**Date:** 2026-02-28
**System:** community (home-assistant/supervisor)
**Tier:** lesson
**Category:** performance
**Keywords:** job, task, cleanup, purge, retention, unbounded growth, registry, memory, ha supervisor
**Source:** https://github.com/home-assistant/supervisor/issues/5570

---

## Observation (What Happened)

The HA Supervisor `jobs info` command returned thousands of completed job records with `stage: null`. Jobs were added to the job registry on creation but never removed after completion — the registry accumulated every job ever run without any pruning, causing the output to grow unboundedly and consuming memory over time.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `ha jobs info` returned thousands of completed records that should have been discarded.
**Why #2:** The job registry's `add()` path had no corresponding `cleanup()` / `remove()` call after a job reached `done: true`.
**Why #3:** Job lifecycle management was designed with a create+update pattern; the remove step was assumed to be handled elsewhere but was not.
**Why #4:** No retention policy (TTL, max count, or explicit cleanup after completion) was applied to the registry.
**Why #5:** Long-running services treat runtime registries like databases and routinely omit the cleanup half of the lifecycle.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Every entry added to a runtime registry must have an explicit removal path — either TTL-based, count-capped, or explicit cleanup on job completion | proposed | community | issue #5570 |
| 2 | Add a periodic sweep for any registry or queue that accumulates work items, bounded by a max age or count | proposed | community | issue #5570 |
| 3 | After implementing a job/task registration system, write a test that creates N jobs and verifies the registry size is bounded after all complete | proposed | community | issue #5570 |

## Key Takeaway

Any runtime registry that only has an `add()` path without a `remove()` / TTL policy will grow unboundedly — completed jobs, events, and tasks must have an explicit retention strategy or the registry becomes a memory leak.
