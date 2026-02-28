# Lesson: Scheduler Polling Ignored retry_after Field — Failed Jobs Immediately Requeued

**Date:** 2026-02-28
**System:** ollama-queue
**Tier:** lesson
**Category:** reliability
**Keywords:** retry_after, backoff, scheduler, polling, queue, job, failed, cooldown, requeue, immediate retry, field ignored

---

## Observation (What Happened)

The ollama-queue job scheduler's `get_next_job()` query did not filter on the `retry_after` timestamp. Failed jobs that had been given an explicit backoff delay were immediately returned on the next polling cycle, defeating the backoff entirely and causing thundering-herd retry storms against Ollama.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `get_next_job()` selected the next pending job by status alone, without a `WHERE retry_after IS NULL OR retry_after <= now()` clause.
**Why #2:** The `retry_after` column was added to the schema to record backoff state, but the dequeue query was never updated to read it.
**Why #3:** Schema and query are in different files/layers; adding a column does not automatically surface the need to update all consumers of that table.
**Why #4:** No test exercised the scenario of a failed job being polled before its `retry_after` deadline — the backoff path was untested end-to-end.
**Why #5:** The dequeue behavior was only validated by "does a job eventually run?", not "does a job with active backoff stay invisible to the scheduler?"

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `AND (retry_after IS NULL OR retry_after <= datetime('now'))` to `get_next_job()` query | proposed | Justin | ollama-queue #1 |
| 2 | Add a test: enqueue a failed job with `retry_after = now + 10s`, poll immediately, assert no job returned | proposed | Justin | — |
| 3 | Apply schema change rule: adding a column that affects query semantics requires updating all query consumers in the same commit | proposed | Justin | — |

## Key Takeaway

A backoff field in the schema is inert until the dequeue query explicitly enforces it — adding a column and updating all consumers must happen in the same change.
