# Lesson: Distributed Lock TTL Must Exceed the Worst-Case Operation Duration

**Date:** 2026-02-28
**System:** community (langgenius/dify)
**Tier:** lesson
**Category:** integration
**Keywords:** distributed lock, Redis, TTL, expiry, concurrent migration, Kubernetes, Alembic, race condition, LockNotOwnedError, deployment
**Source:** https://github.com/langgenius/dify/issues/32297

---

## Observation (What Happened)

Dify used a Redis lock with a fixed 60-second TTL to prevent concurrent database migrations across Kubernetes pods. When a migration took longer than 60 seconds (large tables, slow DB under load), the lock expired while the first migration was still running. A second pod acquired the lock and started a concurrent migration, causing deadlocks and partial schema upgrades. Worse, when the migration attempt finished and tried to release the lock, it raised `LockNotOwnedError` — which was not caught cleanly — masking the real underlying migration failure in logs.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The Redis lock TTL was hardcoded to 60 seconds.
**Why #2:** The actual migration duration in production (slow DB, large tables, Kubernetes resource constraints) was not measured and compared against the TTL.
**Why #3:** The lock error handler caught `LockNotOwnedError` but surfaced it as the primary error, burying the real migration exception.
**Why #4:** The lock release was unconditional in the `finally` block — it did not check ownership before releasing.
**Why #5:** The migration strategy (every service tries to migrate on startup) creates a thundering herd that is inherently racy under Kubernetes rolling updates; the lock was a workaround for an architectural problem.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Set the lock TTL to at least 10× the observed p95 migration duration, or use an auto-extending lock (e.g., Redis Redlock with heartbeat) | proposed | community | https://github.com/langgenius/dify/issues/32297 |
| 2 | Check lock ownership before release in `finally`; if not owner, log the real migration error first | proposed | community | issue |
| 3 | Move migrations out of service startup into a dedicated Kubernetes init-container or job that runs exactly once per deployment | proposed | community | issue |

## Key Takeaway

A distributed lock whose TTL is shorter than the operation it guards provides false mutual exclusion — always measure worst-case operation duration before setting the lock TTL, and use ownership-checked release to prevent `LockNotOwnedError` from masking the actual failure.
