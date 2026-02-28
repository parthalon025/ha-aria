# Lesson: DB Connections Created in Framework-Managed Threads Are Not Cleaned Up — Stale Connection Leaks

**Date:** 2026-02-28
**System:** community (slackapi/bolt-python)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** Django ORM, thread, DB connection, cleanup, thread-local, unmanaged thread, stale connection, connection leak, worker thread, finally
**Source:** https://github.com/slackapi/bolt-python/issues/280

---

## Observation (What Happened)

Bolt for Python's listener runner spawned threads for async `ack()` handling. These threads used Django ORM models, which automatically create per-thread DB connections. However, because the threads were created by Bolt (not Django's request lifecycle), Django's `request_finished` signal never fired — the DB connections created in those threads were never closed, accumulating as stale connections.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Stale DB connections and eventual connection pool exhaustion under high load.
**Why #2:** Django ORM created a new connection per thread but only auto-closes connections on `request_finished` signal — which only fires for Django-managed request threads.
**Why #3:** Bolt's listener threads are unmanaged by Django — they live outside the request/response lifecycle.
**Why #4:** No `finally` block or completion callback in Bolt's thread runner called `django.db.connections.close_all()`.
**Why #5:** Framework thread spawning patterns do not account for ORM lifecycle hooks in other frameworks they interact with.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Any thread spawned outside a framework's managed lifecycle that uses that framework's ORM must call `connections.close_all()` (or equivalent) in its `finally` block | proposed | community | issue #280 |
| 2 | Add a `completion_handler` / `finally` callback to thread runner implementations, invoked regardless of success or error | proposed | community | issue #280 |
| 3 | When mixing two frameworks (e.g., FastAPI + SQLAlchemy, Celery + Django ORM), audit what each framework expects from thread lifecycle and add explicit cleanup for the gaps | proposed | community | issue #280 |

## Key Takeaway

DB connections created in unmanaged threads are never cleaned up by the ORM's built-in lifecycle hooks — any thread spawned outside the framework's request/response cycle must explicitly close DB connections in a `finally` block, or connections accumulate as a silent leak.
