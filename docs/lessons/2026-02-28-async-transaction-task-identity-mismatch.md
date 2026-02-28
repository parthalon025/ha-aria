# Lesson: Async Transaction State Bound to Task Identity — setUp/tearDown in Different Tasks Breaks Transaction Scope

**Date:** 2026-02-28
**System:** community (coleifer/peewee)
**Tier:** lesson
**Category:** integration
**Keywords:** asyncio, transaction, task identity, pytest, setup, teardown, rollback, non-deterministic, aiosqlite, pwasyncio
**Source:** https://github.com/coleifer/peewee/issues/3030

---

## Observation (What Happened)
A pytest suite used an `autouse` fixture to open a transaction in setUp and roll it back in tearDown for test isolation. Tests passed ~50% of the time; the other 50% failed during teardown. The failure was non-deterministic across runs.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Async ORMs (and aiosqlite under the hood) bind the active transaction to the current asyncio `Task` identity — the task that opened the transaction "owns" it.

**Why #2:** pytest-asyncio runs `asyncSetUp`, the test body, and `asyncTearDown` as three separate `Task` objects.

**Why #3:** The teardown task attempts to roll back a transaction opened by a different task (setUp). The transaction state is not visible from the teardown task's perspective.

**Why #4:** The rollback either silently no-ops or intermittently succeeds depending on event loop timing — producing non-deterministic results.

**Why #5:** Developers carry the mental model of sync test fixtures (single thread, single stack) into async code where coroutine tasks have independent execution contexts.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Ensure setUp and tearDown run within the same asyncio `Task` — use a single async context manager that wraps the entire test, not separate fixtures | proposed | community | issue #3030 comment |
| 2 | For transaction-based test isolation in async code, use `async with db.atomic() as txn:` inside the test body itself, not in a fixture lifecycle split across tasks | proposed | community | issue #3030 |
| 3 | Treat non-deterministic test teardown failures as a task-identity bug — investigate which task opened the resource before assuming a race in application code | proposed | community | issue #3030 |

## Key Takeaway
Async transaction state is scoped to the task that opened it — pytest setUp/tearDown fixtures that run as separate tasks cannot share or roll back a transaction opened by another task.
