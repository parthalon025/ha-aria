# Lesson: SQLite SQLITE_BUSY on Concurrent Writers — Default Timeout Too Short

**Date:** 2026-02-28
**System:** community (omnilib/aiosqlite)
**Tier:** lesson
**Category:** data-model
**Keywords:** sqlite, concurrent, SQLITE_BUSY, database is locked, timeout, aiosqlite, per-connection
**Source:** https://github.com/omnilib/aiosqlite/issues/234

---

## Observation (What Happened)
A developer ran 380 concurrent asyncio coroutines each opening their own aiosqlite connection and executing an INSERT. Above ~380 concurrent writers they received `sqlite3.OperationalError: database is locked` with no timeout error — the SQLite default 5-second busy timeout was being exhausted because hundreds of connections were queued waiting for the write lock.

## Analysis (Root Cause — 5 Whys)
**Why #1:** SQLite supports only one active writer at a time; concurrent write transactions block on a file-level lock.

**Why #2:** Each of the 380 coroutines opened an independent connection and began a write transaction immediately.

**Why #3:** The last connection in the queue waited longer than the default sqlite3 busy timeout (5 seconds) for the lock to be released.

**Why #4:** The developer assumed aiosqlite provided a connection pool that would serialize writes gracefully; it does not — aiosqlite is a thin async wrapper, not a pool.

**Why #5:** SQLite's "database is locked" error surfaces as OperationalError, which is easily confused with a programming error rather than a contention signal.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use a single shared long-lived connection for all writes instead of one connection per coroutine | proposed | community | issue #234 — 5x throughput improvement observed |
| 2 | Set `timeout=` on `sqlite3.connect()` / `aiosqlite.connect()` to extend busy wait (e.g., 30s) when multiple writers are unavoidable | proposed | community | issue #234 comment |
| 3 | Apply concurrency cap (`aioitertools.asyncio.gather(..., limit=N)`) to bound simultaneous writers | proposed | community | issue #234 comment |

## Key Takeaway
aiosqlite is not a connection pool — opening a new connection per write under high concurrency triggers SQLITE_BUSY; use one shared write connection and serialize writes through it.
