# Lesson: Transaction COMMIT Failures Can Be Silently Swallowed, Returning Phantom Success Data

**Date:** 2026-02-28
**System:** community (prisma/prisma)
**Tier:** lesson
**Category:** error-handling
**Keywords:** transaction, COMMIT, silent failure, data loss, Promise, PlanetScale, adapter, planetscale, prisma
**Source:** https://github.com/prisma/prisma/issues/29138

---

## Observation (What Happened)

The `@prisma/adapter-planetscale` adapter's `startTransactionInner` method handled the transaction `execute()` promise but did not await the result of `commit()`. When PlanetScale killed a long-running transaction server-side (20s timeout), the COMMIT silently failed. Prisma resolved the operation as successful, returned data from the uncommitted transaction, but nothing was persisted to the database — data loss with no error thrown.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `commit()` call returned a Promise that was not awaited — its rejection was unhandled.
**Why #2:** The adapter's transaction orchestration code awaited `execute()` but treated `commit()` as fire-and-forget.
**Why #3:** The PlanetScale client's COMMIT throws on server-side transaction kill, but the calling code didn't await it.
**Why #4:** No integration test simulated a mid-flight COMMIT failure (server-side transaction timeout).
**Why #5:** The data appeared written (was readable within the uncommitted transaction) before COMMIT, making the silent failure invisible without a subsequent read.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always `await commit()` — never fire-and-forget a transaction commit | proposed | community | prisma#29138 |
| 2 | Treat COMMIT errors as fatal — re-throw, never swallow | proposed | community | prisma#29138 |
| 3 | Add integration test that simulates commit failure and verifies no data is returned | proposed | community | prisma#29138 |
| 4 | Log COMMIT result (success/failure) before returning from transaction handler | proposed | community | prisma#29138 |

## Key Takeaway

A transaction COMMIT that isn't awaited is a silent data-loss trap — always `await commit()`, always treat COMMIT failures as fatal, never return data from a transaction before verifying the commit succeeded.
