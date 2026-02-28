# Lesson: BigInt Values Crash JSON.stringify Silently in Internal Caching

**Date:** 2026-02-28
**System:** community (drizzle-team/drizzle-orm)
**Tier:** lesson
**Category:** error-handling
**Keywords:** BigInt, JSON.stringify, serialization, cache, TypeError, crash, TypeScript, drizzle
**Source:** https://github.com/drizzle-team/drizzle-orm/issues/5227

---

## Observation (What Happened)

Drizzle ORM's internal cache layer calls `JSON.stringify()` on all query parameters including those containing BigInt values. Any query using `inArray("column", arrayOfBigInts)` throws `TypeError: Cannot serialize BigInt` — not at the application layer, but deep inside an internal cache utility, making the stack trace misleading.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `JSON.stringify` does not support BigInt natively — it throws rather than coercing.
**Why #2:** The cache key hash function (`hashQuery`) accepted arbitrary query parameters without sanitizing types.
**Why #3:** BigInt was added to the query API as a valid type, but the serialization path was never updated to handle it.
**Why #4:** No type guard or serialization test existed for BigInt in the cache layer.
**Why #5:** The assumption that "all query parameters are JSON-serializable" is embedded as an undocumented invariant with no enforcement.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Provide a custom `JSON.stringify` replacer that converts BigInt to string | proposed | community | issue #5227 |
| 2 | Add type guard: before any `JSON.stringify()`, verify value is serializable | proposed | community | issue #5278 |
| 3 | Write a test that passes BigInt through every serialization path | proposed | community | issue #5278 |

## Key Takeaway

`JSON.stringify` silently fails on BigInt, Symbol, undefined (in objects), and circular refs — validate serializability at the boundary where data enters a serialization path, not where it exits.
