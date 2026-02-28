# Lesson: ChromaDB Bulk Delete Fails with "Too Many SQL Variables" on Large Filter Results

**Date:** 2026-02-28
**System:** community (chroma-core/chroma)
**Tier:** lesson
**Category:** performance
**Keywords:** chromadb, delete, bulk, sql variables, SQLite, sqlite limit, too many, SQLITE_MAX_VARIABLE_NUMBER, metadata filter
**Source:** https://github.com/chroma-core/chroma/issues/4802

---

## Observation (What Happened)
A developer called `collection.delete(where={"workspace_id": {...}})` on a collection where the WHERE clause matched 8,239 rows. ChromaDB raised `InternalError: Query error: Database error: error returned from database: (code: 1) too many SQL variables`. The operation succeeded for smaller workspaces but failed at scale.

## Analysis (Root Cause — 5 Whys)
**Why #1:** ChromaDB resolves the metadata WHERE filter to a list of matching IDs, then generates a SQLite `IN (?, ?, ?, ...)` clause with one parameter per ID.

**Why #2:** SQLite has a hard per-query compile-time limit on bound parameters (`SQLITE_MAX_VARIABLE_NUMBER`, default 999 in older builds, 32766 in newer).

**Why #3:** The 8,239-item result set far exceeded this limit; the DELETE was not batched.

**Why #4:** The API surface (`collection.delete(where=...)`) gives no indication that the underlying implementation has a cardinality limit.

**Why #5:** This pattern is invisible in development/staging environments where workspace sizes are small.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Chunk large deletes by ID batch (e.g., `collection.delete(ids=ids[i:i+500])`) rather than using a single metadata WHERE clause | proposed | community | issue #4802 |
| 2 | For any SQLite-backed system, apply chunked IN-clause queries when the set size is unbounded; use a max of 500-900 IDs per batch | proposed | community | issue #4802 |
| 3 | Fixed in ChromaDB 1.0.15 — upgrade rather than implementing client-side workaround when on older versions | resolved | maintainer | issue #4802 |

## Key Takeaway
SQLite `IN (...)` clauses with thousands of parameters exceed SQLite's SQLITE_MAX_VARIABLE_NUMBER limit — any bulk operation resolved to a list of IDs must be chunked into batches of <999 (safe) or <500 (conservative).
