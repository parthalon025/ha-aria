# Lesson: LanceDB Index Silently Disappears After merge_insert Into an Empty Table

**Date:** 2026-02-28
**System:** community (lancedb/lancedb)
**Tier:** lesson
**Category:** data-model
**Keywords:** lancedb, index, merge_insert, empty table, scalar index, BTREE, disappears, silent data loss
**Source:** https://github.com/lancedb/lancedb/issues/2661

---

## Observation (What Happened)
A developer created a scalar BTREE index on an empty table, confirmed it existed via `list_indices()`, then called `merge_insert(...).when_not_matched_insert_all().execute([row])`. After the call, `list_indices()` returned an empty list — the index was gone. When the same sequence was run on a non-empty table, the index was preserved.

## Analysis (Root Cause — 5 Whys)
**Why #1:** `merge_insert` into an empty table takes a code path equivalent to overwrite — it writes an entirely new dataset version, replacing the empty initial version.

**Why #2:** The new version did not inherit the index metadata from the pre-data version because no rows existed to anchor the index.

**Why #3:** Lance's versioning model stores index metadata alongside data fragments; an overwrite-like merge on an empty table creates a fresh version with no inherited index artifacts.

**Why #4:** The API does not warn that indexes created before first data insertion are at risk; the semantics are only documented at the Lance protocol level.

**Why #5:** Developers create indexes before inserting data as a one-time setup step, expecting the index to survive subsequent data loads.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Create scalar and vector indexes only after the table has at least one row of data | proposed | community | issue #2661 |
| 2 | After any `merge_insert`, `overwrite`, or bulk load operation, verify `list_indices()` and rebuild missing indexes | proposed | community | issue #2661 |
| 3 | Fixed in LanceDB 0.24.3+ (Lance PR #4033) — pin to >= 0.24.3 when using merge_insert workflows | resolved | maintainer | issue #2661 |

## Key Takeaway
LanceDB indexes created on empty tables are silently discarded by the first `merge_insert` — create indexes only after initial data is present, and verify `list_indices()` after any bulk write operation.
