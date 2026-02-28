# Lesson: Vector DB BTREE Index Page Lookup Corrupted After Compaction When Deletions Change Page Distribution

**Date:** 2026-02-28
**System:** community (lancedb/lance)
**Tier:** lesson
**Category:** data-model
**Keywords:** lancedb, lance, BTREE index, compaction, deletion, page lookup, corruption, scalar index, remap
**Source:** https://github.com/lancedb/lance/issues/5826

---

## Observation (What Happened)
A dataset with a BTREE scalar index had ~80% of its rows deleted, then `optimize.compact_files()` was called. After compaction, range queries near deleted value boundaries returned zero results instead of the expected rows. The `a = 10000` filter on a row that existed returned no results because the page lookup pointed to a page that no longer existed post-compaction.

## Analysis (Root Cause — 5 Whys)
**Why #1:** The BTREE index stores a per-page min/max lookup table. During remap (after compaction), deleted rows are removed from page data, but the min/max lookup was not updated.

**Why #2:** Heavy deletions change the distribution of values across pages — pages may merge or disappear — but the index's page count and min/max bounds still reflected the pre-deletion state.

**Why #3:** The remap function assumed row ID changes alone (not row count changes) — so it only re-mapped row IDs, not page boundaries.

**Why #4:** The assumption was valid for small deletions but breaks when deletions cause page boundary shifts (e.g., going from 4 pages to 2 pages).

**Why #5:** No validation step after compaction verifies that scalar index page counts and bounds match the actual post-compact data distribution.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After any bulk delete + compaction workflow, rebuild scalar indexes explicitly (`create_scalar_index(..., replace=True)`) | proposed | community | issue #5826 |
| 2 | Treat compaction as an index-invalidating operation for BTREE indexes when deletion ratio is high (>20% of pages affected) | proposed | community | issue #5826 |
| 3 | Fixed in lance (fix: update BTREE page lookup after compaction) — upgrade to the patched version | resolved | maintainer | issue #5826 |

## Key Takeaway
BTREE scalar index page lookups are invalidated by high-deletion compaction — the index records stale page boundaries; always rebuild scalar indexes after bulk-delete + compact workflows.
