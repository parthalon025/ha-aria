# Lesson: Vector DB Scalar Index Returns Extra NULL Rows on OR Predicate — Three-Valued Logic Not Applied

**Date:** 2026-02-28
**System:** community (lancedb/lance)
**Tier:** lesson
**Category:** data-model
**Keywords:** lancedb, lance, BTREE index, scalar index, NULL, OR predicate, three-valued logic, incorrect results, filter
**Source:** https://github.com/lancedb/lance/issues/5895

---

## Observation (What Happened)
A LanceDB table with a BTREE scalar index on a nullable `int64` column had one NULL and one non-NULL row. Applying the filter `(c1 != 0) OR (c1 < 5)` returned 2 rows via the index path but only 1 row via full scan (the NULL row was incorrectly included in the index path result).

## Analysis (Root Cause — 5 Whys)
**Why #1:** The scalar index uses bitmask operations internally: AllowList (TRUE rows) and BlockList (FALSE rows). NULL rows are neither.

**Why #2:** The OR operation of AllowList | BlockList incorrectly promoted NULL rows into the result set — the bit was set in the combined mask even though it should remain NULL (excluded).

**Why #3:** SQL three-valued logic (TRUE / FALSE / NULL) was not preserved in the bitmask combination: `AllowList | BlockList` treated BlockList NULLs as if they were FALSE — convertible to TRUE under OR.

**Why #4:** The bug was in `NullableRowAddrMask`'s OR implementation in Rust — an edge case not covered by unit tests for the NULL/NULL combination.

**Why #5:** Index-accelerated queries and full-scan queries returned different row counts with no warning — a silent correctness divergence.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Validate query results by spot-checking `to_table(filter=expr).num_rows` against full-scan results when NULL values are present in indexed columns | proposed | community | issue #5895 |
| 2 | For nullable columns used in OR predicates, temporarily disable index with `fast_search=False` until a patched version is deployed | proposed | community | issue #5895 |
| 3 | Treat any mismatch between indexed and non-indexed filter results as a three-valued logic bug — file an upstream issue | proposed | community | issue #5895 |

## Key Takeaway
Scalar index OR predicates on nullable columns can silently return extra rows (NULL rows treated as matching) due to three-valued logic not being preserved in bitmask operations — always cross-check nullable column filters against full-scan results.
