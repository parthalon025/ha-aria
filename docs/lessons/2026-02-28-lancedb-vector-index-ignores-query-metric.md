# Lesson: Vector Index Built with One Distance Metric Is Silently Used for Queries Requesting a Different Metric

**Date:** 2026-02-28
**System:** community (lancedb/lance)
**Tier:** lesson
**Category:** data-model
**Keywords:** lancedb, lance, vector index, metric, distance, dot product, l2, cosine, ANN, silent wrong results, IVF_PQ
**Source:** https://github.com/lancedb/lance/issues/5608

---

## Observation (What Happened)
A developer built an IVF_PQ index with `metric="dot"`, then queried with `metric="l2"`. The ANN index was used regardless of the requested metric — producing dot-product distances for an l2 query. All three metrics (l2, dot, cosine) returned identical distances when the query should have produced different rankings.

## Analysis (Root Cause — 5 Whys)
**Why #1:** The query planner checked only whether an ANN index existed on the target column, not whether the index metric matched the requested query metric.

**Why #2:** The `explain_plan()` output showed the ANN index being used for all metric types — the planner had no metric-compatibility gate.

**Why #3:** Developers assume that specifying `metric=` on a query will either use a compatible index or fall back to full scan — neither happened; the wrong index was used silently.

**Why #4:** The test suite for ANN did not include cross-metric consistency checks (build with metric A, query with metric B, assert fallback to brute force).

**Why #5:** The bug manifested only at scale — small collections were fast enough on brute force that the index path was not critical, masking the issue in development.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Build separate indexes for each metric you intend to query — one index per (column, metric) combination | proposed | community | issue #5608 |
| 2 | After creating an index, verify with `explain_plan()` that the query metric matches the index metric | proposed | community | issue #5608 |
| 3 | Fixed in lance (critical-fix tag) — upgrade to the patched version before deploying multi-metric search pipelines | resolved | maintainer | issue #5608 |

## Key Takeaway
A vector ANN index built with metric M is silently reused for queries requesting a different metric, producing wrong distances — always build one index per distance metric per column, and verify metric alignment with `explain_plan()`.
