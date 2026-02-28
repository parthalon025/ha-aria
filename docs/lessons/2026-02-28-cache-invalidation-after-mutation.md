# Lesson: Stale Cache After User-Triggered Mutation — No Invalidation on POST

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** cache invalidation, stale data, mutation, POST, retrain, useCache, refetch, Preact, optimistic update
**Files:** aria/dashboard/spa/src/pages/Predictions.jsx

---

## Observation (What Happened)

`Predictions.jsx` cached model metrics via `useCache`. After the user triggered a manual retrain via `POST /api/ml/retrain`, the page continued showing pre-retrain metrics — the successful POST response did not invalidate or refetch the metrics cache (issue #290).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `POST` handler awaited the retrain response and reported success, but took no action on the local cache state.

**Why #2:** The `useCache` hook was treated as read-only — the component had no mechanism to signal "this cache key is now stale."

**Why #3:** The developer modeled reads and writes as independent — GET on mount, POST on button click, with no explicit coupling between write-success and cache freshness.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After a successful mutation POST, either refetch the affected cache key or call a cache invalidation method on the relevant `useCache` hook | proposed | Justin | Predictions.jsx #290 |
| 2 | Establish a pattern: every mutation handler that touches server state must either invalidate or optimistically update the local cache | proposed | Justin | — |

## Key Takeaway

A successful POST does not automatically invalidate related GET caches — every mutation handler must explicitly refetch or invalidate any local cache it affects.
