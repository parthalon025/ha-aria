# Lesson: API Error Response and Empty Data Are Indistinguishable When safeFetch Returns null

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** safeFetch, error handling, empty state, null, 500, 404, indistinguishable, error vs empty, API contract, silent failure, Preact
**Files:** aria/dashboard/spa/src/api.js

---

## Observation (What Happened)

`safeFetch` in `api.js` (lines 88-92) returned `null` for all error responses — including 500, 503, and network timeouts — while also returning `null` for genuinely empty data sets. Components consuming `safeFetch` had no way to distinguish "the API errored" from "the API returned nothing" — both produced the same empty/blank render with no user-visible error state (issues #266, #293).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The catch block returned `null` unconditionally without preserving any signal about why the data was absent.

**Why #2:** The function was designed with only the happy path and 404 in mind — no error object or status code was threaded through to callers.

**Why #3:** The developer conflated "no data" (valid server state) with "fetch failed" (error) — both were collapsed into the same null sentinel, making the component's empty-state and error-state renders identical.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Return a typed error object (e.g. `{ error: true, status, message }`) for non-404 failures instead of `null` | proposed | Justin | api.js #266 |
| 2 | Re-throw or propagate non-404 errors so callers that `await safeFetch` can distinguish error from empty | proposed | Justin | api.js #293 |
| 3 | Components should render distinct states: loading / empty (no data) / error (fetch failed) — never collapse error into empty | proposed | Justin | — |

## Key Takeaway

A fetch utility that returns `null` for both errors and empty responses makes server failures invisible — distinguish them with a typed error sentinel so components can show error UI instead of silently rendering nothing.
