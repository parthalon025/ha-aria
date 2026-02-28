# Lesson: Bulk Selection State Leaks Across Pagination — Not Reset on Navigation

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** component state, pagination, navigation, bulk select, stale state, useEffect, dependency array, Preact
**Files:** aria/dashboard/spa/src/pages/DataCuration.jsx

---

## Observation (What Happened)

`DataCuration.jsx` maintained a bulk-selection set in component state. Navigating between pages of a paginated list did not clear the selection — items selected on page 1 remained selected when viewing page 2, causing confusing cross-page selections that users could not see (issue #291).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `useEffect` that resets selection state did not include `page` or `filter` in its dependency array.

**Why #2:** The developer added a reset effect for navigation (unmount/remount) but not for in-page context changes like pagination.

**Why #3:** Selection state was modeled as belonging to the component lifecycle, not to the current data context (page + filter combination) — a mismatch between how the state was scoped and how the UI would be used.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `page` and `filter` to the `useEffect` dependency array that resets the selection set | proposed | Justin | DataCuration.jsx #291 |
| 2 | Treat selection state as scoped to the current view context — any context change (page, filter, sort) must reset selection | proposed | Justin | — |

## Key Takeaway

Bulk selection state must be scoped to the current view context (page + filter), not just the component lifecycle — add all context-change dependencies to the reset effect.
