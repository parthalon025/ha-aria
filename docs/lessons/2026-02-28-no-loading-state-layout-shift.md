# Lesson: Missing Loading Skeleton Causes Layout Shift on Data Arrival

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** loading state, skeleton, placeholder, layout shift, CLS, Preact, null data, render, MetricCard
**Files:** aria/dashboard/spa/src/components/MetricCard.jsx

---

## Observation (What Happened)

`MetricCard.jsx` rendered immediately with no loading placeholder. When data arrived from the API, the layout shifted as content was inserted — producing a jarring visual jump (issue #284). Components consuming the card also had no guard for `data === null`, rendering partial or broken states during the async window.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The component rendered its full structure immediately, using `data` before it was populated.

**Why #2:** There was no `if (data === null) return <Skeleton />` early-return guard in the component.

**Why #3:** The developer assumed the data would arrive before the first paint, or did not model the asynchronous loading window as a distinct render state requiring explicit handling.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add an early-return skeleton render for `data === null` in MetricCard and all data-dependent components | proposed | Justin | MetricCard.jsx #284 |
| 2 | Reserve the same fixed dimensions in the skeleton as the fully-loaded state to prevent layout shift | proposed | Justin | — |
| 3 | Treat null/loading as a first-class render state — not an edge case | proposed | Justin | — |

## Key Takeaway

Every component that renders async data must have a fixed-dimension skeleton for the `null` loading state — rendering nothing until data arrives guarantees layout shift.
