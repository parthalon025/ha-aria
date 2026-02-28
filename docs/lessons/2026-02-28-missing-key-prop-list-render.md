# Lesson: Missing key Prop on List Render Causes Silent Reconciliation Failures

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** key prop, list render, reconciliation, Preact, React, DOM update, performance, reorder, warning
**Files:** aria/dashboard/spa/src/components/CapabilityCard.jsx

---

## Observation (What Happened)

`CapabilityCard.jsx` rendered a list of items without stable `key` props on each element. This produced Preact reconciliation warnings and could cause incorrect DOM updates when the list reordered — the framework could not efficiently identify which items changed (issue #280).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `.map()` call produced JSX elements without a `key` attribute on each item.

**Why #2:** In development mode, the framework emits a warning; in production the bug is silent — incorrect DOM recycling can cause state to bleed between list items.

**Why #3:** The developer treated `key` as optional or decorative, not understanding its role as the identity anchor for the virtual DOM diffing algorithm.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `key={item.id}` (or another stable unique identifier) to every element produced by `.map()` in JSX | proposed | Justin | CapabilityCard.jsx #280 |
| 2 | Treat index-as-key (`key={i}`) as a last resort — it prevents the warning but does not fix reorder correctness | proposed | Justin | — |

## Key Takeaway

Every `.map()` in JSX must produce elements with a stable `key` prop from a data-derived unique identifier — index-as-key masks the problem without fixing the reconciliation correctness issue.
