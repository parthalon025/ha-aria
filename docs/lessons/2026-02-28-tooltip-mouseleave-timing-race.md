# Lesson: Tooltip Stays Visible After Fast Mouseleave — No Pending Timeout Cancellation

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** tooltip, mouseleave, mouseover, timing, clearTimeout, useRef, race condition, hover, Preact, event handler
**Files:** aria/dashboard/spa/src/components/PipelineSankey.jsx

---

## Observation (What Happened)

`PipelineSankey.jsx` showed a tooltip on `mouseover` using a delayed display (setTimeout or direct show). The `mouseleave` handler dismissed it, but fast mouse movements could trigger `mouseover` on one element, then `mouseleave` before the show timeout fired — leaving the tooltip visible indefinitely, overlapping other UI (issue #278).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The tooltip show path used a timeout (or synchronous show), but the hide path only dismissed the visible tooltip — it did not cancel any pending show timeout.

**Why #2:** The timeout ID was not stored in a ref accessible to both the `mouseover` and `mouseleave` handlers.

**Why #3:** The developer handled show and hide as independent event responses, not as a cancellable pair — the async show had no corresponding abort path.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Store the show-timeout ID in a `useRef` accessible to both event handlers | proposed | Justin | PipelineSankey.jsx #278 |
| 2 | On `mouseleave`, call `clearTimeout(timeoutRef.current)` before hiding the tooltip | proposed | Justin | — |
| 3 | Pattern: every delayed-show UI element must have a cancellation path that fires on the corresponding leave/close event | proposed | Justin | — |

## Key Takeaway

A delayed tooltip show must store its timeout ID in a ref and cancel it on `mouseleave` — without cancellation, fast mouse movements leave orphaned timeouts that show the tooltip after the cursor has already left.
