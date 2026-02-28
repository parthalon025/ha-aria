# Lesson: useEffect Poll Interval Not Cleared on Unmount — Memory Leak

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** useEffect, cleanup, setInterval, clearInterval, memory leak, unmount, stale state, Preact, poll, timer
**Files:** aria/dashboard/spa/src/components/ActivityFeed.jsx

---

## Observation (What Happened)

`ActivityFeed.jsx` started a `setInterval` poll inside a `useEffect` but returned nothing from the effect — no cleanup function. The interval continued running after the component unmounted, producing stale state updates and a growing set of orphaned timers (issue #282).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `useEffect` call that created the interval had no return value, so Preact/React had no callback to invoke on unmount.

**Why #2:** The interval ID was never stored in a variable accessible to a cleanup function — it was created inline without capturing the reference.

**Why #3:** The developer treated `useEffect` as a mount-only side effect, not as a setup/teardown pair — the cleanup contract of `useEffect` was not applied.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Return `() => clearInterval(intervalId)` from every `useEffect` that calls `setInterval` or `setTimeout` | proposed | Justin | ActivityFeed.jsx #282 |
| 2 | Treat every `useEffect` that creates a resource (timer, subscription, observer) as having a mandatory cleanup return | proposed | Justin | — |

## Key Takeaway

Every `useEffect` that creates a timer must return `() => clearInterval(id)` — no return means no teardown, and the timer outlives the component forever.
