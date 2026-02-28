# Lesson: useComputed Runs on Every Render Even When Tracked Signals Are Unchanged

**Date:** 2026-02-28
**System:** community (preactjs/signals)
**Tier:** lesson
**Category:** performance
**Keywords:** preact, signals, useComputed, computed, recompute, closure, memoization, useCallback, stale closure, render
**Source:** https://github.com/preactjs/signals/issues/772

---

## Observation (What Happened)

`useComputed(() => expensiveFn(signal.value))` in `@preact/signals-react` v2+ re-runs on every component re-render — even when none of the signals tracked inside the callback have changed. An expensive computation keyed only on `count.value` also fired when an unrelated `newTask` signal changed, defeating the purpose of `useComputed`.

This is intentional: signals v2 re-runs the computed function on each render to keep the closure fresh (React's model), but does not propagate the new computed value downstream if the result did not change. However, if the computed function returns a new object reference on every call (even with equal semantic content), any downstream code using reference equality will re-trigger on every render.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `useComputed` fires an expensive calculation on every render regardless of signal state.

**Why #2:** Signals v2 changed the semantics of `useComputed` to re-execute the callback on each render cycle (to avoid stale closures), as opposed to v1 which only re-ran on signal graph changes.

**Why #3:** The change was shipped as a non-major version bump (minor/patch), breaking applications that relied on computed signals only re-running when their signal dependencies changed.

**Why #4:** The official documentation still shows the old pattern (`useComputed(() => fn(signal.value))`) without noting the memoization requirement for expensive callbacks.

**Why #5:** The new behavior cannot be avoided without wrapping the callback in `useCallback`, which is not mentioned in any documentation and must be discovered by debugging or reading issues.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap the `useComputed` callback in `useCallback` when computation is expensive or produces object references: `useComputed(useCallback(() => fn(signal.value), []))`. | proposed | community | https://github.com/preactjs/signals/issues/772#issuecomment-3499064223 |
| 2 | If `useComputed` returns a new object on every call and you rely on reference stability (e.g., for child memo bailouts), switch to a plain `signal` + `useEffect` that updates it: the computed callback is not stable across renders. | proposed | community | https://github.com/preactjs/signals/issues/789 |
| 3 | When upgrading `@preact/signals` across minor versions, audit every `useComputed` call that produces objects or performs expensive work — treat them as potentially broken. | proposed | community | https://github.com/preactjs/signals/issues/789 |

## Key Takeaway

`useComputed` in `@preact/signals-react` v2+ re-executes its callback on every render (not just on signal changes) to keep closures fresh — wrap expensive callbacks in `useCallback` or the performance benefit is zero.
