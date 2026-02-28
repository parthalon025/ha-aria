# Lesson: Async Signal Updates Inside useEffect Are Invisible to Test `act()`

**Date:** 2026-02-28
**System:** community (preactjs/signals)
**Tier:** lesson
**Category:** testing
**Keywords:** preact, signals, useSignalEffect, useEffect, async, act, microtask, test, unit test, timing, promise
**Source:** https://github.com/preactjs/signals/issues/636

---

## Observation (What Happened)

A test using `act()` to wrap an async `useEffect` that awaits a Promise before assigning to a signal fails: `useSignalEffect` (which subscribes to that signal) never fires within the `act()` boundary. The test passes at runtime but fails in the test environment.

Root cause: in `@preact/signals` v2+, signal updates that happen after a micro-task `await` inside `useEffect` are scheduled outside the synchronous `act()` flush. The `act()` call exits before the Promise microtask resolves, so the signal write and its downstream `useSignalEffect` callback are never executed within the test's flush window.

The same behavior affects plain `useState` + async `useEffect` under newer React/Preact test utils — it is a general async boundary issue, not signals-specific.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `useSignalEffect` callback does not execute in test, even though it works at runtime.

**Why #2:** The signal is only written after an `await new Promise(...)` inside a `useEffect`. The Promise resolves as a microtask after `act()` exits.

**Why #3:** `act()` flushes synchronous state updates and effects, but does not automatically drain all pending microtasks queued after the initial flush.

**Why #4:** In signals v2, signal writes from async contexts are batched and scheduled differently than synchronous ones — the transition from sync-to-async in `useEffect` breaks the `act()` timing contract.

**Why #5:** Mixing async code (awaited Promises) inside `useEffect` with reactive signal subscriptions (`useSignalEffect`) creates a test-invisible update path that only works in real browser environments with their full event loop.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap the full async test with `await act(async () => { ... })` — passing an async function to `act` waits for all pending microtasks and scheduled effects before asserting. | proposed | community | https://github.com/preactjs/signals/issues/636 |
| 2 | If `useSignalEffect` is not firing in tests, check whether the signal is written synchronously or after an `await`. Async writes require `await act(async () => {})` not `act(() => {})`. | proposed | community | https://github.com/preactjs/signals/issues/636 |
| 3 | Treat async `useEffect` + signal write as an anti-pattern in test-critical code paths. Extract the async fetch into a separate function and call it synchronously in tests using a mocked/resolved dependency. | proposed | community | https://github.com/preactjs/signals/issues/636 |
| 4 | When a test fails on signal reaction that works at runtime, first verify: is the signal write behind an `await`? This is the most common cause. | proposed | community | https://github.com/preactjs/signals/issues/636 |

## Key Takeaway

Signal writes that happen after an `await` inside `useEffect` are scheduled outside `act()`'s synchronous flush — use `await act(async () => {})` (async act) in tests, or the signal update and its reactive effects will never execute, causing silent test failures.
