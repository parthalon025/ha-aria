# Lesson: Router Context Memoization Fails to Update When Value Resets to Empty String

**Date:** 2026-02-28
**System:** community (molefrog/wouter)
**Tier:** lesson
**Category:** frontend
**Keywords:** preact, react, wouter, router, memoization, useMemo, context, base, empty string, falsy, stale, navigation
**Source:** https://github.com/molefrog/wouter/issues/531

---

## Observation (What Happened)

A wouter `<Router>` with a dynamically changing `base` prop (e.g., `/uk` when a language is selected, `''` for English) fails to render any routes when switching back to the empty-string base. The router context retains the previous non-empty value (`/uk`) even though `base` is now `''`. Navigating between non-empty bases works correctly; the bug only appears when the new value is falsy (empty string).

Root cause: wouter's internal router value memoization used a comparison like `if (!base)` or checked `base` as a truthy condition before re-creating the context object. An empty string (`''`) is falsy in JavaScript, so the reset to an empty base was treated as "no change" and the memo was not invalidated. A custom `hook` prop on `<Router>` was also required to trigger the bug — the hook prevented base from being resolved through the default path.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Routes stop rendering after switching back to the English (empty base) locale.

**Why #2:** The router context still holds the previous `/uk` base string after base resets to `''`.

**Why #3:** The memoization logic for the router value object used a falsy check on `base`, treating `''` as the absence of a base change rather than an explicit assignment.

**Why #4:** JavaScript's falsy coercion treats `''`, `0`, `null`, `undefined` identically in `if (base)` guards — a reset to `''` is indistinguishable from "not set."

**Why #5:** The custom `hook` prop broke the default base-resolution path, causing the empty-string base to never propagate through the router context update.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | In router/context value memoization: use `=== undefined` or `!== currentBase` instead of truthiness checks (`if (!base)`) — empty string is a valid, semantically meaningful base value. | resolved | molefrog/wouter v3.7.1 | https://github.com/molefrog/wouter/issues/531 |
| 2 | When a context value can legitimately be an empty string, never gate its update on a truthy check — always compare by identity (`!== prev`). | proposed | community | https://github.com/molefrog/wouter/issues/531 |
| 3 | Test router base transitions to empty string explicitly in unit tests — this edge case is missed by tests that only cover non-empty-to-non-empty transitions. | proposed | community | https://github.com/molefrog/wouter/issues/531 |

## Key Takeaway

Using a falsy check (`if (!base)`) to gate context re-memoization silently drops updates when `base` resets to `''` — empty string is a valid state value and must be compared with strict equality (`!== prev`), not truthiness.
