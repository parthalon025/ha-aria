# Lesson: esbuild --target Does Not Polyfill Missing APIs, Only Downtranspiles Syntax

**Date:** 2026-02-28
**System:** community (evanw/esbuild)
**Tier:** lesson
**Category:** build
**Keywords:** esbuild, target, polyfill, Object.groupBy, node20, syntax vs semantics, bundler, TypeScript
**Source:** https://github.com/evanw/esbuild/issues/4268

---

## Observation (What Happened)

A project used `Object.groupBy` and set `--target=node20`. esbuild emitted the code unchanged (correct — `Object.groupBy` is valid syntax). But Node.js 20.0–20.3 does not include `Object.groupBy` (added in 20.4). The bundle ran fine in the bundler's output check, then crashed at runtime with `TypeError: Object.groupBy is not a function`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `Object.groupBy` is not syntax — it is a runtime API. esbuild's `--target` controls syntax downleveling (arrow functions → function, optional chaining → conditionals) but never injects polyfills.
**Why #2:** The developer assumed `--target=node20` means "make this code safe on node20" — it means "use node20 syntax features."
**Why #3:** esbuild documentation distinguishes syntax from API availability, but this is easy to miss.
**Why #4:** No runtime API compatibility test existed for the target platform.
**Why #5:** The mental model of "target = safe to run on" is reinforced by transpilers like Babel which DO polyfill, creating a false expectation for esbuild users.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Treat `--target` as syntax-only; add explicit polyfill (core-js, es-shims) for runtime APIs | proposed | community | esbuild#4268 |
| 2 | Document the target platform's API baseline (MDN / Node.js compat table) separately from the syntax level | proposed | community | esbuild#4268 |
| 3 | Add a CI step that runs the built artifact on the oldest supported Node.js version | proposed | community | esbuild#4268 |

## Key Takeaway

esbuild `--target` transpiles syntax, not semantics — `Object.groupBy`, `Array.at()`, `Promise.any()` and other runtime APIs are invisible to the bundler and will crash on older targets without explicit polyfills.
