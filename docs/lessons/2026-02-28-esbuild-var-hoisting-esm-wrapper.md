# Lesson: esbuild's __esm Lazy Wrappers Break var Hoisting for Inline if Declarations

**Date:** 2026-02-28
**System:** community (evanw/esbuild)
**Tier:** lesson
**Category:** build
**Keywords:** esbuild, var, hoisting, __esm, CJS, ESM, bundler, miscompilation, if statement, ReferenceError
**Source:** https://github.com/evanw/esbuild/issues/4348

---

## Observation (What Happened)

When esbuild creates `__esm` lazy initialization wrappers (triggered by a CommonJS module requiring an ESM module), it fails to hoist `var` declarations from inline `if` statement blocks (`if (condition) var b = getValue()`) to module scope. The result is a `ReferenceError` at runtime in code that was valid JavaScript before bundling. This affected real projects using jszip and @jspm/core.

## Analysis (Root Cause — 5 Whys)

**Why #1:** JavaScript semantics require `var` to be hoisted to function/module scope regardless of the block it appears in.
**Why #2:** esbuild's `__esm` wrapper wraps the module body in a function — but the hoisting pass did not account for `var` inside inline `if` statements (not block statements with braces).
**Why #3:** The pattern `if (condition) var b = ...` is unusual and was not in esbuild's hoisting test matrix.
**Why #4:** The bug only manifests when the CJS→ESM interop path is triggered — pure-ESM or pure-CJS bundles were unaffected.
**Why #5:** The miscompiled output is syntactically valid, so no build-time error occurred — only a runtime ReferenceError.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Avoid `if (condition) var x = ...` style declarations — always use braces and declare vars at function top | proposed | community | esbuild#4348 |
| 2 | Integration-test bundled output by running it, not just inspecting it | proposed | community | esbuild#4348 |
| 3 | Flag use of `var` inside conditionals with eslint `no-inner-declarations` rule | proposed | community | esbuild#4348 |

## Key Takeaway

Bundler `__esm` wrappers may silently break `var` hoisting from unusual patterns like `if (condition) var x = ...` — use explicit `var x; if (condition) x = ...` declarations and test bundled output by execution, not source inspection.
