# Lesson: esbuild Renames let Bindings Across Scopes, Breaking Shadow Variable Semantics

**Date:** 2026-02-28
**System:** community (evanw/esbuild)
**Tier:** lesson
**Category:** build
**Keywords:** esbuild, let, scope, shadowing, miscompilation, rename, bundler, JSX, React
**Source:** https://github.com/evanw/esbuild/issues/4308

---

## Observation (What Happened)

Code using two `let cls` declarations at different scopes (outer scope `let cls = 'a'`, inner scope `let cls = 'b'`) was miscompiled by esbuild 0.25.11. esbuild renamed the inner `cls` to `cls2` to avoid collision — but incorrectly applied `cls2` to a reference in an inner-inner `if` block that should have resolved to the outer `cls`. The wrong variable was read at runtime, producing incorrect class names in JSX output.

## Analysis (Root Cause — 5 Whys)

**Why #1:** esbuild's variable renaming pass resolves same-name `let` bindings in nested scopes by appending a numeric suffix — but the scope resolution had an off-by-one in nested `if` blocks.
**Why #2:** The test case required three levels of nesting with `let` shadows — a rare combination not covered by existing scope tests.
**Why #3:** The miscompilation produces valid JavaScript syntax — no build error, only incorrect runtime behavior.
**Why #4:** JSX output makes scope issues harder to notice visually because the class attribute name looks correct in the source.
**Why #5:** Variable shadowing in nested `let` is a valid and common pattern; the bundler must handle it without altering semantics.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Rename shadowed variables at the start of the outer scope, not inside inner blocks | proposed | community | esbuild#4308 |
| 2 | Add snapshot tests for bundled output of known shadow-variable patterns | proposed | community | esbuild#4308 |
| 3 | When debugging unexpected runtime behavior, diff the bundled output against the source for variable name changes | proposed | community | esbuild#4308 |

## Key Takeaway

esbuild's variable renaming for `let` shadows across nested scopes has had miscompilation bugs — always test bundled output with real data, not just source inspection, when using nested `let` shadowing patterns.
