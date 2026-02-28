# Lesson: Non-Exported const enum Should Inline to Nothing — Bundlers May Emit Unnecessary Mapping Objects

**Date:** 2026-02-28
**System:** community (evanw/esbuild)
**Tier:** lesson
**Category:** build
**Keywords:** TypeScript, const enum, bundler, esbuild, inlining, dead code, non-exported, tsc, enum mapping
**Source:** https://github.com/evanw/esbuild/issues/4390

---

## Observation (What Happened)

A non-exported TypeScript `const enum` used only within the same module should be fully inlined by the compiler — all references replaced with their numeric literal values, no runtime object emitted. esbuild instead emitted a full enum mapping object (`var o = (n => (n[n.A=0]="A", n))(o||{})`) even though the enum was non-exported and never used as an object at runtime.

## Analysis (Root Cause — 5 Whys)

**Why #1:** esbuild does not perform the same constant-folding for `const enum` as TypeScript's `tsc` — it emits a runtime object as a safety fallback.
**Why #2:** `tsc` inlines `const enum` values at usage sites and removes the declaration entirely. esbuild cannot do this for all cases without type information.
**Why #3:** esbuild operates without TypeScript's full type checker — it parses TypeScript syntax but doesn't resolve types the same way.
**Why #4:** The non-exported `const enum` pattern is often used to avoid generating any runtime code — a contract that esbuild breaks silently.
**Why #5:** No build output size test or snapshot test exists to catch regression in `const enum` inlining behavior.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When targeting esbuild, prefer `as const` objects or `enum` (not `const enum`) — behavior is more predictable | proposed | community | esbuild#4390 |
| 2 | Use `tsc` to verify `const enum` inlining in performance-critical paths; don't rely on esbuild | proposed | community | esbuild#4390 |
| 3 | Add bundle size regression tests to CI for any module using `const enum` | proposed | community | esbuild#4390 |

## Key Takeaway

esbuild does not inline `const enum` the same way `tsc` does — non-exported `const enum` may still emit a runtime mapping object, adding dead code to the bundle; use `as const` objects for portable zero-cost constants.
