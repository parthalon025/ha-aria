# Lesson: CI Tools Distinguish Between Unset and Empty-String Environment Variables — Conditional Matrix Values Silently Break

**Date:** 2026-02-28
**System:** community (pypa/cibuildwheel)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, matrix, env-var, empty-string, unset, CIBW_ARCHS, conditional, null-coalescing
**Source:** https://github.com/pypa/cibuildwheel/issues/2533

---

## Observation (What Happened)

A matrix workflow set `CIBW_ARCHS: ${{ matrix.archs }}` where some matrix entries had `archs: null` (unset). When `matrix.archs` is null, `${{ matrix.archs }}` evaluates to an empty string `""`, not to "unset". cibuildwheel's `CIBW_ARCHS` parser treated an empty string differently from an absent variable — the empty string caused unexpected behavior (treated as "no archs selected" rather than "use default archs").

## Analysis (Root Cause — 5 Whys)

**Why #1:** Developer expected a null matrix value to mean "omit this env var" — but `${{ matrix.archs }}` always produces a string (empty when null).
**Why #2:** GitHub Actions expression evaluation always produces a string — there is no way to conditionally emit an `env:` key only when a matrix value is non-null using simple expression syntax.
**Why #3:** cibuildwheel (and many other tools) checks `if CIBW_ARCHS in os.environ` (present/absent) rather than `if os.environ.get("CIBW_ARCHS")` (truthy) — an empty string is "present" but semantically unset.
**Why #4:** The YAML `env:` block includes the key with an empty-string value, which passes it to the process environment as a real (empty) variable.
**Why #5:** Developers model empty string and absent variable as equivalent, but POSIX `${VAR:+}` and tool parsers treat them differently.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use conditional `env:` with `if:` step conditions rather than relying on null matrix values to omit variables: `if: matrix.archs != ''` on the step | proposed | community | https://github.com/pypa/cibuildwheel/issues/2533 |
| 2 | Use `${{ matrix.archs || '' }}` explicitly and document the empty-string behavior rather than assuming null=absent | proposed | community | — |
| 3 | In tool CLIs that read env vars, treat empty string identically to unset for optional configuration variables | proposed | community | — |

## Key Takeaway

GitHub Actions expressions always emit a string (never "absent") — `${{ matrix.null_value }}` produces `""` not "unset", and tools that distinguish empty from absent will behave unexpectedly; use explicit `if:` conditions to guard env var emission.
