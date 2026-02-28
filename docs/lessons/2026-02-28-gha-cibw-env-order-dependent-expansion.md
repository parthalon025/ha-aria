# Lesson: CI Environment Variable Blocks Are Evaluated in Declaration Order — Forward References Silently Expand to Empty

**Date:** 2026-02-28
**System:** community (pypa/cibuildwheel)
**Tier:** lesson
**Category:** configuration
**Keywords:** github-actions, environment, env-block, variable, expansion, order, silent-empty, cibuildwheel, CIBW_ENVIRONMENT
**Source:** https://github.com/pypa/cibuildwheel/issues/2616

---

## Observation (What Happened)

A cibuildwheel `[tool.cibuildwheel.environment]` block declared `FOO = "Hello $BAR"` before `BAR = "World"`. At build time `FOO` evaluated to `"Hello "` — `$BAR` silently expanded to empty string because `BAR` had not yet been defined in the sequential evaluation pass. No error or warning was emitted.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Developer expected the environment block to behave like a declarative config (all variables simultaneously available to each other).
**Why #2:** cibuildwheel (and GitHub Actions `env:` blocks in general) evaluates variable assignments sequentially — each variable can only reference variables defined earlier in the block.
**Why #3:** `$BAR` at expansion time resolves to the shell's current value, which is empty because `BAR` hasn't been set yet in the evaluation pass.
**Why #4:** No error is raised — shell variable expansion of an unset variable produces an empty string by default (equivalent to `${BAR:-}`).
**Why #5:** Developers carry the mental model of Python dict comprehension (all values available simultaneously) rather than sequential bash variable assignment.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Declare variables in dependency order — define `BAR` before any variable that references `$BAR` | proposed | community | https://github.com/pypa/cibuildwheel/issues/2616 |
| 2 | Treat env blocks as sequential bash assignments, not declarative config — no forward references | proposed | community | — |
| 3 | Add `set -u` (nounset) to shell steps to catch empty-variable expansion at runtime rather than silently propagating wrong values | proposed | community | — |

## Key Takeaway

Environment variable blocks in CI configs are evaluated sequentially — a variable referencing another variable defined later in the block silently expands to empty with no error; always define dependencies before use.
