# Lesson: pnpm/npm Can Resolve a Peer Dependency to an Incompatible Major Version via Transitive Chains

**Date:** 2026-02-28
**System:** community (prisma/prisma)
**Tier:** lesson
**Category:** build
**Keywords:** pnpm, npm, semver, peer dependency, signal-exit, transitive, resolution, proper-lockfile, Node.js v24
**Source:** https://github.com/prisma/prisma/issues/29152

---

## Observation (What Happened)

`prisma generate` failed on Node.js v24 with pnpm with `TypeError: onExit is not a function`. Root cause: `proper-lockfile@4.1.2` uses `require('signal-exit')` expecting signal-exit@3 (which exports a function directly). pnpm resolved `signal-exit` to version 4, which changed its export to `{ onExit }` — a named export. The code broke silently at runtime because `require()` returned an object, not a function.

## Analysis (Root Cause — 5 Whys)

**Why #1:** signal-exit v4 changed its export interface from a default function to a named export — a breaking API change despite a new major version.
**Why #2:** `proper-lockfile` did not pin `signal-exit` to `^3` — it likely used a loose or implicit version range.
**Why #3:** pnpm's dependency resolution algorithm hoisted signal-exit@4 (required by another package in the tree) over signal-exit@3 for `proper-lockfile`.
**Why #4:** The failure only occurred on Node.js v24 because Node.js v24 changed how pnpm resolves conflicting peer deps.
**Why #5:** No integration test for `prisma generate` ran on the Node.js v24 + pnpm matrix, so the breakage was not caught before release.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pin transitive deps with breaking API changes using `overrides` / `resolutions` in the consumer's package.json | proposed | community | prisma#29152 |
| 2 | CI matrix must include the latest Node.js LTS + current release + latest pnpm | proposed | community | prisma#29152 |
| 3 | When using `require()` for a dependency, assert the resolved export shape at startup (e.g., `if (typeof onExit !== 'function') throw`) | proposed | community | prisma#29152 |

## Key Takeaway

Package managers can silently hoist an incompatible major version of a transitive dependency — assert the shape of `require()`d exports at import time, and run CI across multiple package managers and Node.js versions.
