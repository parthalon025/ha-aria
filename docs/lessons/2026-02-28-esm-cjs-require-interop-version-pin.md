# Lesson: Semver Range in package.json Can Resolve a CJS-Incompatible ESM-Only Version

**Date:** 2026-02-28
**System:** community (privatenumber/tsx)
**Tier:** lesson
**Category:** build
**Keywords:** ESM, CJS, require, ERR_REQUIRE_ESM, semver, version pin, dotenv, npm, package.json
**Source:** https://github.com/privatenumber/tsx/issues/725

---

## Observation (What Happened)

A project pinned `dotenv` to `^4.19.0` (caret range). npm resolved it to `4.20.x`, which shipped as a pure ESM package. Code that previously used `require('dotenv/config')` broke with `ERR_REQUIRE_ESM` — even though nothing in the project changed. Pinning to exact `4.19.0` fixed it immediately.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `^` semver range allows minor/patch updates within a major version.
**Why #2:** The dependency (dotenv) dropped CJS support in a patch release without a major version bump.
**Why #3:** The project consumed the package via `require()` — compatible with CJS but not ESM.
**Why #4:** No CI lockfile check or audit caught the resolved version changing.
**Why #5:** The ecosystem norm that "semver patch bumps are backward-compatible" does not hold when the module system changes from CJS to pure ESM.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Commit `package-lock.json` / `yarn.lock` and fail CI if it drifts | proposed | community | tsx#725 |
| 2 | When a dep crosses ESM boundary mid-range, treat as breaking change and pin to last CJS version | proposed | community | tsx#724 |
| 3 | CI matrix should test with `npm ci` (uses lockfile) AND `npm install` (fresh resolution) | proposed | community | tsx#725 |

## Key Takeaway

A caret semver range is not a safety net: a dependency can drop CJS support in a minor/patch release, silently breaking `require()` consumers — always commit a lockfile and verify the resolved version on every dep upgrade.
