# Lesson: CLI Tools Using Bundled CJS Internals Break When the Config File Is Pure ESM

**Date:** 2026-02-28
**System:** community (drizzle-team/drizzle-orm)
**Tier:** lesson
**Category:** build
**Keywords:** require, ESM, CJS, drizzle-kit, config file, module system, package.json type, bundler
**Source:** https://github.com/drizzle-team/drizzle-orm/issues/5170

---

## Observation (What Happened)

`drizzle-kit push` failed with `require is not defined in ES module scope` when the user's project had `"type": "module"` in `package.json` or a `.js` config file in an ESM project. The tool's internal code used `require()` to load the user's config file, which is incompatible with pure-ESM modules.

## Analysis (Root Cause — 5 Whys)

**Why #1:** When `package.json` has `"type": "module"`, all `.js` files are treated as ESM. `require()` does not exist in ESM scope.
**Why #2:** The CLI tool (drizzle-kit) dynamically loaded user config with `require()` — a CJS-only mechanism.
**Why #3:** The tool was written as CJS but attempted to load user-provided config that followed the project's own module conventions.
**Why #4:** No runtime guard distinguished ESM vs CJS config file format before loading.
**Why #5:** The `"type": "module"` field in package.json is a project-level ESM declaration that affects every `.js` file — tools that `require()` user configs must use dynamic `import()` instead.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `import()` (dynamic import) for loading user config files, not `require()` | proposed | community | drizzle-orm#5170, #5158 |
| 2 | Config files used by CLI tools should be `.cjs` or `.mjs` with explicit extension to avoid ambiguity | proposed | community | drizzle-orm#5170 |
| 3 | When publishing a CLI that loads user config, test with both `"type": "module"` and `"type": "commonjs"` project setups | proposed | community | drizzle-orm#5158 |

## Key Takeaway

Any CLI tool that dynamically loads user-provided config files must use dynamic `import()` rather than `require()` — `package.json "type": "module"` in the user's project makes `require()` unavailable and will crash the tool.
