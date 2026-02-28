# Lesson: Adding Files Without Bumping Version Breaks Client Cache Invalidation

**Date:** 2026-02-28
**System:** community (wshobson/agents)
**Tier:** lesson
**Category:** configuration
**Keywords:** versioning, cache invalidation, plugin, marketplace, Claude Code, skill, semantic versioning, manifest, cache, path not found
**Source:** https://github.com/wshobson/agents/issues/143

---

## Observation (What Happened)

A commit added 20 new skills and 5 agents across multiple plugins in the `wshobson/agents` Claude Code marketplace, but did not increment version numbers in `marketplace.json`. Existing users who had installed the plugins before the commit had cached metadata at the old version. After `git pull`, the content changed but the version key did not, so the cache did not invalidate. At runtime Claude Code resolved paths from stale cached metadata and produced "Path not found" errors for every newly added skill file.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Commit added files but did not update version strings in `marketplace.json`.
**Why #2:** No automated check (CI lint, pre-commit hook, or PR template) verified that adding files to a plugin required a version bump.
**Why #3:** The caching system used the version string as the sole invalidation signal — content hashing was not used.
**Why #4:** The contributing guide did not document the version-bump requirement for file additions.
**Why #5:** Maintainers treated version bumps as optional semantic versioning conventions rather than required operational contracts for cache correctness.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add a CI check: if any file under a plugin directory is added/modified in a PR, verify the plugin's version in `marketplace.json` is also modified | proposed | community | https://github.com/wshobson/agents/issues/143 |
| 2 | Document the version-bump requirement in CONTRIBUTING.md and the PR template | proposed | community | issue |
| 3 | Consider content-hash-based cache keys as a supplement to version strings | proposed | community | issue |

## Key Takeaway

When clients use version strings as cache invalidation keys, any content change that skips a version bump creates a silent stale-cache bug for all existing users; the invariant "content change = version change" must be enforced by CI, not by convention.
