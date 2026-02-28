# Lesson: Explicit Tool Version Ignored at Execution Time Due to Missing Propagation

**Date:** 2026-02-28
**System:** community (ComposioHQ/composio)
**Tier:** lesson
**Category:** integration
**Keywords:** tool version, versioning, SDK, execute, getRawComposioToolBySlug, toolkit version, propagation, API call, composio
**Source:** https://github.com/ComposioHQ/composio/issues/2471

---

## Observation (What Happened)

In Composio's TypeScript SDK, when a user called `tools.execute('TOOL_SLUG', { version: '20260122_00' })` with an explicit version, the execution path internally called `getRawComposioToolBySlug(slug)` without passing the specified version. The tool lookup used the globally configured `toolkitVersions` (which pinned to a different version) and failed with `ComposioToolNotFoundError: Unable to retrieve tool with slug TOOL_SLUG`. The explicit per-call version was silently discarded.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `Tools.execute()` accepted a `version` parameter but did not pass it to `getRawComposioToolBySlug()`.
**Why #2:** `getRawComposioToolBySlug()` used the globally configured `this.toolkitVersions` for lookup, with no way to override it per call.
**Why #3:** The `version` field was added to the execute API but the internal lookup chain was not updated to thread the version through.
**Why #4:** No test covered the case of explicitly passing a version that differs from the globally configured toolkit version.
**Why #5:** The feature was partially implemented — the public API accepted the field but the internals ignored it.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pass the per-call `version` to `getRawComposioToolBySlug()` and use it to override `toolkitVersions` for that specific lookup | proposed | community | https://github.com/ComposioHQ/composio/issues/2471 |
| 2 | Add a test: execute with an explicit version different from the global toolkit version and assert the correct version is used | proposed | community | issue |

## Key Takeaway

When an API accepts an explicit override parameter, that parameter must be threaded through every internal layer that performs the relevant lookup — accepting a parameter in the public API but silently discarding it internally is a silent partial implementation that is harder to debug than a feature being absent entirely.
