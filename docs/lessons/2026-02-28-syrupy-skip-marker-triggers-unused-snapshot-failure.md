# Lesson: pytest.mark.skip on Snapshot Tests Causes Syrupy to Report Unused Snapshots as Failures

**Date:** 2026-02-28
**System:** community (syrupy-project/syrupy)
**Tier:** lesson
**Category:** testing
**Keywords:** syrupy, snapshot, skip, skipif, unused snapshot, exit code, false failure, snapshot testing
**Source:** https://github.com/syrupy-project/syrupy/issues/842

---

## Observation (What Happened)
Tests marked with `@pytest.mark.skip` or `@pytest.mark.skipif` that had corresponding snapshot files caused syrupy to report those snapshots as "unused" and exit with a non-zero error code. The test run was reported as failed even though all *executed* tests passed. This broke CI pipelines for platform-conditional tests (e.g., tests skipped on Python 3.12+).

## Analysis (Root Cause — 5 Whys)
**Why #1:** Syrupy tracks which snapshot names were asserted during the test run. Skipped tests never execute their assert statements.
**Why #2:** Syrupy's unused-snapshot detection does not differentiate between "skipped by marker" and "genuinely orphaned snapshot from a deleted test."
**Why #3:** At session end, syrupy iterates all snapshot files and flags any snapshot key that was not touched during the run.
**Why #4:** Syrupy exits with a failure exit code on unused snapshots by default, making skipped-test snapshots indistinguishable from stale orphan snapshots.
**Why #5:** No mechanism existed (before the fix) to mark snapshots as "conditionally used" when their owning test is marked for conditional skipping.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Do not use `pytest.mark.skip` or `pytest.mark.skipif` on snapshot tests; instead use `pytest.importorskip` or conditional skip at collection time to prevent snapshot from being registered | proposed | community | issue #842 |
| 2 | When conditional platform tests with snapshots are unavoidable, use `--snapshot-warn-unused` instead of the default failure mode, or use `--snapshot-default-extension` to segregate platform-specific snapshots into separate files | proposed | community | issue #842 |
| 3 | Pin syrupy to a version ≥4.7.0 which fixed this behavior (skip-marked snapshot tests are no longer counted as unused) | proposed | community | issue #842 |

## Key Takeaway
Syrupy treats snapshot files from `pytest.mark.skip`-decorated tests as orphaned/unused and fails the suite — use `pytest.importorskip` or collection-time conditionals instead of run-time skip markers on snapshot tests to avoid false failures.
