# Lesson: Auto-Pause on Timeout Snapshots Filesystem Before Buffers Are Flushed

**Date:** 2026-02-28
**System:** community (e2b-dev/E2B)
**Tier:** lesson
**Category:** integration
**Keywords:** sandbox, timeout, auto-pause, filesystem, snapshot, fsync, buffer flush, race condition, data loss, agent environment, E2B
**Source:** https://github.com/e2b-dev/E2B/issues/1104

---

## Observation (What Happened)

When an E2B sandbox auto-paused due to timeout, data written before the pause was sometimes lost after resume. Manually triggered pauses preserved all data correctly. The hypothesis confirmed by maintainers: the timeout-triggered auto-pause snapshotted the filesystem before kernel buffers were fully flushed to disk, capturing an inconsistent state. Manual pauses included an explicit flush step; the timeout path did not.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The auto-pause timeout handler triggered a filesystem snapshot without first calling `sync` or `fsync` on open file handles.
**Why #2:** The manual pause API included a flush step; the timeout code path was a separate implementation that missed it.
**Why #3:** Filesystem buffer semantics (writes are async, data may be in page cache, not on disk) were not accounted for in the timeout handler.
**Why #4:** Testing focused on the manual pause path; no test verified that a timeout-triggered pause preserved identical state to a manual pause.
**Why #5:** The two code paths (manual and timeout-triggered pause) shared no common snapshot utility that enforced the flush invariant.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Before taking a filesystem snapshot (in all code paths), call `sync` or `fsync` on all open file handles to flush kernel buffers | proposed | community | https://github.com/e2b-dev/E2B/issues/1104 |
| 2 | Unify manual and auto-pause snapshot paths through a single function that enforces the flush-before-snapshot invariant | proposed | community | issue |
| 3 | Add a test: write a file, trigger a timeout-induced pause, resume, verify file contents are intact | proposed | community | issue |

## Key Takeaway

Any system that snapshots filesystem state (for pause/resume, backup, or checkpoint) must explicitly flush kernel page cache buffers before taking the snapshot — the assumption that in-progress writes are on disk when the snapshot runs is wrong unless explicitly enforced with `sync`/`fsync`.
