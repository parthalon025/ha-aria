# Lesson: xdist Workers Share tmp_path basetemp — Parallel Tests Can Delete Each Other's Directories

**Date:** 2026-02-28
**System:** community (pytest-dev/pytest-xdist)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, xdist, tmp_path, basetemp, parallel, worker, FileNotFoundError, shared temp, race condition
**Source:** https://github.com/pytest-dev/pytest-xdist/issues/1285

---

## Observation (What Happened)
A test using `tmp_path` under xdist worksteal mode raised `FileNotFoundError` intermittently during setup. The root directory `/tmp/pytest-of-<user>/<num>/` was deleted mid-run. The bug was flaky — it only occurred when multiple workers ran simultaneously and the pytest temp cleanup logic ran on one worker while another was still using it.

## Analysis (Root Cause — 5 Whys)
**Why #1:** pytest creates temp directories under a shared `basetemp` (e.g., `/tmp/pytest-of-<user>/pytest-<run>/`). All xdist workers on the same machine share this basetemp path by default.
**Why #2:** pytest's cleanup logic retains only the last 3 `basetemp` directories and deletes older ones. Under xdist, different workers may have different views of "old vs current."
**Why #3:** When one worker triggers cleanup of the shared basetemp root during active use by another worker, the second worker's `tmp_path` subdirectory is deleted.
**Why #4:** The pytest documentation recommends setting a distinct `--basetemp` per parallel execution, but pytest-xdist does not enforce or document this.
**Why #5:** This only manifests as a flaky failure (not consistent) because cleanup timing is non-deterministic.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When using xdist, always set `--basetemp` to a unique path per run: `pytest -n 4 --basetemp=/tmp/pytest-$(date +%s)` or use a per-job workspace in CI | proposed | community | issue #1285 |
| 2 | In CI, configure `basetemp` in `pyproject.toml` or `pytest.ini` via `tmp_path_retention_count = 1` and `tmp_path_retention_policy = "failed"` to avoid premature cleanup | proposed | community | issue #1285 |
| 3 | Prefer `tmp_path` over `tmpdir` (deprecated) — `tmp_path` uses per-test subdirectories that are less likely to collide, but basetemp collision is still possible without the above fixes | proposed | community | issue #1285 |

## Key Takeaway
xdist workers share a basetemp root by default, and pytest's basetemp cleanup is not xdist-aware — set `--basetemp` to a unique directory per parallel run in CI to prevent workers from deleting each other's `tmp_path` directories mid-test.
