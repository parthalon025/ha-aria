# Lesson: Mutually Recursive Helper Functions Without Base Case Cause RecursionError on Agent Init

**Date:** 2026-02-28
**System:** community (Fosowl/agenticSeek)
**Tier:** lesson
**Category:** error-handling
**Keywords:** recursion, RecursionError, infinite recursion, base case, init, startup, agent, tool, create_work_dir, safe_get_work_dir_path
**Source:** https://github.com/Fosowl/agenticSeek/issues/400

---

## Observation (What Happened)

On startup, the agenticSeek agent's `tools.py` raised a `RecursionError` before any task was executed. `create_work_dir()` called `safe_get_work_dir_path()` to get the directory path, and `safe_get_work_dir_path()` called `create_work_dir()` when the path was not yet set — creating a mutually recursive cycle with no base case. The stack trace showed the two functions alternating until maximum recursion depth was exceeded.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `create_work_dir()` delegated path resolution to `safe_get_work_dir_path()`.
**Why #2:** `safe_get_work_dir_path()` called `create_work_dir()` as a fallback when the path was not yet initialized.
**Why #3:** No base case existed: the path was never set to a valid value before either function was called.
**Why #4:** The initialization order was incorrect — these methods were called before the configuration that provides the work directory path was loaded.
**Why #5:** No test or startup health check validated that required configuration was available before tool initialization.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Break the mutual recursion: `safe_get_work_dir_path()` should return a default/constant path if not configured, not call `create_work_dir()` | proposed | community | https://github.com/Fosowl/agenticSeek/issues/400 |
| 2 | Initialize the work directory path from config before calling any path-dependent method | proposed | community | issue |
| 3 | Add a startup assertion that required config values (work_dir) are non-None before calling tool constructors | proposed | community | issue |

## Key Takeaway

Any pair of functions where A calls B as a fallback and B calls A as a fallback will recurse infinitely unless one of them can produce a result without delegating — design fallback chains as one-directional with a terminal base case, and validate required config before init-time tool construction.
