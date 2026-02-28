# Lesson: Non-Atomic File Prune via write_text Destroys Data on SIGKILL

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** data-model
**Keywords:** atomic write, prune, JSONL, write_text, truncate, SIGKILL, OOM, data loss, tempfile, os.replace
**Files:** aria/hub/core.py:538

---

## Observation (What Happened)

The snapshot log prune at `hub/core.py:538` uses `log_path.write_text(pruned_content)`. `write_text()` truncates the file before writing the new content. A SIGKILL, OOM kill, or power loss between truncation and write completion destroys all retained log entries — the file is left empty with no recovery path.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `pathlib.write_text()` is not atomic — it opens the file for writing (truncating it) and writes in a separate step.

**Why #2:** The developer used the convenient `write_text()` API without considering the crash window between truncate and write.

**Why #3:** The snapshot log is append-only JSONL — this prune is the only rewrite operation, making the atomicity requirement easy to overlook.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace `log_path.write_text(...)` with atomic write: write to `tempfile.NamedTemporaryFile` in same directory, then `os.replace(tmp_path, log_path)` | proposed | Justin | issue #313 |
| 2 | Apply the same pattern to any other file rewrite operations (`write_text` on existing files) in the codebase | proposed | Justin | — |

## Key Takeaway

Any file rewrite that must survive process crashes must use atomic write: write to a temp file in the same directory, then `os.replace()` — `write_text()` truncates before writing, creating a data-loss window.
