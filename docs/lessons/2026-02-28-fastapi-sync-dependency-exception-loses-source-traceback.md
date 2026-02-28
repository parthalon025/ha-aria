# Lesson: FastAPI Sync Generator Dependencies Swallow Source Traceback When Re-Raising Exceptions
**Date:** 2026-02-28
**System:** community (tiangolo/fastapi)
**Tier:** lesson
**Category:** error-handling
**Keywords:** FastAPI, sync dependency, generator, yield, traceback, contextmanager_in_threadpool, exception, re-raise, debugging
**Source:** https://github.com/tiangolo/fastapi/issues/13067
---
## Observation (What Happened)
When a synchronous `yield` dependency raises an exception, the traceback shown in the console contains only frames from FastAPI's `contextmanager_in_threadpool` wrapper (in `fastapi/concurrency.py`) and Python's `contextlib` internals. The original source file and line number where the exception was raised are absent. Debugging dependency failures requires either wrapping the dependency in a `try/except` with explicit `logging.exception()` or adding print statements to find the actual error location.

## Analysis (Root Cause — 5 Whys)
Synchronous `yield` dependencies are run in a thread pool via `contextmanager_in_threadpool`. When the dependency raises, the exception is transported across the thread boundary via Python's exception chaining mechanism. The `raise e` in FastAPI's concurrency wrapper re-raises the exception, but because the exception was caught and re-raised in a different frame, the original traceback chain is truncated — the `raise e` syntax replaces the traceback rather than re-raising with `raise` (which would preserve the chain). This is a known footgun in Python's exception transport across thread boundaries.

## Corrective Actions
- Wrap critical sync `yield` dependencies in explicit `try/except` with `logging.exception("dependency failed")` before re-raising, so the source is always logged regardless of traceback truncation.
- Prefer `async def` generators over sync `def` generators for FastAPI dependencies in ARIA to avoid the thread-pool transport entirely — this also eliminates the traceback truncation.
- When a sync dependency mysteriously fails with a useless traceback, add `import traceback; traceback.print_exc()` inside the dependency to capture the full frame before FastAPI intercepts it.

## Key Takeaway
Synchronous FastAPI `yield` dependencies run in a thread pool — exceptions lose their source traceback when transported across the thread boundary; prefer `async def` dependencies or add explicit `logging.exception()` inside sync ones.
