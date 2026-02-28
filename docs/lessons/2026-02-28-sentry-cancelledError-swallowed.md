# Lesson: Broad Exception Capture Context Managers Swallow asyncio.CancelledError

**Date:** 2026-02-28
**System:** community (getsentry/sentry-python)
**Tier:** lesson
**Category:** async
**Keywords:** asyncio, CancelledError, exception swallowing, context manager, task cancellation, BaseException
**Source:** https://github.com/getsentry/sentry-python/issues/4853

---

## Observation (What Happened)

A `capture_internal_exceptions()` context manager in the Sentry Python SDK suppressed all exceptions including `asyncio.CancelledError`, making it impossible to cancel asyncio tasks that were executing inside the guard. Tasks using OpenAI streaming completions wrapped by this guard could never be cancelled — `await task` would hang indefinitely even after `task.cancel()` was called.

## Analysis (Root Cause — 5 Whys)

The `__exit__` method returned `True` unconditionally, telling Python to suppress all exceptions. `asyncio.CancelledError` is a subclass of `BaseException`, not `Exception`, precisely because it is a control-flow signal that must propagate. Any broad `except Exception` or suppressing `__exit__` that doesn't explicitly re-raise `BaseException` subclasses (`CancelledError`, `KeyboardInterrupt`, `SystemExit`) breaks the asyncio cooperative cancellation contract. The fix is a `isinstance(ty, asyncio.CancelledError): return False` check before the generic suppress.

## Corrective Actions

- In any `__exit__` that returns `True` to suppress exceptions, add an explicit carve-out: `if ty is not None and issubclass(ty, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)): return False`.
- Apply the same rule to `except Exception` blocks that swallow and log: replace with `except Exception` only, never `except BaseException`.
- Test cancellation explicitly: `task.cancel(); await asyncio.sleep(0); assert task.cancelled()`.

## Key Takeaway

Never suppress `asyncio.CancelledError` (or any `BaseException` subclass) inside exception guards — these are cooperative control-flow signals, not errors.
