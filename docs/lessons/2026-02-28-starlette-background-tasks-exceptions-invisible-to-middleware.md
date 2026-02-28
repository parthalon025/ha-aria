# Lesson: Starlette BackgroundTask Exceptions Are Not Propagated to Middleware or Exception Handlers
**Date:** 2026-02-28
**System:** community (encode/starlette)
**Tier:** lesson
**Category:** error-handling
**Keywords:** starlette, BackgroundTasks, exception, middleware, exception handler, BaseHTTPMiddleware, silent failure, background task
**Source:** https://github.com/encode/starlette/issues/2658
---
## Observation (What Happened)
When a `BackgroundTasks` task raises an exception, the exception is not surfaced to `BaseHTTPMiddleware` or to registered exception handlers. The background task failure is invisible — the route returns its normal response, the middleware completes normally, and the exception is swallowed inside the task runner. Applications that rely on exception handlers to log or report errors will miss all background task failures.

## Analysis (Root Cause — 5 Whys)
`BackgroundTasks` runs after the response is sent, as a post-response hook. At that point, the middleware stack has already completed its `dispatch` cycle and the exception handler chain is no longer active. The task runner in `starlette/background.py` calls the tasks sequentially but does not wrap them in the app's exception handler scope. Any unhandled exception from a background task bubbles up to the ASGI server's task runner (uvicorn), which logs it to stderr but does not forward it to the application's error handling infrastructure.

## Corrective Actions
- Wrap every `BackgroundTasks` task function in an explicit `try/except Exception` that logs to the application logger: `logger.exception("background task failed: %s", task_name)`.
- For tasks where failure must trigger an alert (e.g., Telegram notification send), implement a fallback inside the task itself (write to `/tmp/aria-missed-alerts.jsonl` as ARIA already does for Telegram).
- Never rely on middleware exception handlers or Sentry integrations to catch background task exceptions — they run outside the middleware scope.
- For ARIA: audit all `background_tasks.add_task(...)` call sites — each registered function should have an internal `try/except` with `logger.exception()`.

## Key Takeaway
Starlette `BackgroundTasks` run outside the middleware exception handler scope — background task failures are invisible to error handlers and must be caught and logged inside each task function.
