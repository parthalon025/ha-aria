# Lesson: Starlette BaseHTTPMiddleware Silently Swallows Exceptions From Mounted Sub-Applications
**Date:** 2026-02-28
**System:** community (encode/starlette)
**Tier:** lesson
**Category:** error-handling
**Keywords:** starlette, BaseHTTPMiddleware, exception, mount, sub-application, silent failure, exception handler, middleware stack
**Source:** https://github.com/encode/starlette/issues/2625
---
## Observation (What Happened)
When `BaseHTTPMiddleware` is applied to a Starlette/FastAPI app that uses `app.mount("/path", sub_app)`, exceptions raised in the sub-application routes are silently swallowed — no traceback appears in the console and no error response is produced for the client. Removing the middleware restores the exception behavior. The issue is a recurring regression in Starlette and affects any `Exception` subclass (but not `BaseException` subclasses, which bypass the swallowing).

## Analysis (Root Cause — 5 Whys)
`BaseHTTPMiddleware` uses anyio task groups internally to run the route handler concurrently with the send stream. Exceptions raised in the sub-application are caught inside the task group's inner scope and passed through an internal `app_exc` nonlocal variable. The exception propagation path from mounted apps through `BaseHTTPMiddleware`'s `call_next` coroutine has historically had edge cases where exceptions are caught, stored, and not re-raised correctly — particularly for non-`ServerErrorMiddleware` routes in mounted sub-apps. The fix requires using Pure ASGI middleware (`app = Middleware(...)`) instead of `BaseHTTPMiddleware` for exception transparency.

## Corrective Actions
- Replace `BaseHTTPMiddleware` with Pure ASGI middleware for any middleware that must observe exceptions from all mounted sub-applications. Pure ASGI middleware does not intercept the exception propagation chain.
- If `BaseHTTPMiddleware` must be kept, add explicit exception logging inside the `dispatch` method: wrap `await call_next(request)` in `try/except Exception as exc: logger.exception(exc); raise`.
- Test middleware exception visibility explicitly: add a test that mounts a sub-app, registers the middleware, raises an exception in the sub-app, and asserts that the exception handler fires and returns a 500.
- For ARIA's hub: any logging/auth middleware should be Pure ASGI style to ensure exceptions from all API routes surface correctly.

## Key Takeaway
Starlette `BaseHTTPMiddleware` can silently absorb exceptions from mounted sub-applications — use Pure ASGI middleware or add explicit exception re-raise inside `dispatch` to guarantee visibility.
