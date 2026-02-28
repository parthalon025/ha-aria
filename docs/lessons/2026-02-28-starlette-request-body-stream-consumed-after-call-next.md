# Lesson: Starlette BaseHTTPMiddleware Request Body Cannot Be Read After call_next Unless Pre-Cached
**Date:** 2026-02-28
**System:** community (encode/starlette)
**Tier:** lesson
**Category:** integration
**Keywords:** starlette, BaseHTTPMiddleware, request body, call_next, stream consumed, RuntimeError, caching, middleware
**Source:** https://github.com/encode/starlette/issues/2555
---
## Observation (What Happened)
In Starlette's `BaseHTTPMiddleware`, calling `await request.body()` *after* `await call_next(request)` raises `RuntimeError: Stream consumed`. The request body stream is a one-time-read ASGI stream — once the route handler reads it during `call_next`, the stream is exhausted and the body data is gone from the middleware's perspective.

## Analysis (Root Cause — 5 Whys)
ASGI request bodies are streaming by design — the underlying transport delivers body chunks once. `BaseHTTPMiddleware` wraps the route in a `call_next` coroutine that passes the stream through to the route handler. The route handler (or its framework layer) reads the stream to completion. There is no automatic caching or rewinding — once consumed, `request._body` is populated only if the framework explicitly caches it, which only happens for the route handler's copy of the request object, not for the middleware's `request` object. The middleware's `request` is a `_CachedRequest` wrapper, but the `_body` attribute is set from the route-side read, not exposed back to the middleware.

## Corrective Actions
- To read the request body in middleware both before and after `call_next`, read and cache it *before* calling `call_next`: `body = await request.body()`. This forces the body to be cached in `request._body`; subsequent reads hit the cache.
- Never attempt to read `request.body()` post-`call_next` for a different reason than the route read — it is architecturally impossible in streaming ASGI.
- For audit/logging middleware in ARIA's hub that captures request bodies, always read before `call_next` and log, then pass through; never attempt to re-read after.

## Key Takeaway
Starlette's `BaseHTTPMiddleware` request body stream is consumed by the route handler during `call_next` — read and cache the body before `call_next` if the middleware also needs it.
