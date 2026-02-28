# Lesson: FastAPI Resets contextvars Per Request — ContextVar Values Do Not Leak Between Requests
**Date:** 2026-02-28
**System:** community (tiangolo/fastapi)
**Tier:** lesson
**Category:** async
**Keywords:** FastAPI, contextvars, ContextVar, per-request isolation, request context, thread reuse, sync routes, asyncio
**Source:** https://github.com/tiangolo/fastapi/issues/13119
---
## Observation (What Happened)
A FastAPI sync route sets a `ContextVar` value during one request, but on the next request to the same endpoint, the `ContextVar` returns its default value — not the value set in the previous request. Developers expecting `ContextVar` values to persist across requests (as a form of request-scoped state) find the value is cleared between requests.

## Analysis (Root Cause — 5 Whys)
This is correct and intentional behavior, not a bug. FastAPI (via Starlette and uvicorn) uses asyncio's context copying semantics — each new request creates a copy of the current context at dispatch time. For sync routes, the thread pool executor further scopes the context. `ContextVar.set()` in request N only modifies the copy of the context for that request; the base context used for future requests remains unchanged. This provides per-request isolation and prevents state leakage between concurrent requests.

## Corrective Actions
- Use `ContextVar` only for within-request propagation (e.g., passing request-scoped data from middleware to route handlers within a single request). Never use it as a persistence mechanism across requests.
- For state that must persist across requests, use module-level variables, a shared dict, or a proper cache (e.g., `hub.set_cache()`).
- When debugging "missing" `ContextVar` values, confirm whether the access is in the same request context (same coroutine chain) or a different request — different requests always see the default or their own set value.
- Document any `ContextVar` usage at the declaration site: `# scope: per-request only — reset on each new request dispatch`.

## Key Takeaway
FastAPI creates a fresh copy of the asyncio context for each request — `ContextVar` values set during one request are never visible to other requests; use `ContextVar` only for within-request propagation.
