# Lesson: Sentry SDK Asyncio Integration Adds Significant Per-Request Overhead via Context Patching

**Date:** 2026-02-28
**System:** community (getsentry/sentry-python)
**Tier:** lesson
**Category:** performance
**Keywords:** sentry, asyncio, performance overhead, context patching, FastAPI, uvicorn, tracing, instrumentation cost
**Source:** https://github.com/getsentry/sentry-python/issues/4660

---

## Observation (What Happened)

Adding `sentry_sdk.init()` alone (no error capture, no transaction capture, no event sending) to FastAPI + Uvicorn services reduced throughput by 2-4x in benchmarks — from ~20k req/s to ~5k req/s. The `AsyncioIntegration` patches `asyncio` context management internals, adding a copy-context call on every task creation and every coroutine boundary in the request lifecycle.

## Analysis (Root Cause — 5 Whys)

The Sentry SDK's asyncio integration intercepts `asyncio.Task` creation to propagate isolation scopes via `contextvars`. This interception adds `contextvars.copy_context()` calls at task-creation boundaries. In high-throughput async services where many small coroutines are created per request, this overhead compounds. The cost scales with the depth of the coroutine chain, not just the number of requests. `copy_context()` is O(n) in the number of context variables set, and Sentry sets several.

## Corrective Actions

- Benchmark before and after `sentry_sdk.init()` in any high-throughput async service — treat the SDK as a performance dependency, not a zero-cost observer.
- Use `traces_sample_rate=0.01` (1%) rather than `1.0` to reduce the proportion of requests that pay full tracing overhead.
- For services where throughput matters more than distributed tracing, use `integrations=[]` to disable automatic instrumentation and capture exceptions manually with `capture_exception()`.

## Key Takeaway

Observability SDKs that patch asyncio internals have non-trivial per-request overhead in high-throughput services — benchmark and tune `traces_sample_rate` before deploying to production.
