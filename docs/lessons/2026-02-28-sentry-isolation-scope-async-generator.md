# Lesson: Sentry isolation_scope Context Manager Inside Async Generator Leaks Token Context

**Date:** 2026-02-28
**System:** community (getsentry/sentry-python)
**Tier:** lesson
**Category:** async
**Keywords:** asyncio, context variable, isolation scope, async generator, GeneratorExit, contextvars.Token, context leak
**Source:** https://github.com/getsentry/sentry-python/issues/4925

---

## Observation (What Happened)

Using `sentry_sdk.isolation_scope()` as a context manager wrapping `yield` statements inside an async generator caused a `RuntimeError: Token was created in a different Context` when the generator exited early (via `break`, exception, or garbage collection). Background tasks using FastAPI's `BackgroundTasks` also silently lost their scope context, dropping breadcrumbs and custom tags.

## Analysis (Root Cause — 5 Whys)

`contextvars.ContextVar.set()` returns a `Token` that must be reset in the same `Context` it was created in. Async generators and background tasks run in a copied context — Python's `asyncio` forks the context at generator creation or task dispatch. When the generator is garbage-collected or interrupted at a `yield` midway through the `__aexit__`, the `Token.reset()` call runs in a different context than the one that issued it, triggering the error. The scope that was set on task dispatch is invisible inside the task because each task carries its own copy.

## Corrective Actions

- Do not use `isolation_scope()` or any `contextvars`-backed context manager that spans a `yield` inside an async generator. Restructure so scope setup and cleanup are in the same stack frame.
- For background tasks that need Sentry scope: capture the scope before dispatch and pass it as an argument, then call `scope.continue_trace()` inside the task.
- Test by explicitly sending a background task that logs breadcrumbs and verifying they appear in the captured event.

## Key Takeaway

Python forks `contextvars` context at `asyncio.create_task()` and async generator creation — any `ContextVar` token set in the parent context cannot be reset from a forked context.
