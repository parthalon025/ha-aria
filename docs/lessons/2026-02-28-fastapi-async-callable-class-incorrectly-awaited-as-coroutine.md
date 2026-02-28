# Lesson: FastAPI Incorrectly Identifies Callable Class Instances as Coroutines and Awaits Them
**Date:** 2026-02-28
**System:** community (tiangolo/fastapi)
**Tier:** lesson
**Category:** async
**Keywords:** FastAPI, dependency injection, callable class, __call__, is_coroutine_callable, await, TypeError, coroutine detection
**Source:** https://github.com/tiangolo/fastapi/issues/14466
---
## Observation (What Happened)
In FastAPI 0.123.6+, a class with `async def __call__(self)` is incorrectly identified as a "coroutine callable" by `Dependant.is_coroutine_callable`. When an instance of the class is used as a dependency, FastAPI tries to `await` the *instance itself* (not the result of calling it), raising `TypeError: 'MyClass' object can't be awaited`. The bug is in FastAPI's introspection logic — it inspects the class's `__call__` method's async nature but then mistakenly applies the await to the object rather than the call result.

## Analysis (Root Cause — 5 Whys)
FastAPI's dependency resolution uses `is_coroutine_callable(call)` to determine whether to `await` the result of calling a dependency. The implementation changed in 0.123.6 to use `inspect.iscoroutinefunction` on the class itself, which returns `True` when `__call__` is `async def`. This causes `solve_dependencies` to use `solved = await call(...)` syntax, which first awaits the class instance (not the call result), producing the TypeError. The correct check is whether calling the object produces a coroutine, not whether the class is callable with an async method.

## Corrective Actions
- Pin FastAPI version and validate callable class dependencies in a test that actually invokes the endpoint, not just imports the app.
- For callable class dependencies with `async def __call__`, add a smoke test route in the test suite that exercises the dependency end-to-end.
- As a workaround when hitting this on older FastAPI: define the dependency as a plain `async def` function instead of a callable class until the version is upgraded.
- When upgrading FastAPI, run the full test suite including routes that use callable class dependencies before deploying.

## Key Takeaway
FastAPI's `is_coroutine_callable` introspection for class-based dependencies can misfire on version changes — always test callable class dependencies end-to-end, not just at import time.
