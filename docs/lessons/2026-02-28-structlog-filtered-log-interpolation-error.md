# Lesson: structlog Filtering Bound Logger Must Forward All Arguments to the Noop Stub

**Date:** 2026-02-28
**System:** community (hynek/structlog)
**Tier:** lesson
**Category:** error-handling
**Keywords:** structlog, filtering, bound logger, _nop, positional arguments, log level, interpolation, TypeError
**Source:** https://github.com/hynek/structlog/issues/476

---

## Observation (What Happened)

`log.debug('hello %s', 'world')` raised `TypeError: _nop() takes 2 positional arguments but 3 were given` when the log level was set to INFO (filtering out DEBUG). The filtering bound logger replaced the real log method with a `_nop()` stub, but that stub only accepted `(event)` without `*args` — so any positional interpolation arguments caused a crash on the filtered (no-op) path.

## Analysis (Root Cause — 5 Whys)

The `make_filtering_bound_logger(logging.INFO)` stub for below-threshold levels was generated with a signature that did not accept variadic positional arguments. This meant the filtering itself was broken for the common Python logging pattern `log.debug(msg, *args)`. A no-op stub must accept `(event, *args, **kwargs)` to be a transparent replacement for any log call signature — the contract for a filter stub is "accept everything, do nothing."

## Corrective Actions

- No-op log stubs must always be `def _nop(event, *args, **kw): pass` — never fixed-arity — to be safely substitutable for any log method.
- When building filtering wrappers, test with both `log.debug("msg")` and `log.debug("msg %s", "arg")` and `log.debug("msg", key="val")` — all three must not raise.
- If using `structlog.make_filtering_bound_logger()`, upgrade to 22.3.0+ where this is fixed; avoid pinning old versions.

## Key Takeaway

No-op stubs that replace logging methods must accept `(*args, **kwargs)` — fixed-arity stubs break any call site that passes positional interpolation arguments.
