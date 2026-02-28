# Lesson: OTEL Span Context Manager Marks GeneratorExit and SystemExit(0) as ERROR

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** error-handling
**Keywords:** opentelemetry, span, GeneratorExit, SystemExit, error status, BaseException, context manager, generator cleanup
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4484

---

## Observation (What Happened)

`opentelemetry-sdk 1.31.0` regressed such that any span whose `with` block exited via `GeneratorExit` (generator closed normally) or `sys.exit(0)` (clean exit) had its status set to `ERROR`. This produced false-positive error spans for all generators that use spans internally, making dashboards report routine cleanup operations as failures.

## Analysis (Root Cause — 5 Whys)

A PR changed the span context manager's `__exit__` to record any exception as an error and set status to `ERROR`. The change did not distinguish between exception classes that represent errors (`Exception` subclasses) and those that represent control flow (`GeneratorExit`, `SystemExit(0)`, `KeyboardInterrupt`). `GeneratorExit` is raised by Python when a generator is `.close()`d — it is not an error. `SystemExit(0)` signals clean process exit. Both are `BaseException` subclasses, not `Exception` subclasses, for exactly this reason.

## Corrective Actions

- In any span `__exit__` that sets error status, guard: only set `ERROR` if the exception is an `Exception` subclass, or more specifically if it is not in `(GeneratorExit, KeyboardInterrupt)` and (for `SystemExit`) if `sys.exc_info()[1].code != 0`.
- Use OTEL's `record_exception` + `set_status` pattern inside a `with tracer.start_as_current_span(..., record_exception=False)` for generators where you manually control error reporting.
- Add integration tests that verify span status for normal generator close and `sys.exit(0)` paths.

## Key Takeaway

Span error recording must filter out `GeneratorExit`, `KeyboardInterrupt`, and `SystemExit(0)` — these are control-flow signals, not errors, and marking them ERROR produces false-positive alert noise.
