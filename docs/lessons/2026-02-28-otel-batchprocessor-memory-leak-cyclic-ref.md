# Lesson: OTEL BatchLogRecordProcessor Not Garbage Collected Due to Cyclic Reference

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** opentelemetry, BatchLogRecordProcessor, memory leak, garbage collection, cyclic reference, shutdown, background thread, weak reference
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4422

---

## Observation (What Happened)

Each instantiation of `BatchLogRecordProcessor` after `.shutdown()` and `del` still appeared in `gc.get_objects()`, consuming ~240 KiB per instance. Applications that create multiple providers over their lifetime (tests, hot-reload, dynamic tracing setup) leaked unbounded memory.

## Analysis (Root Cause — 5 Whys)

The `BatchLogRecordProcessor` spawns a background daemon thread. The thread holds a strong reference back to the processor (via `self` in its target function), and the processor holds a reference to the thread. This cyclic reference is not automatically broken by Python's reference counter — it requires the cyclic garbage collector to identify and collect it. Because the thread was still alive (daemon threads keep running until process exit), the cycle was never broken and the processor was never collected.

## Corrective Actions

- Background workers that reference their owner object must use `weakref.ref(self)` in the thread target to avoid preventing garbage collection.
- Always call `.shutdown()` before deleting a processor, then verify collection: `gc.collect(); assert not any(isinstance(o, BatchLogRecordProcessor) for o in gc.get_objects())`.
- In tests, create and shut down OTEL providers in a fixture with `yield` to ensure cleanup runs even on test failure.

## Key Takeaway

Background threads that hold a strong reference back to their owning object prevent garbage collection — use `weakref.ref` in thread targets or ensure shutdown explicitly breaks the reference cycle.
