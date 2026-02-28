# Lesson: Prometheus Metric Lock Held During GC Callback Causes Deadlock

**Date:** 2026-02-28
**System:** community (prometheus/client_python)
**Tier:** lesson
**Category:** reliability
**Keywords:** prometheus, deadlock, garbage collector, gc callback, mutex, lock, metrics collection, gc.callbacks
**Source:** https://github.com/prometheus/client_python/issues/363

---

## Observation (What Happened)

A Prometheus HTTP metrics endpoint hung permanently. The sequence: thread A acquired the `MutexValue` lock inside `get()` to read a metric value; the GC ran during `get()` and triggered a gc-callback that tried to update the GC metrics, which in turn tried to acquire the same lock. Thread A waited for the GC callback to finish; the callback waited for thread A's lock — classic deadlock.

## Analysis (Root Cause — 5 Whys)

GC callbacks in Python run synchronously in the thread that triggered GC. If that thread holds a non-reentrant lock, and the GC callback tries to acquire the same lock, the result is deadlock. Prometheus default GC metrics register `gc.callbacks` that call into metric update code. Any metric access pattern that holds a lock during code that can allocate objects (triggering GC) is vulnerable. The root fix is to not use reentrant locks in GC-triggered paths or to ensure GC callbacks do not attempt to acquire locks held by the GC-triggering thread.

## Corrective Actions

- GC metric collectors must use lock-free atomic types (e.g., `threading.local` counters) or use `threading.RLock` (re-entrant) to allow the same thread to reacquire the lock.
- Do not allocate Python objects inside `MutexValue.get()` (or any locked section that GC-triggered callbacks may re-enter).
- When implementing custom metrics that use callbacks (gc, memory profilers), test under `gc.set_threshold(0)` to make GC maximally aggressive and expose this class of bug.

## Key Takeaway

GC callbacks run on the thread that triggered GC — holding a non-reentrant lock during any allocating code in a metrics path can deadlock with GC-callback metric updates.
