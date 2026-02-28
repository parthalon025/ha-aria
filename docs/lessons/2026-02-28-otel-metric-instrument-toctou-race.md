# Lesson: OTEL Metric Instrument Registry TOCTOU Race Raises KeyError Under Concurrent Creation

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** opentelemetry, metrics, race condition, TOCTOU, KeyError, instrument registry, threading, meter, create_histogram
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4892

---

## Observation (What Happened)

Under high concurrency (100 threads simultaneously calling `meter.create_histogram()` with the same name), a `KeyError` was raised intermittently. One thread published the instrument ID to the "exists" set before storing the actual instrument object in the instrument dict. A second thread saw "already registered," skipped creation, and immediately performed a dict lookup that found nothing — producing a `KeyError`.

## Analysis (Root Cause — 5 Whys)

The SDK maintained two separate registries under separate locks: an `instrument_id_set` (API layer) and an `instrument_id_instrument` dict (SDK layer). The "set" was updated before the "dict," creating a window where existence was signaled but the object was absent. This is a classic TOCTOU split-brain pattern — `check` and `store` must be atomic over a single data structure under a single lock.

## Corrective Actions

- Any "get-or-create" pattern must use a single dict under a single lock: `with lock: return registry.setdefault(key, factory())` — atomically returns existing value or stores and returns the new one.
- Never split "existence check" and "object storage" across two separate data structures unless both are updated under the same lock and in the same critical section.
- Write a concurrent stress test: spin up N threads calling `create_histogram(same_name)` and assert all return non-None without error.

## Key Takeaway

"Check then act" across two separate data structures is never atomic — use a single registry with a single lock and `setdefault`-style get-or-create.
