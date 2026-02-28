# Lesson: OTEL Metrics Collection Raises RuntimeError When Metric Labels Updated During Iteration

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** opentelemetry, metrics, dictionary iteration, RuntimeError, label set, metric_reader_storage, concurrent modification, collect
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4785

---

## Observation (What Happened)

`RuntimeError: dictionary changed size during iteration` occurred intermittently (~3 times/day) in `metric_reader_storage.collect()` when Prometheus scraped metrics while new label combinations were being recorded concurrently. The exception was caught by Sentry but silently swallowed by the collection loop, dropping the entire metric export for that scrape cycle.

## Analysis (Root Cause — 5 Whys)

The `collect()` method iterated over the internal label-keyed dict without holding a lock or taking a snapshot copy. A concurrent `record()` call (from a request handler on another thread) added a new label set while `collect()` was mid-iteration, causing Python to raise `RuntimeError` because the dict's internal structure changed. This is the exact scenario where `dict.copy()` or `list(dict.items())` must be used before iteration in any multi-threaded context.

## Corrective Actions

- Any iteration over a dict that is concurrently modified must be snapshot-first: `for key, value in list(d.items()):` or hold the lock for the duration of iteration.
- Wrap metric collection loops in `try/except RuntimeError as e: if "dictionary changed size" in str(e): retry_once_with_lock()`.
- Expose a metric for collection failures — silent loss of a scrape cycle is worse than a logged error.

## Key Takeaway

Iterating a `dict` that another thread can mutate will raise `RuntimeError` — always take a `list(d.items())` snapshot or hold the lock across the entire iteration.
