# Lesson: joblib Memory Cache Under Parallel Execution Produces Partial Writes and Silent Recomputation

**Date:** 2026-02-28
**System:** community (joblib/joblib)
**Tier:** lesson
**Category:** reliability
**Keywords:** joblib, parallel, memory-cache, race-condition, EOFError, cache-corruption, partial-write, multiprocessing
**Source:** https://github.com/joblib/joblib/issues/490

---

## Observation (What Happened)

When a `joblib.Memory`-cached function was executed in parallel using `Parallel(n_jobs=-1)`, multiple workers attempted to write the same cache key simultaneously. One worker would begin writing the pickle file while another read it before the write was complete. The reading worker received an incomplete (truncated) file, causing `EOFError` during unpickling. `joblib` caught the error silently, re-ran the function, and logged a WARNING — the computation was re-executed unnecessarily and the warning was easy to miss.

## Analysis (Root Cause — 5 Whys)

`joblib.Memory` uses filesystem-level pickle files as its cache store. There is no write-lock around cache file creation, so two workers targeting the same cache key race: one writes, another reads mid-write. The partial file passes `os.path.exists()` but fails to unpickle. The `except` block in `_cached_call` catches all exceptions from `_load_output`, silently treats them as cache misses, and re-runs. This means the user sees no error but loses caching efficiency and may silently produce non-deterministic behavior if the function has side effects.

## Corrective Actions

- Use `joblib.Memory` with `Parallel` only for embarrassingly parallel jobs with no shared cache keys — if the same function arguments will be called from multiple workers simultaneously, pre-warm the cache serially before parallelizing.
- When parallel caching is required, wrap the cached call in a file lock (e.g., `fcntl.flock`) or use `diskcache.FanoutCache` which provides proper concurrent write protection.
- In ARIA: the batch ML engine uses joblib for parallel model training. Any function decorated with a disk cache must either be pre-computed before `Parallel` dispatch or accept that cache hits are not guaranteed under concurrent execution.

## Key Takeaway

`joblib.Memory` cache files are not write-safe under concurrent workers — parallel execution of a cached function with the same arguments produces silent re-computation via partial-write / read race.
