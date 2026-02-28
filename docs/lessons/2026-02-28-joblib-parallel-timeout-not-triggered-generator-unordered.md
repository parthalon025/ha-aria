# Lesson: joblib Parallel timeout Parameter Is Not Enforced for return_as="generator_unordered"

**Date:** 2026-02-28
**System:** community (joblib/joblib)
**Tier:** lesson
**Category:** reliability
**Keywords:** joblib, parallel, timeout, generator-unordered, return-as, hanging, multiprocessing, generator, infinite-loop
**Source:** https://github.com/joblib/joblib/issues/1586

---

## Observation (What Happened)

`Parallel(n_jobs=2, timeout=3, return_as="generator_unordered")` with infinite-loop worker functions never raised a `TimeoutError` and never terminated. The same call with `return_as` omitted (default list mode) correctly raised `TimeoutError` after 3 seconds. The process hung indefinitely when generator-unordered mode was used, making timeout-based resource protection inoperative for streaming result patterns.

## Analysis (Root Cause — 5 Whys)

The timeout mechanism in `joblib.Parallel` was implemented in the result-collection path that aggregates all results into a list. When `return_as="generator_unordered"` is used, result collection is lazy — the generator yields results as they complete, and the timeout check only ran in the eager collection path. The generator path did not have a corresponding timeout check on each `yield`, so workers that never completed were never interrupted. This means `timeout=` is silently a no-op for generator modes.

## Corrective Actions

- Never rely on `Parallel(timeout=...)` for resource control when `return_as="generator_unordered"` or `return_as="generator"` is used; the timeout is not enforced.
- Wrap generator-mode parallel calls with an external `signal.alarm()` or `threading.Timer` to enforce a wall-clock deadline.
- In ARIA: the batch ML engine uses `Parallel` for parallel model training segments. If any call uses `return_as="generator_unordered"` for streaming partial results, add an explicit deadline guard using `concurrent.futures.as_completed(timeout=...)` instead of relying on joblib's `timeout=` parameter.

## Key Takeaway

`joblib.Parallel(timeout=...)` is silently a no-op when `return_as="generator_unordered"` — use an external wall-clock deadline for timeout enforcement in generator-mode parallel calls.
