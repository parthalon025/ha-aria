# Lesson: cross_validate With n_jobs > 1 Is ~100x Slower on First Call Due to Subprocess Startup Cost

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** performance
**Keywords:** sklearn, cross-validate, n-jobs, parallel, subprocess-spawn, loky, cold-start, first-call, performance, joblib-worker-pool
**Source:** https://github.com/scikit-learn/scikit-learn/issues/33112

---

## Observation (What Happened)

The first call to `cross_validate(est, X, y, n_jobs=2)` took ~1.15 seconds; subsequent calls took ~0.01 seconds — a 100x penalty on the first call. The overhead was not due to the model or data size (a simple `Ridge` on 100 samples × 5 features). It only occurred with `n_jobs > 1`. Subsequent calls were fast because the joblib Loky worker pool was already warm.

## Analysis (Root Cause — 5 Whys)

joblib's default Loky backend spawns worker processes on first use. Process spawn on macOS/Linux involves fork+exec, dynamic linker startup, Python interpreter initialization, and module import in each worker — totaling 0.5-1.5 seconds on typical hardware. This cost is amortized across subsequent calls because Loky reuses the worker pool. The issue is not a bug per se but an undocumented cold-start tax. For benchmarks or correctness tests that run `cross_validate` once and compare to a sequential baseline, the cold-start makes parallel appear slower.

## Corrective Actions

- When benchmarking or timing cross-validation, always warm the worker pool first: call `cross_validate(est, X[:10], y[:10], n_jobs=2)` once before the timed run, or pre-warm explicitly via `joblib.parallel.get_active_backend()`.
- For ARIA's batch ML engine: schedule parallel CV calls after the process has been running for a minute (warm pool), or explicitly pre-warm during hub startup if CV latency is latency-sensitive.
- Never reject `n_jobs > 1` based on a single-call benchmark; always average over at least 3 calls after a warmup iteration.

## Key Takeaway

The first `n_jobs > 1` parallel call in a process pays a 0.5–1.5s worker-pool spawn cost; always warm the pool before benchmarking or latency-sensitive parallel CV calls.
