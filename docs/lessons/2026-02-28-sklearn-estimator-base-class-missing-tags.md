# Lesson: sklearn Estimator Missing `__sklearn_tags__` — `BaseEstimator` Inheritance Required for Forward Compatibility

**Date:** 2026-02-28
**System:** community (yzhao062/pyod)
**Tier:** lesson
**Category:** integration
**Keywords:** scikit-learn, sklearn, BaseEstimator, __sklearn_tags__, get_tags, version compatibility, estimator, inheritance, breaking change
**Source:** https://github.com/yzhao062/pyod/issues/649

---

## Observation (What Happened)

pyod models broke with `AttributeError: 'IForest' object has no attribute '__sklearn_tags__'` after upgrading to scikit-learn 1.8.0. The `default_tags` function was removed and what was previously a deprecation warning became a hard exception — any custom estimator that did not properly inherit from `BaseEstimator` or implement `__sklearn_tags__` was broken at `decision_function()` call time.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `AttributeError` on `__sklearn_tags__` raised in `check_is_fitted()`.
**Why #2:** pyod's wrapper classes did not inherit from `sklearn.base.BaseEstimator` on the right side of the MRO, so `__sklearn_tags__` was not resolved.
**Why #3:** sklearn's tag system used `default_tags()` as a fallback that was deprecated in an earlier version but only removed in 1.8.0.
**Why #4:** The deprecation warning was silently ignored during a dependency bump — no CI test verified compatibility with the new sklearn release.
**Why #5:** Custom estimator wrappers frequently copy sklearn's public interface without inheriting from its base classes, creating a compatibility debt that surfaces only on major version bumps.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Any custom ML estimator class must inherit from `sklearn.base.BaseEstimator` — not just implement `fit`/`predict`/`transform` | proposed | community | issue #649 |
| 2 | Pin upper bounds on sklearn in `pyproject.toml` when wrapping sklearn internals; add CI matrix job for the next sklearn release candidate | proposed | community | issue #649 |
| 3 | When a sklearn deprecation warning appears in test output, treat it as a P2 issue — it will become a hard break in the next major version | proposed | community | issue #649 |

## Key Takeaway

Custom sklearn estimator wrappers that implement only the public API without inheriting from `BaseEstimator` accumulate silent compatibility debt — every sklearn major version may promote their deprecation warnings to hard exceptions.
