# Lesson: sklearn ConvergenceWarning Is a Warning, Not an Exception — Must Be Caught Explicitly

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** sklearn, convergence, MLPRegressor, ConvergenceWarning, warnings, catch_warnings, silent, training quality
**Files:** aria/engine/models/autoencoder.py:45-52

---

## Observation (What Happened)

`MLPRegressor.fit()` in the autoencoder emits a `ConvergenceWarning` (not an exception) when training doesn't converge. In production, this Python warning is silently swallowed — it's not logged, not surfaced in the return dict, and invisible in journald. Callers have no way to know whether the autoencoder trained well or barely trained at all.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `ConvergenceWarning` is a Python `warnings.warn()` call, not an exception — `try/except` cannot catch it.

**Why #2:** sklearn uses warnings instead of exceptions for non-fatal training issues, making it easy to miss: the call returns a model regardless.

**Why #3:** The training pipeline checks return values but not the warning channel, so a poorly-converged model is indistinguishable from a fully-converged one from the caller's perspective.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap `.fit()` in `warnings.catch_warnings(record=True)`, check for `ConvergenceWarning`, and log at WARNING level | proposed | Justin | issue #204 |
| 2 | Include a convergence flag in the training result dict so callers can gate on model quality | proposed | Justin | issue #204 |
| 3 | Apply the same pattern to all other `MLPRegressor.fit()` / `SGDClassifier.fit()` calls in the codebase | proposed | Justin | — |

## Key Takeaway

sklearn `ConvergenceWarning` is a Python warning, not an exception — it cannot be caught with `try/except`; use `warnings.catch_warnings(record=True)` and check the warning list explicitly to detect and log poor model convergence.
