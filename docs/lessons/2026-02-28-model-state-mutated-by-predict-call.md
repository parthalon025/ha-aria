# Lesson: `predict()` Mutates `model.model` — Internal Loop Overwrites Public State

**Date:** 2026-02-28
**System:** community (unit8co/darts)
**Tier:** lesson
**Category:** data-model
**Keywords:** model state, mutation, predict, quantile regression, stateful, public attribute, QuantileRegressor, post-fit, side effect
**Source:** https://github.com/unit8co/darts/issues/2836

---

## Observation (What Happened)

In Darts's `LinearRegressionModel` with `likelihood="quantile"`, calling `model.predict(n=6, num_samples>1)` overwrote the public `model.model` attribute with the last `QuantileRegressor` iterated in an internal loop. After prediction, accessing `model.model` (e.g., to retrieve coefficients) returned the wrong estimator. The root cause was `model.model = fitted` inside `_estimator_predict()` — a line that was not meant to be a persistent side effect.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `model.model[0.5].coef_` raised `AttributeError` after calling `predict()` — `model.model` was no longer a `QuantileModelContainer` but a single `QuantileRegressor`.
**Why #2:** The internal `_estimator_predict()` loop assigned `model.model = fitted` during iteration — intended as a temporary local reference, not a persistent assignment.
**Why #3:** The container pattern (`_model_container`) was added later and the original assignment was not removed.
**Why #4:** No test verified `model.model` identity before and after `predict()`.
**Why #5:** `predict()` was not treated as a pure read operation — it silently wrote to public state.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | `predict()` must be a read-only operation — any internal iteration that needs a temporary reference must use a local variable, not assign to a public attribute | proposed | community | issue #2836 |
| 2 | Add a test that checks `model.model is model.model` (identity) before and after `predict()` for any model with quantile or ensemble internals | proposed | community | issue #2836 |
| 3 | Audit all `_estimator_predict()` and similar internal methods for assignments to `self.*` that are not explicitly documented as state-changing | proposed | community | issue #2836 |

## Key Takeaway

`predict()` must never assign to public attributes — internal iteration variables that temporarily hold sub-estimator references must use local variables only; any assignment to `model.model` inside predict silently corrupts state for every subsequent introspection call.
