# Lesson: Incremental Learning Curve Pre-computes Metadata for Full Dataset — Silently Returns NaN for Sub-sets

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** data-model
**Keywords:** sklearn, learning-curve, incremental-learning, partial-fit, sample-weight, metadata-routing, nan, silent-failure, scorer
**Source:** https://github.com/scikit-learn/scikit-learn/issues/33283

---

## Observation (What Happened)

`learning_curve(..., exploit_incremental_learning=True, params={'sample_weight': weights})` returned `NaN` training scores for all training-size steps except the final one. The incremental path pre-computed the scorer's metadata arguments (including `sample_weight`) once for the full training set, then passed those full-length arrays to scorers operating on subsets. The resulting shape mismatch raised a `ValueError` inside the scorer, which `learning_curve` silently caught and replaced with `NaN`.

## Analysis (Root Cause — 5 Whys)

The `_incremental_fit_estimator` helper was written before sklearn's metadata routing system existed. It pre-fetched params from the routing system at full-training-size granularity, never slicing them to match the current training subset size. The `ValueError` from the scorer was swallowed by a bare `except`, leaving the user with `NaN` scores and no traceback. This is a variant of the "silent NaN from size mismatch" pattern: the scorer receives `X_train[:n]` but `sample_weight[:N]`.

## Corrective Actions

- When wrapping any sklearn learning-curve call with `exploit_incremental_learning=True` and metadata params, run a sanity check: assert no NaNs in training scores; if found, fall back to `exploit_incremental_learning=False` and emit a warning.
- In ARIA's batch engine: any `partial_fit` loop that passes `sample_weight` must slice the weight array to match the current batch size — never pre-extract for the full dataset and reuse.
- Treat `NaN` in sklearn metric arrays as an error condition requiring investigation, not a missing-data sentinel.

## Key Takeaway

When using `partial_fit` or incremental learning with metadata arguments like `sample_weight`, slice every auxiliary array to the current subset size — passing full-dataset arrays to a subset scorer silently returns NaN.
