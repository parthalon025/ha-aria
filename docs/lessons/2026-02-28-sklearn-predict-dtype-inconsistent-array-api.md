# Lesson: Classifier predict() Output dtype Varies by Input dtype and Array Dispatch Mode — Not Stable

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** data-model
**Keywords:** sklearn, predict, dtype, array-api, classifier, float32, int64, output-type, inconsistency, array-dispatch, classifier-output
**Source:** https://github.com/scikit-learn/scikit-learn/issues/33308

---

## Observation (What Happened)

`RidgeClassifier.predict()` returned different output dtypes depending on input dtype and whether Array API dispatch was enabled: float64 input + Array API dispatch returned `int64`; float32 input returned `float32`; pandas float64 input returned `int64`; array_api_strict float64 returned `float64`. The documented rule ("everything follows X") was not implemented consistently, making downstream dtype assertions brittle.

## Analysis (Root Cause — 5 Whys)

The Array API support path and the standard numpy path used different dtype-coercion logic. The numpy path coerced predictions to match input dtype for float32 but fell back to int64 for float64 (since class labels are integers). The Array API path preserved the input array library but used the label dtype for the output. The two paths were independently added without a shared dtype-resolution function, so they diverged silently. No test existed that asserted output dtype consistency across input dtype variants.

## Corrective Actions

- Never assert a specific output dtype from `predict()` without testing against the exact sklearn version and array dispatch mode used in production; use `np.asarray(pred).astype(np.int64)` explicitly if integer class labels are required.
- When writing ARIA feature pipelines that feed sklearn classifiers: normalize all outputs to `int64` (for class predictions) or `float64` (for probabilities) immediately after the `predict()` / `predict_proba()` call, before storing or comparing.
- Add a dtype assertion in the classifier integration test: `assert pred.dtype in (np.int32, np.int64)` so any future sklearn version change that alters output dtype is caught immediately.

## Key Takeaway

Classifier `predict()` output dtype is not guaranteed to follow input dtype across sklearn versions and Array API dispatch modes — always normalize prediction outputs to a known dtype immediately after calling predict.
