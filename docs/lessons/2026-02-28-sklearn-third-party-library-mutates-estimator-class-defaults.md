# Lesson: Third-Party Library Can Mutate sklearn Estimator Class-Level Defaults at Import Time

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** configuration
**Keywords:** sklearn, global-state, estimator, class-mutation, import-side-effect, third-party, FastICA, parameter-constraints, namespace-pollution
**Source:** https://github.com/scikit-learn/scikit-learn/issues/32929

---

## Observation (What Happened)

Importing the `picard` ICA library (a third-party scikit-learn plugin) caused `FastICA`'s `_parameter_constraints` class-level dict to be modified. After import, the default value for the `fun` parameter (`'logcosh'`) was no longer accepted — the constraints were replaced with those from the picard implementation. Code that never used picard but shared the Python process with code that imported it would get a mysterious `InvalidParameterError` for a previously valid default.

## Analysis (Root Cause — 5 Whys)

`picard` registers itself as an sklearn plugin via the dispatch mechanism and patches `FastICA._parameter_constraints` at import time to add its own constraint set. Python class attributes are shared global state across all instances in the process. Any import of `picard` in any module — including transitive imports — silently poisoned every subsequent `FastICA` instantiation, regardless of whether the caller knew about `picard`. The error message ("Got 'logcosh' instead") was deeply confusing because `logcosh` is the documented default.

## Corrective Actions

- Never rely on sklearn estimator parameter constraints being stable across the full process lifetime; if a third-party package is used alongside sklearn in the same process, run a sanity check: `FastICA._parameter_constraints['fun']` contains the expected sklearn values.
- Isolate third-party sklearn plugins in subprocess calls when possible, or pin both sklearn and plugin versions together in a constraints file.
- In ARIA: when adding any new Python dependency that declares `sklearn_tags` or `__sklearn_tags__`, verify it doesn't mutate any estimator's `_parameter_constraints` by running `FastICA().get_params()` before and after the import in a test.

## Key Takeaway

A third-party scikit-learn plugin can silently mutate estimator class-level parameter constraints at import time, breaking other estimators in the same process for callers that never used the plugin.
