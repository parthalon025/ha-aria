# Lesson: LogisticRegressionCV Produces Wrong Coefficients When a Fold Is Missing a Class

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** data-model
**Keywords:** sklearn, logistic-regression, cross-validation, missing-class, fold-class-imbalance, coefficient-error, multiclass, kfold, silent-wrong-result
**Source:** https://github.com/scikit-learn/scikit-learn/issues/32748

---

## Observation (What Happened)

`LogisticRegressionCV` trained on the Iris dataset with `KFold(3)` (unshuffled) produced non-zero coefficients for classes that should have been forced to zero — because those classes were entirely absent from the training fold. The ordered Iris dataset means each consecutive 50-sample block belongs to one class, so `KFold(3)` creates folds where each training set contains exactly 2 of 3 classes. The missing class's coefficients were not zeroed out; they received small near-zero values, and in some versions of the bug, the reshape failed with a `ValueError` entirely.

## Analysis (Root Cause — 5 Whys)

`_logistic_regression_path` assumed all classes were present in the training fold. When a class is absent, the coefficient initialization or result array indexing uses the full number of classes rather than the observed number, causing either wrong values or a shape mismatch. The root cause is that `LogisticRegressionCV` does not validate that all classes appear in each fold before dispatching to the solver. Sorted datasets naturally create this condition with sequential folds.

## Corrective Actions

- Always shuffle data before `KFold` cross-validation for multi-class problems with ordered datasets: use `StratifiedKFold` (which guarantees class representation per fold) instead of plain `KFold`.
- Add a pre-flight assertion: `assert len(np.unique(y_train)) == len(np.unique(y))` before fitting any multi-class model, or use `StratifiedKFold` to enforce it automatically.
- In ARIA's presence classifier and sequence anomaly detector: use `StratifiedKFold` for all cross-validation that involves multi-class labels; never use plain `KFold` on unaggregated time-ordered data.

## Key Takeaway

Use `StratifiedKFold` for multi-class problems; plain `KFold` on sorted datasets will create training folds missing entire classes, producing wrong or crashed model fits.
