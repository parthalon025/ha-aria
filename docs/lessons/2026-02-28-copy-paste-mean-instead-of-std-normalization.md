# Lesson: Copy-Paste Error Assigns Mean to `std` Variable — Silent Normalization Corruption

**Date:** 2026-02-28
**System:** community (yzhao062/pyod)
**Tier:** lesson
**Category:** data-model
**Keywords:** normalization, preprocessing, mean, std, standard deviation, copy-paste, silent bug, autoencoder, training, pytorch
**Source:** https://github.com/yzhao062/pyod/issues/391

---

## Observation (What Happened)

The PyTorch autoencoder in pyod assigned the training data's mean to the `std` variable during preprocessing (`std = X.mean()` instead of `std = X.std()`). All normalization was therefore dividing by the mean rather than the standard deviation, silently producing wrong-scaled inputs to the neural network with no error or warning.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Model training produced unexpectedly poor anomaly detection scores.
**Why #2:** Normalization divided by mean instead of std, distorting feature scales.
**Why #3:** Copy-paste of the mean computation line was not updated to `std()` when creating the std variable.
**Why #4:** No unit test verified that `mean ≈ 0` and `std ≈ 1` after preprocessing on known input.
**Why #5:** Preprocessing functions in ML pipelines are treated as boilerplate and rarely have per-step validation.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add a preprocessing sanity check after normalization: assert `abs(X_norm.mean()) < 0.1` and `abs(X_norm.std() - 1.0) < 0.1` on a known input | proposed | community | issue #391 |
| 2 | Use `sklearn.preprocessing.StandardScaler` instead of manual mean/std computation — it is tested and audited | proposed | community | issue #391 |
| 3 | Any data pipeline step that computes statistics must have a test with a known-answer fixture (e.g., input=[0,2,4] → mean=2, std=2) | proposed | community | issue #391 |

## Key Takeaway

Manual mean/std normalization code is a high-probability copy-paste failure zone — use a tested library (StandardScaler) or add a post-normalization assertion that verifies mean≈0 and std≈1, because silent wrong-scale preprocessing corrupts every downstream model.
