# Lesson: Histogram Bin Indexing Crashes on Out-of-Range Inference Values — Missing Clamp

**Date:** 2026-02-28
**System:** community (yzhao062/pyod)
**Tier:** lesson
**Category:** data-model
**Keywords:** histogram, HBOS, bins, IndexError, out-of-range, inference, training range, distribution shift, clamp, clip
**Source:** https://github.com/yzhao062/pyod/issues/643

---

## Observation (What Happened)

HBOS (Histogram-Based Outlier Detection) with `n_bins='auto'` raised `IndexError: index N is out of bounds for axis 0 with size N` during prediction when test data contained values outside the training data range. The bin edges were computed from training data and the lookup array was sized exactly for that range — no guard existed for values beyond the histogram's right edge.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `IndexError` on `outlier_scores[j, i] = out_score_i[bin_inds[j] - 1]` during prediction.
**Why #2:** `np.digitize` assigned a bin index equal to `len(bin_edges)` for values beyond the right edge, which is out of bounds for an array of size `len(bin_edges) - 1`.
**Why #3:** The `n_bins='auto'` code path did not clamp out-of-range values to the nearest valid bin (as the fixed `n_bins=int` path did).
**Why #4:** Histogram bin lookup assumes inference data stays within training range — a valid assumption at design time but violated in practice by distribution shift, sensor noise, or test data edge cases.
**Why #5:** The two code paths (`auto` vs `int`) had inconsistent boundary handling — a divergence introduced during the auto-bin feature addition.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Clamp inference values to `[bin_edges.min(), bin_edges.max()]` before bin lookup: `X_clipped = np.clip(X, bin_edges.min(), bin_edges.max())` | proposed | community | issue #643 |
| 2 | When two code paths handle the same operation (histogram scoring), extract shared logic to prevent divergent boundary handling | proposed | community | issue #643 |
| 3 | Add a test that predicts on a sample with values outside the training range and verifies it returns a valid score rather than raising | proposed | community | issue #643 |

## Key Takeaway

Histogram-based scoring must clamp inference values to the training range before bin lookup — distribution shift will always produce out-of-range values, and a missing clamp turns a normal operational event into an `IndexError` crash.
