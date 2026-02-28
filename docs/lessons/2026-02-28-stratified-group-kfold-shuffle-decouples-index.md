# Lesson: Shuffling One Array Without Shuffling Its Index Map Silently Corrupts CV Splits

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** data-model
**Keywords:** cross-validation, stratification, shuffle, index-mapping, group-kfold, silent-corruption, fold-leakage, ml-pipeline
**Source:** https://github.com/scikit-learn/scikit-learn/issues/32478

---

## Observation (What Happened)

`StratifiedGroupKFold(shuffle=True)` shuffled its internal `y_counts_per_group` matrix to introduce randomness, but did not apply the same permutation to the companion `groups_inv` index that maps matrix rows back to group IDs. After shuffling, the greedy fold-assignment algorithm operated on a permuted matrix, but `test_indices` were built using the unshuffled `groups_inv`, so every fold received a random permutation of the groups the algorithm actually balanced. The result was that stratification was silently dropped — the splitter behaved like a plain `GroupKFold`.

## Analysis (Root Cause — 5 Whys)

The shuffle was added only to the data matrix, not to all structures that depend on row-to-group identity. Any algorithm that maintains parallel arrays (data matrix + index map + groups array) must apply every permutation to all three simultaneously. The bug was invisible at the API level: no exception, no warning, fold sizes looked reasonable. Detection required computing per-fold positive-class ratios and noticing they deviated from the expected 1/3. The stable sort used after shuffling meant the effect was subtle: only groups with identical variance were reordered, but the mismatch still corrupted the group membership lookup.

## Corrective Actions

- When shuffling any array that is paired with an index structure, apply `numpy.argsort` / the same permutation to all paired arrays in the same operation.
- Add a post-split assertion: for `StratifiedGroupKFold`, verify that each fold's class ratio is within tolerance of the global class ratio — fail loudly if not.
- In ARIA: any time cross-validation splits are used for time-series data, explicitly pass `groups=` and verify no group leaks across train/test folds with `assert len(set(train_groups) & set(test_groups)) == 0`.

## Key Takeaway

Shuffling a data matrix without synchronously shuffling every companion index or map that relies on row identity silently corrupts all downstream lookups.
