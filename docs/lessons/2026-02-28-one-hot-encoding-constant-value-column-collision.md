# Lesson: One-Hot Encoding Constant-Value Categorical Features Produces Colliding Column Names — Inverse Transform Fails

**Date:** 2026-02-28
**System:** community (unit8co/darts)
**Tier:** lesson
**Category:** data-model
**Keywords:** one-hot encoding, OHE, column name collision, constant value, zero cardinality, categorical, inverse transform, static covariates, sklearn
**Source:** https://github.com/unit8co/darts/issues/2705

---

## Observation (What Happened)

`StaticCovariatesTransformer` with `OneHotEncoder` failed to invert when two categorical features had the same constant value (e.g., both always `'foo'`). Zero-cardinality features (only one unique value) were encoded to a column named just `foo` rather than `cov_a_foo` — so two different features produced the same column name. Inverse transform then failed with `IndexError: boolean index did not match` because the column mapping was irrecoverably ambiguous.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `inverse_transform()` raised `IndexError` on any dataset with two constant-value categorical covariates sharing the same value.
**Why #2:** The encoded column for a zero-cardinality feature was named `{value}` instead of `{feature}_{value}`, causing name collision.
**Why #3:** The encoder's naming logic had a special case for single-value features that omitted the feature name prefix.
**Why #4:** Column name uniqueness was not enforced after encoding — collision was only detected at inverse transform time.
**Why #5:** The bug is latent in small/toy datasets (where constant features are common for testing) and only surfaces in production with real feature diversity.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Encoded column names must always include the source feature name as a prefix regardless of cardinality: `{feature}_{value}`, never just `{value}` | proposed | community | issue #2705 |
| 2 | After encoding, assert that all output column names are unique before returning — fail loudly rather than produce an uninvertable transform | proposed | community | issue #2705 |
| 3 | Add a test with two categorical features sharing the same constant value to verify encode → inverse_transform round-trips correctly | proposed | community | issue #2705 |

## Key Takeaway

One-hot encoding column names must always be qualified with the source feature name — any shortcut that omits the feature prefix for "simple" cases (zero or one cardinality) creates silent column collisions that only surface as cryptic IndexErrors at inverse transform time.
