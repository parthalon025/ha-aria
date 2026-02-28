# Lesson: FeatureUnion With Dataframe Output Fails When Adapter hstack Interface Lacks Feature Renaming

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** integration
**Keywords:** sklearn, feature-union, polars, dataframe-output, duplicate-columns, hstack, set-output, verbose-feature-names, adapter-interface
**Source:** https://github.com/scikit-learn/scikit-learn/issues/32852

---

## Observation (What Happened)

`FeatureUnion` with `set_config(transform_output="polars")` raised `polars.exceptions.DuplicateError: column with name 'x0' has more than one occurrence` when two transformers produced outputs with identically named columns. The `verbose_feature_names_out=True` default was supposed to prefix column names (`scaler1__x0`, `scaler2__x0`), but the Polars `ContainerAdapterProtocol.hstack()` implementation didn't accept a `feature_names` parameter, so the renaming step was silently skipped before concatenation.

## Analysis (Root Cause — 5 Whys)

The adapter interface between sklearn's output-API abstraction and Polars was designed for pandas first. When Polars support was added, the `hstack()` adapter method signature didn't include the feature-name-renaming hook that the pandas path used. The architecture assumed adapter implementations would be feature-complete, but the Polars adapter was a minimal port. Silent skip rather than an explicit error means users discover the mismatch only at the Polars layer.

## Corrective Actions

- When using `set_config(transform_output="polars")` with `FeatureUnion`, explicitly call `union.get_feature_names_out()` after fit and assign the resulting names to the output DataFrame manually, or fall back to `transform_output="pandas"`.
- Treat any `DuplicateError` from a concat/hstack operation after a pipeline transform as a signal that feature renaming was silently dropped upstream, not a data error.
- In ARIA: the SPA uses `FeatureUnion`-equivalent patterns (parallel transformer columns merged into one feature matrix) — always assert that the output column count equals the sum of per-transformer output dimensions before merging.

## Key Takeaway

`FeatureUnion` with non-pandas dataframe output may silently skip column renaming if the adapter interface is incomplete — always verify output column names are unique and prefixed after fit_transform.
