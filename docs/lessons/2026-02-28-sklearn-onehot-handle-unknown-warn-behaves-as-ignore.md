# Lesson: OneHotEncoder handle_unknown='warn' Silently Behaves as 'ignore' Not 'infrequent_if_exist'

**Date:** 2026-02-28
**System:** community (scikit-learn/scikit-learn)
**Tier:** lesson
**Category:** data-model
**Keywords:** sklearn, one-hot-encoder, handle-unknown, infrequent-categories, documentation-mismatch, silent-wrong-output, preprocessing, transformer
**Source:** https://github.com/scikit-learn/scikit-learn/issues/32589

---

## Observation (What Happened)

`OneHotEncoder(handle_unknown='warn', min_frequency=2)` was expected (per docs) to map unknown categories to the infrequent bucket — the same behavior as `handle_unknown='infrequent_if_exist'` but with a warning. In practice, `'warn'` produced an all-zeros row for unknown categories, matching `handle_unknown='ignore'`. No exception was raised, no assertion failed — downstream models silently received a zero vector instead of the infrequent indicator.

## Analysis (Root Cause — 5 Whys)

The `'warn'` path shared its transform logic with the `'ignore'` path rather than the `'infrequent_if_exist'` path, despite the documentation stating the opposite. The warning was emitted correctly, so the encoder appeared to be working. Zero-vector output is numerically valid, so no dtype error or shape error surfaced. Any model trained with `'infrequent_if_exist'` but evaluated with `'warn'` would silently receive different encoded features for unknown categories, causing unexplained prediction shifts.

## Corrective Actions

- Do not use `handle_unknown='warn'` as a drop-in for `'infrequent_if_exist'` until this is verified fixed in the installed sklearn version; use `'infrequent_if_exist'` directly for production pipelines.
- After fitting any encoder with infrequent-category logic, inject a known-unknown test case in the integration test suite and assert the output is non-zero (i.e., mapped to the infrequent indicator).
- In ARIA: feature preprocessing pipelines should pin the `handle_unknown` value explicitly — never assume `'warn'` is a safe alias for another mode.

## Key Takeaway

Never assume a 'warn' variant of an encoder parameter is behaviorally equivalent to the documented fallback mode — test with a known unknown category and assert the output matches the non-warn variant.
