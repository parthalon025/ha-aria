# Lesson: Numeric Score Sort Is Lexicographic — String Comparison Silently Breaks on Varying Precision

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** sort, lexicographic, numeric, parseFloat, string comparison, JavaScript, Array.sort, score, Preact
**Files:** aria/dashboard/spa/src/pages/Anomalies.jsx

---

## Observation (What Happened)

`Anomalies.jsx` sorted anomaly scores using JavaScript's default `Array.sort()` comparator, which compares values as strings. Scores like `"0.9"`, `"0.85"`, `"0.1"` sorted correctly by accident (2-decimal precision happened to produce correct lexicographic order), masking the bug until a score with different precision (e.g., `"0.123"`) appeared (issue #289).

## Analysis (Root Cause — 5 Whys)

**Why #1:** `Array.sort()` without a comparator function uses string coercion — `0.9 > 0.85` lexicographically because `"9" > "8"` at the second character.

**Why #2:** The bug was latent: with uniform 2-decimal scores, lexicographic and numeric order happen to agree, so no test caught it.

**Why #3:** The developer did not specify a numeric comparator, relying on JavaScript's default sort behavior without knowing it stringifies values.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace bare `array.sort()` with `array.sort((a, b) => parseFloat(b.score) - parseFloat(a.score))` for any numeric field | proposed | Justin | Anomalies.jsx #289 |
| 2 | Treat any sort of numeric-valued strings as requiring an explicit `parseFloat()` / `parseInt()` comparator | proposed | Justin | — |

## Key Takeaway

JavaScript's default `Array.sort()` is lexicographic — always provide an explicit numeric comparator (`parseFloat(b) - parseFloat(a)`) when sorting score or numeric fields.
