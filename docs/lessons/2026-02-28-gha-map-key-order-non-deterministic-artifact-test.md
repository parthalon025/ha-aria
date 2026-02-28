# Lesson: Map/Dict Key Order Is Non-Deterministic in CI Test Assertions — Equality Checks on Artifact Structs Are Flaky

**Date:** 2026-02-28
**System:** community (goreleaser/goreleaser)
**Tier:** lesson
**Category:** testing
**Keywords:** github-actions, flaky-test, map, dict, key-order, non-deterministic, artifact, equality, struct
**Source:** https://github.com/goreleaser/goreleaser/issues/5504

---

## Observation (What Happened)

A GoReleaser test that checked full artifact struct equality failed intermittently in CI (`actions/runs/13048540240`). The artifact struct contained a `map` field — map iteration order in Go (and Python dicts before 3.7, and JSON objects generally) is non-deterministic. The test compared the entire artifact including the map, so it passed when iteration happened to produce a consistent order and failed when it did not.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Test asserted `artifact_a == artifact_b` where both contain map fields.
**Why #2:** Map key ordering is non-deterministic in Go (randomized per process for security reasons); two maps with identical key-value pairs may serialize to different string representations.
**Why #3:** The full-struct equality check included the map rather than extracting and comparing individual fields.
**Why #4:** The test passed locally because local runs happened to hit consistent map ordering, while CI parallelism and different OS entropy seeds triggered different orderings.
**Why #5:** Developers used full-struct equality for brevity without auditing which fields are order-sensitive.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Never use equality assertions on structs containing maps — compare individual fields, or sort map keys before comparison | proposed | community | https://github.com/goreleaser/goreleaser/issues/5504 |
| 2 | Use `assert.ElementsMatch` (Go: `assert.Equal` with sorted slices, Python: `sorted()`) when comparing collections where order is not semantically meaningful | proposed | community | — |
| 3 | Add map-containing struct tests to the flaky-test watch list — mark them with explicit sort-before-compare comments | proposed | community | — |

## Key Takeaway

Full-struct equality assertions that include map fields are non-deterministically flaky in CI — maps have no guaranteed iteration order; always compare map contents field-by-field or after explicit key sorting.
