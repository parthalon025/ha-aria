# Lesson: Syrupy Snapshot Matcher Is Consumed After First assert — Subsequent Asserts Use Default Serializer

**Date:** 2026-02-28
**System:** community (syrupy-project/syrupy)
**Tier:** lesson
**Category:** testing
**Keywords:** syrupy, snapshot, matcher, path_type, fixture, assert consumed, snapshot testing, serializer
**Source:** https://github.com/syrupy-project/syrupy/issues/800

---

## Observation (What Happened)
A pytest fixture returned `snapshot(matcher=path_type({"field": (str,)}))`. The first `assert {"field": "string"} == my_snapshot` correctly stored the type (`str`) in the snapshot. The second `assert {"field": "string"} == my_snapshot` in the same test stored the raw value (`'string'`), ignoring the matcher entirely. Only the first assertion honored the custom matcher.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Syrupy's `SnapshotAssertion` is stateful — each call to `== snapshot` advances an internal counter and uses `snapshot` as a callable that auto-increments snapshot names (`.0`, `.1`, etc.).
**Why #2:** The `matcher` passed to `snapshot(matcher=...)` is stored on the assertion object for the *initial* call. After the first assertion consumes it, subsequent assertions on the same object use the object's default serializer because the configured-per-call state was not re-applied.
**Why #3:** The fixture creates one `SnapshotAssertion` instance with a fixed matcher, but each `== snapshot` call is a new assertion invocation that internally may not re-read the fixture-level matcher.
**Why #4:** The pattern `snapshot(matcher=...)` creates a new assertion; calling it again in the same test re-uses the same assertion object without re-configuring the matcher.
**Why #5:** This behavior is undocumented — the natural expectation is that a matcher attached to a fixture applies to all assertions using that fixture.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pass the matcher at each assertion call site: `assert obj == snapshot(matcher=path_type(...))` — do not rely on a fixture-level matcher persisting across multiple assertions in one test | proposed | community | issue #800 |
| 2 | If a consistent matcher must be reused, create a helper function or parametrize the matcher in a conftest-level fixture that wraps each `snapshot(matcher=...)` call with an explicit fresh invocation | proposed | community | issue #800 |
| 3 | When writing a syrupy snapshot fixture for reuse, verify that the snapshot file contains the expected serialized form (type vs value) after the first run — mismatch reveals matcher not being applied | proposed | community | issue #800 |

## Key Takeaway
Syrupy's `snapshot(matcher=...)` configures a snapshot assertion object that does not reapply the matcher on each subsequent `==` comparison in the same test — pass the matcher explicitly at every assertion call site, or accept that only the first assert uses it.
