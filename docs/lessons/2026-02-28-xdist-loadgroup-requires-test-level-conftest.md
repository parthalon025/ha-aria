# Lesson: xdist_group Markers Added in conftest.py pytest_collection_modifyitems Are Ignored When conftest Is Above Test Level

**Date:** 2026-02-28
**System:** community (pytest-dev/pytest-xdist)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, xdist, loadgroup, xdist_group, conftest, marker, collection, worker assignment, parallelism
**Source:** https://github.com/pytest-dev/pytest-xdist/issues/1253

---

## Observation (What Happened)
A `conftest.py` at `tests/integration/` level used `pytest_collection_modifyitems` to add `pytest.mark.xdist_group(name=...)` to all items. Running with `pytest -n 4` showed tests were not grouped as expected — tests that should have been serialized on one worker ran across multiple workers. The markers appeared to not be applied.

## Analysis (Root Cause — 5 Whys)
**Why #1:** xdist distributes collected items to workers before the full `pytest_collection_modifyitems` hook chain completes when the conftest is not at the test-item's directory level.
**Why #2:** The `conftest.py` at a parent directory level may be processed after xdist has already serialized items for distribution.
**Why #3:** `xdist_group` marker assignment requires the marker to be present on the `Item` object at the point when xdist's scheduler reads it — adding it in a late `modifyitems` hook in a parent conftest is a race condition.
**Why #4:** The `pytest_collection_modifyitems` hook in a higher-level conftest runs after lower-level conftest hooks, and xdist's scheduler may read markers during collection finalization.
**Why #5:** This is a hook ordering issue specific to xdist's early scheduler initialization.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Place the `pytest_collection_modifyitems` hook that assigns `xdist_group` markers in a `conftest.py` at the same directory level as the test files, not in a parent directory conftest | proposed | community | issue #1253 |
| 2 | Alternatively, set xdist_group markers directly on test functions/classes using `@pytest.mark.xdist_group("group_name")` in the test file itself to ensure markers are present at collection time | proposed | community | issue #1253 |
| 3 | Verify grouping is working by running with `--collect-only` and checking that items show the expected `xdist_group` mark before relying on parallelism for correctness | proposed | community | issue #1253 |

## Key Takeaway
`xdist_group` markers assigned via `pytest_collection_modifyitems` in a parent-directory conftest.py may be silently ignored by xdist's scheduler — place the hook at the same directory level as the tests or use decorators directly on test items to guarantee markers are read before worker assignment.
