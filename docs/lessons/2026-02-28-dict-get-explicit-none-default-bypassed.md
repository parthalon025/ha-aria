# Lesson: dict.get(key, default) Does Not Cover Explicit None Values

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** dict, get, None, default, cache, cold-start, explicit null, or-idiom
**Files:** aria/hub/api.py:444, aria/modules/intelligence.py

---

## Observation (What Happened)

`GET /api/ml/pipeline` returns 500 on cold start because `intel.get("feature_selection", {})` returns `None` when the cache explicitly stores `"feature_selection": None`. The caller then calls `.get()` on `None`, raising `AttributeError`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `dict.get(key, default)` only uses the default when the key is **absent**. If the key is present but its value is `None`, `.get()` returns `None` — the default is ignored.

**Why #2:** The intelligence cache stores keys with explicit `None` during cold start, before ML data is populated. The API assumes `{}` as the default but the key exists.

**Why #3:** The distinction between "key absent" and "key present with value None" is collapsed by `.get(key, {})`, making them look the same but behave differently — the pattern looks correct but silently fails on explicit nulls.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace `x.get(key, {})` with `x.get(key) or {}` wherever the stored value could be explicit `None` | proposed | Justin | issue #334, api.py:444 |
| 2 | Audit all `intel.get(...)` calls in api.py for the same pattern | proposed | Justin | issue #316 |

## Key Takeaway

`dict.get(key, default)` does not protect against explicit `None` — use `dict.get(key) or default` when None is a possible stored value.
