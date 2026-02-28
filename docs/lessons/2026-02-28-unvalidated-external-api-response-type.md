# Lesson: External API Response Type Must Be Validated Before Iteration

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** integration
**Keywords:** API response, type validation, list, dict, TypeError, isinstance, HA, automation sync, external service
**Files:** aria/shared/ha_automation_sync.py:179-184

---

## Observation (What Happened)

`ha_automation_sync.py:_fetch_automations()` calls `await response.json()` and immediately returns the result. The caller does `for auto in raw_automations:` without checking the type first. If HA returns a non-list JSON response (`{"error": "..."}`, empty object, or version-mismatch string — all real possibilities during HA downtime or version upgrade), `TypeError: 'dict' object is not iterable` is raised. The automation cache is never updated, silently.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The response type is not validated before iterating — the code assumes the API always returns a list.

**Why #2:** External APIs are not guaranteed to return the expected type on all code paths — error responses often return a dict or string instead of the expected list.

**Why #3:** The assumption is invisible in the code: `for auto in data:` looks like it handles any iterable, but `dict` is iterable (over keys) in a way that produces wrong results before `TypeError` even fires on non-iterable types.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `isinstance(data, list)` check after `await response.json()`; log ERROR and return `None` if not a list | proposed | Justin | issue #240 |
| 2 | Apply the same pattern to all other external API response parsers that assume a specific type | proposed | Justin | issue #240 |
| 3 | Consider validating against a schema (pydantic or jsonschema) for critical external API responses | proposed | Justin | — |

## Key Takeaway

Always validate the type of external API response data before iterating — `isinstance(data, list)` before `for x in data:` prevents `TypeError` on error responses and ensures the caller gets `None` instead of silent wrong behavior.
