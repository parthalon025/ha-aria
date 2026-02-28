# Lesson: Undocumented Default Limit Silently Truncates Agent Tool Lists

**Date:** 2026-02-28
**System:** community (ComposioHQ/composio)
**Tier:** lesson
**Category:** integration
**Keywords:** tool list, limit, pagination, default, undocumented, silent truncation, agent, composio, SDK, API
**Source:** https://github.com/ComposioHQ/composio/issues/2138

---

## Observation (What Happened)

Composio's Python SDK `composio.tools.get()` silently applied an undocumented default limit when no `limit` parameter was passed. Users who expected all registered tools to be returned received a truncated list without any warning or indication that results were incomplete. Agent systems built on top silently had fewer tools available than expected, causing capabilities to be missing without any error.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `composio.tools.get()` had an internal default `limit` that was never surfaced in the function signature or docstring.
**Why #2:** The default was inherited from the underlying API pagination behavior and not re-exposed in the SDK wrapper.
**Why #3:** No warning was emitted when the number of returned tools equaled the limit (indicating potential truncation).
**Why #4:** The function name `get` implied "get all matching tools" to callers, not "get at most N tools".
**Why #5:** API pagination defaults were treated as an implementation detail rather than a behavioral contract that callers depend on.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Expose the default limit in the function signature: `def get(limit: int = DEFAULT_LIMIT)` and document it | proposed | community | https://github.com/ComposioHQ/composio/issues/2138 |
| 2 | Emit a warning when the returned count equals the limit: "Results may be truncated (received {limit} items). Pass a higher limit to retrieve more." | proposed | community | issue |
| 3 | Provide a `get_all()` method or `paginate=True` option that automatically fetches all pages | proposed | community | issue |

## Key Takeaway

Any API that applies a default pagination limit must expose that limit as a documented parameter, emit a warning when results may be truncated, and provide an explicit "get all" path — silent truncation of agent tool lists removes capabilities without any observable signal.
