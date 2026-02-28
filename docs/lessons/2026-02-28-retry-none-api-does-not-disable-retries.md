# Lesson: retry=None Silently Applies Default Retry Behavior Instead of Disabling Retries

**Date:** 2026-02-28
**System:** community (strands-agents/sdk-python)
**Tier:** lesson
**Category:** configuration
**Keywords:** retry, None, disable, API design, LLM, agent, retry strategy, configuration, surprising default, DevX, backoff
**Source:** https://github.com/strands-agents/sdk-python/issues/1579

---

## Observation (What Happened)

In the Strands SDK, passing `retry_strategy=None` to an Agent did not disable retries — it silently applied the same default retry behavior as if no argument was passed. To actually disable retries, users had to pass `retry_strategy=ModelRetryStrategy(max_attempts=1)`. This counterintuitive API caused users who expected `None` to mean "no retries" to experience unexpected repeated API calls and throttling behavior.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The code path for `retry_strategy=None` fell through to a default initialization of `ModelRetryStrategy`, rather than disabling the retry loop.
**Why #2:** The function signature used `retry_strategy=None` as a sentinel for "not provided, use default" — the same value a user would pass to explicitly say "disable retries".
**Why #3:** Python convention for "use default" is `None`, but the API also needed `None` to mean "disable" — these two semantics were conflated.
**Why #4:** No documentation or type annotation called out that `None` ≠ "no retries"; the only correct path was a non-obvious `max_attempts=1`.
**Why #5:** The design was recognized as backwards-incompatible to fix in v1; the corrected behavior was scheduled for v2.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use a sentinel object (e.g., `_UNSET = object()`) to distinguish "not provided" from `None`; let `None` explicitly mean "no retries" | proposed | community | https://github.com/strands-agents/sdk-python/issues/1579 |
| 2 | In the interim, document with a type: `retry_strategy: ModelRetryStrategy | Literal[False] | None` and process `False` as "disable" | proposed | community | issue |

## Key Takeaway

When `None` serves as a "use default" sentinel in a function signature, it cannot simultaneously mean "disable this feature" — use a distinct sentinel object or an explicit boolean `disabled=True` parameter, and document the difference clearly; conflating "not provided" with "explicitly disabled" creates invisible behavior for callers who follow Python intuition.
