# Lesson: Context Window Exception Detection Via String Matching Breaks on Provider Variants

**Date:** 2026-02-28
**System:** community (strands-agents/sdk-python)
**Tier:** lesson
**Category:** error-handling
**Keywords:** context window overflow, error message, string matching, provider, Bedrock, Anthropic, ContextWindowOverflowException, exception handling, LLM, whitelist
**Source:** https://github.com/strands-agents/sdk-python/issues/1712

---

## Observation (What Happened)

Strands SDK's Bedrock model detected context window overflows by checking if the error message contained one of three hardcoded strings. When using Anthropic models via Bedrock, the actual error message was `"prompt is too long: 203470 tokens > 200000 maximum"` — a format not in the whitelist. The `ContextWindowOverflowException` was never raised, the conversation manager's `reduce_context()` was never called, and the agent crashed instead of recovering.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES` only contained three string patterns from specific model families.
**Why #2:** Anthropic-on-Bedrock uses a different error format (`"prompt is too long: N tokens > M maximum"`) than direct Bedrock models.
**Why #3:** The whitelist was populated from observed error messages at implementation time, not from provider documentation.
**Why #4:** New model providers / routing paths were added without auditing which error formats they produced.
**Why #5:** No test covered the case of an unrecognized overflow message format — the whitelist grew stale silently.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `"prompt is too long"` pattern to `BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES` | proposed | community | https://github.com/strands-agents/sdk-python/issues/1712 |
| 2 | Prefer regex or token-count numeric detection over exact string matching; log unmatched overflow errors with their full text | proposed | community | issue |
| 3 | Add a test that fires a mock `ValidationException` with each known provider format and asserts `ContextWindowOverflowException` is raised | proposed | community | issue |

## Key Takeaway

Detecting provider errors by checking for specific substrings in error messages is a fragile contract — every new model provider or API version adds new message formats; use numeric checks (token counts), error codes, or catch broader exception types and re-classify, rather than maintaining an error-string whitelist.
