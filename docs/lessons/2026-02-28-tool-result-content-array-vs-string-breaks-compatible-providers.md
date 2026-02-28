# Lesson: Tool Result Content as Array Breaks OpenAI-Compatible Endpoints That Require String Format

**Date:** 2026-02-28
**System:** community (strands-agents/sdk-python)
**Tier:** lesson
**Category:** integration
**Keywords:** tool result, content format, array, string, OpenAI compatible, provider, Kimi, hallucination, tool call, message format, API compatibility
**Source:** https://github.com/strands-agents/sdk-python/issues/1696

---

## Observation (What Happened)

Strands SDK's `OpenAIModel.format_request_tool_message()` sent tool result `content` as an array of content blocks (OpenAI's extended format). OpenAI's official API accepts both array and string formats, but many OpenAI-compatible endpoints (e.g., Kimi K2.5, Moonshot) only correctly process the string format. When the array format was sent, the model silently received the tool call but ignored the tool result content, producing hallucinated values instead of the actual tool output. No error was raised — the call succeeded with wrong behavior.

## Analysis (Root Cause — 5 Whys)

**Why #1:** SDK used the OpenAI extended content-block array format for tool results.
**Why #2:** OpenAI's own API accepts both formats; testing was done against OpenAI directly, not against all "compatible" providers.
**Why #3:** The OpenAI-compatible API surface is not standardized — "compatible" means "accepts the same paths" not "accepts all content formats."
**Why #4:** No cross-provider conformance test verified that the same agent task returned correct results across all supported providers.
**Why #5:** Provider-specific format requirements were not documented in the SDK or surfaced when instantiating with a non-OpenAI endpoint.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Default tool result content to string format (`json.dumps(result)`); offer opt-in array format for providers that need it | proposed | community | https://github.com/strands-agents/sdk-python/issues/1696 |
| 2 | Add a provider conformance test matrix: run a tool-using agent task against each supported provider and assert the tool result is used (not hallucinated) | proposed | community | issue |
| 3 | Document per-provider content format requirements and raise a warning when connecting to a non-OpenAI endpoint | proposed | community | issue |

## Key Takeaway

"OpenAI-compatible" does not mean "accepts all OpenAI content formats" — when a framework supports multiple providers through a shared client, validate tool message formatting against each provider's actual parser, not just the reference implementation; silent behavioral differences (tool result silently ignored, hallucination returned) are far harder to debug than format errors.
