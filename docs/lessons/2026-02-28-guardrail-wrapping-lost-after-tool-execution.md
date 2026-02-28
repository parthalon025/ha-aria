# Lesson: Security Guardrail Wrapping Lost After Tool Execution in Multi-Turn Agent Loop

**Date:** 2026-02-28
**System:** community (strands-agents/sdk-python)
**Tier:** lesson
**Category:** security
**Keywords:** guardrail, security, tool execution, message formatting, Bedrock, guardContent, tool result, multi-turn, agent loop, last message
**Source:** https://github.com/strands-agents/sdk-python/issues/1651

---

## Observation (What Happened)

When `guardrail_latest_message=True` was set on `BedrockModel`, the SDK wrapped the most recent user message in `guardContent` blocks so Bedrock only evaluated that message against guardrail input policies. However, after a tool execution cycle the last message in the conversation was a `toolResult` (role=user) — which lacks a `"text"` key. The SDK's check `if "text" in formatted_content or "image" in formatted_content` failed, so no `guardContent` wrapping was applied to any message. Bedrock then evaluated ALL prior messages against the guardrail, causing spurious false-positive guardrail interventions on tool output content.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `_format_bedrock_messages()` function applied `guardContent` wrapping only to the absolute last message.
**Why #2:** After tool execution the last message is a `toolResult`, not a text message, so the type check fails.
**Why #3:** The code was written to handle the simple single-turn case and assumed the last message is always a user text.
**Why #4:** No test covered the multi-turn tool execution path where security wrapping semantics change.
**Why #5:** The feature was documented at the per-message level, but the invariant "guardrail must wrap the most recent text/image input" was never encoded as a behavioral contract.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When `guardrail_latest_message=True`, scan backwards from the last message to find the most recent message with `"text"` or `"image"` content and wrap that one | proposed | community | https://github.com/strands-agents/sdk-python/issues/1651 |
| 2 | Add integration test: single agent turn with tool execution, verify `guardContent` is present in the second API call | proposed | community | issue |

## Key Takeaway

Security features that apply to "the latest message" must be re-evaluated after each interaction cycle that changes what the latest message is — tool results, system messages, and function outputs all become the "last message" in their respective turns, and security wrappers must track semantic recency (most recent user input), not positional recency (absolute last entry).
