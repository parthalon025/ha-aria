# Lesson: Thinking Model Context Dropped After MCP Tool Call — Model Loses Its Plan

**Date:** 2026-02-28
**System:** community (ggozad/oterm)
**Tier:** lesson
**Category:** integration
**Keywords:** ollama, thinking, MCP, tool-call, context, reasoning-trace, multi-step, chain-of-thought, context-window, persistence
**Source:** https://github.com/ggozad/oterm/issues/254

---

## Observation (What Happened)

When using a thinking-enabled model (Qwen3) with MCP tools, the model's in-progress reasoning (chain-of-thought) was dropped from context after the first tool call. The model appeared to lose its multi-step plan and re-planned from scratch after each tool result, making complex multi-tool tasks unreliable or impossible.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The model re-planned from scratch after each tool call instead of continuing its chain of thought.
**Why #2:** The thinking/reasoning trace was not included in the messages sent back to the model after the tool result was injected.
**Why #3:** The implementation explicitly excluded thinking traces from persistence "to save context" — a deliberate optimization that was not disclosed to users as a behavioral limitation.
**Why #4:** The assumption was that thinking is ephemeral; in practice, for multi-step tool use, the plan formed during thinking is the critical state that must survive round-trips.
**Why #5:** No distinction was made between "thinking during a single response" (safe to drop) and "thinking that forms a multi-step plan" (must be retained across tool calls).

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When a tool call interrupts in-progress thinking, include the thinking trace in the message context sent with the tool result | proposed | community | issue #254 |
| 2 | Separate "session persistence" (where dropping thinking is fine) from "in-flight multi-step context" (where dropping thinking breaks the agent) | proposed | community | issue #254 |
| 3 | Document explicitly: thinking traces are dropped after tool calls — users relying on multi-step tool use should be aware | proposed | community | documentation |
| 4 | Add a "retain thinking" mode that includes the reasoning trace in tool call context | proposed | community | feature request |

## Key Takeaway

Dropping reasoning/thinking traces from context is safe for single-response generation but catastrophically breaks multi-step agentic tool use where the thinking trace contains the multi-step plan — treat reasoning context as required state during any tool-call round-trip.
