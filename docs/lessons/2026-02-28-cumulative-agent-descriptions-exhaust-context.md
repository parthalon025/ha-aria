# Lesson: Cumulative Agent/Tool Descriptions Exhaust Context Window Before First Task

**Date:** 2026-02-28
**System:** community (wshobson/agents)
**Tier:** lesson
**Category:** performance
**Keywords:** context window, agent descriptions, token budget, tool definitions, cumulative, Claude Code, plugin, performance, context exhaustion
**Source:** https://github.com/wshobson/agents/issues/93

---

## Observation (What Happened)

Installing all available plugins in the `wshobson/agents` Claude Code marketplace caused Claude Code to warn: "Large cumulative agent descriptions will impact performance (~404.1k tokens > 15.0k)". With all plugins installed, the agent description context alone exceeded the practical working limit, leaving almost no tokens for actual task content. Any command triggered "Context low · Run /compact to compact & continue" before meaningful work could begin.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Each agent/plugin contributes its full description, system prompt, and tool definitions to every context window.
**Why #2:** Descriptions were authored individually without considering aggregate token cost across all installed plugins.
**Why #3:** No per-agent token budget cap or lazy-loading strategy was implemented; all descriptions are injected unconditionally at session start.
**Why #4:** The marketplace design assumed users would install a small subset of plugins, not all of them simultaneously.
**Why #5:** No CI gate or install-time warning validated that the total installed token footprint stayed within a safe threshold.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Define a per-plugin token budget (e.g., max 500 tokens for description + tool list) and lint against it | proposed | community | https://github.com/wshobson/agents/issues/93 |
| 2 | Implement lazy loading: only inject agent descriptions for agents that are actively invoked, not all installed agents | proposed | community | issue |
| 3 | Add install-time validation that the total token footprint of selected plugins stays within a configurable threshold | proposed | community | issue |

## Key Takeaway

In multi-agent systems each agent's description consumes tokens from a shared finite context window — always budget the aggregate token cost of all active agent definitions, not just the per-agent cost in isolation, and enforce a hard ceiling at plugin install or session init.
