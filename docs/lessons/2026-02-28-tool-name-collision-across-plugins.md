# Lesson: Tool Name Collisions Across Multiple Plugins Cause API 400 Errors

**Date:** 2026-02-28
**System:** community (wshobson/agents)
**Tier:** lesson
**Category:** integration
**Keywords:** tool name, duplicate, collision, API 400, plugin, Claude, tool uniqueness, namespace, multi-agent, invalid_request_error
**Source:** https://github.com/wshobson/agents/issues/111

---

## Observation (What Happened)

After installing many plugins from the `wshobson/agents` marketplace, Claude Code returned `API Error: 400 {"type":"error","error":{"type":"invalid_request_error","message":"tools: Tool names must be unique."}}`. Multiple plugins registered tool definitions with identical names (e.g., a generic `code-reviewer` tool), and when they were all submitted in the same API call, the Anthropic API rejected the request.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Multiple plugins registered tools with the same name (e.g., `code-reviewer`).
**Why #2:** Plugin authors did not namespace their tool names — each author named tools by function without a plugin prefix.
**Why #3:** No uniqueness check existed at plugin registration or at API request construction time.
**Why #4:** The marketplace did not enforce naming conventions (e.g., `<plugin-slug>/<tool-name>`) in plugin submissions.
**Why #5:** The LLM API's requirement for globally unique tool names within a single request was not documented in the plugin authoring guide.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Namespace all tool names with the plugin slug: `<plugin-slug>_<tool-name>` (e.g., `code-reviewer_review` not `review`) | proposed | community | https://github.com/wshobson/agents/issues/111 |
| 2 | Add a CI check that scans `marketplace.json` for duplicate tool names across all plugins | proposed | community | issue |
| 3 | Detect and deduplicate at request construction time; log a warning with the conflicting plugin names | proposed | community | issue |

## Key Takeaway

Tool names submitted together in a single LLM API call must be globally unique within that call — when loading tools from multiple plugins or agents, enforce a namespace convention at authoring time (not just at runtime) to prevent hard API failures when plugin counts grow.
