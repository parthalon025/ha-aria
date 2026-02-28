# Lesson: Frontend Omitting a Discriminator Field Silently Redirects Runtime to Wrong Backend Path

**Date:** 2026-02-28
**System:** community (langgenius/dify)
**Tier:** lesson
**Category:** integration
**Keywords:** frontend serialization, discriminator field, type field, default fallback, plugin routing, MCP, agent node, silent failure, PluginNotFoundError
**Source:** https://github.com/langgenius/dify/issues/32204

---

## Observation (What Happened)

In Dify's Agent node, when users added MCP tools via the GUI the frontend serializer did not write the `type` field into the saved tool configuration. At runtime `agent_node.py` used `.get("type", ToolProviderType.BUILT_IN)` to resolve the provider. The missing `type` field caused silent fallback to `BUILT_IN`, which routed to `get_builtin_provider` instead of `get_mcp_provider_controller`, raising `PluginNotFoundError`. The same MCP tool worked correctly as a standalone Tool node because that node's serializer did write `type: mcp`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The frontend Agent node component did not include `type: mcp` when serializing tool entries for MCP tools.
**Why #2:** The backend API used a permissive default fallback (`ToolProviderType.BUILT_IN`) rather than failing loudly on a missing required discriminator.
**Why #3:** The Standalone Tool node and Agent node were implemented separately; only the standalone node's serializer was updated when MCP support was added.
**Why #4:** No contract test or schema validation existed between frontend serialized output and backend expected input for tool entries.
**Why #5:** The default fallback pattern hid the bug in development (if built-in plugins exist with similar names) until the specific combination of MCP tool in Agent node was tested end-to-end.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Fix the frontend Agent node serializer to include `type` field when writing MCP tool entries | proposed | community | https://github.com/langgenius/dify/issues/32204 |
| 2 | Replace silent default fallback with explicit error: if `type` is missing from tool config, raise `ValueError` with field name and allowed values | proposed | community | issue |
| 3 | Add a contract test that serializes an MCP tool in the Agent node and asserts `"type": "mcp"` is present in output | proposed | community | issue |

## Key Takeaway

When a discriminator field routes execution to different provider implementations, its absence must be a hard error — not a silent default fallback; missing required discriminators should fail at config load time, not at runtime when the wrong provider silently fails.
