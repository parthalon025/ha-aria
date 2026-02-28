# Lesson: Async Tool Deadlock When Sync Wrapper Creates New Event Loop

**Date:** 2026-02-28
**System:** community (ag2ai/ag2)
**Tier:** lesson
**Category:** async
**Keywords:** async, deadlock, event loop, asyncio, thread, MCP, tool execution, multi-agent, run_async_in_thread
**Source:** https://github.com/ag2ai/ag2/issues/2144

---

## Observation (What Happened)

When `DefaultPattern` in AG2 executed async MCP tools, tool execution hung indefinitely. The synchronous group tool executor used `_run_async_in_thread()` which created a new event loop in a separate thread. The async MCP `ClientSession` object was bound to the original event loop, so cross-loop communication deadlocked.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `_Group_Tool_Executor._generate_group_tool_reply()` is synchronous but needs to call async tools.
**Why #2:** It calls `_run_async_in_thread()` as a bridge, which spins up a new event loop in a background thread.
**Why #3:** The async tool (MCP `ClientSession`) holds state — WebSocket connections, coroutine chains — attached to the original event loop.
**Why #4:** Calls from the new thread's event loop into the original loop's objects create cross-loop awaits that never resolve.
**Why #5:** The executor was designed before MCP async tools were added; no audit of session affinity was performed when wiring them together.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Make the group tool executor async or use `asyncio.run_coroutine_threadsafe(coro, original_loop)` to dispatch back to the owning loop | proposed | community | https://github.com/ag2ai/ag2/issues/2144 |
| 2 | Document which async objects are loop-affine (ClientSession, WebSocket) so callers know they cannot be handed to a different event loop | proposed | community | issue |

## Key Takeaway

Objects created on one asyncio event loop (WebSocket sessions, HTTP clients, MCP ClientSessions) cannot be awaited from a different loop — a sync wrapper that creates a new event loop will deadlock any tool that holds loop-affine state; the fix is to dispatch back to the original loop with `run_coroutine_threadsafe`, not to create a new loop.
