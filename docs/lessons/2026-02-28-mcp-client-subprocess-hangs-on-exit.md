# Lesson: MCP Client Subprocess Hangs on Program Exit Without Explicit Cleanup

**Date:** 2026-02-28
**System:** community (strands-agents/sdk-python)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** MCP, subprocess, hang, exit, background thread, cleanup, context manager, lifecycle, stdio_client, asyncio, join
**Source:** https://github.com/strands-agents/sdk-python/issues/1732

---

## Observation (What Happened)

When using Strands SDK's managed `MCPClient` with a `stdio_client` subprocess, the Python program hung on exit even after printing "DONE". The MCP client launched a background thread for the subprocess I/O; when the program's main function returned, the background thread had not been joined. The program printed completion but the process never terminated. Explicit `mcp_client.stop()` or using it as a context manager prevented the hang.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `MCPClient` background thread was not a daemon thread (or was blocked on a subprocess pipe).
**Why #2:** Python's interpreter waits for all non-daemon threads to finish before exiting.
**Why #3:** The background thread was waiting on the subprocess stdio pipe which was still open.
**Why #4:** The `MCPClient` was not used as a context manager, so `__exit__` / `stop()` was never called.
**Why #5:** The API design allowed constructing an `MCPClient` outside a `with` block without any warning that cleanup was the caller's responsibility.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `MCPClient` as a context manager (`async with`) to guarantee `stop()` is called | proposed | community | https://github.com/strands-agents/sdk-python/issues/1732 |
| 2 | Make the background I/O thread a daemon thread so it does not block interpreter shutdown | proposed | community | issue |
| 3 | In `MCPClient.__del__`, call `stop()` and log a warning if the client was not explicitly stopped | proposed | community | issue |

## Key Takeaway

Any object that owns a background thread communicating with a subprocess must be used as a context manager or have an explicit `stop()` call — non-daemon threads block Python interpreter exit indefinitely; document the cleanup requirement at construction time, not only in the teardown method.
