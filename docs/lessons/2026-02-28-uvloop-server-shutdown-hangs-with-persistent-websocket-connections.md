# Lesson: uvloop Server Shutdown Hangs When Persistent WebSocket or Streaming Connections Remain Open
**Date:** 2026-02-28
**System:** community (MagicStack/uvloop)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** uvloop, shutdown, persistent connection, websocket, streaming, asyncio.Server.close, graceful shutdown, connection drain
**Source:** https://github.com/MagicStack/uvloop/issues/180
---
## Observation (What Happened)
A uvloop-based aiohttp server blocks indefinitely on shutdown when any WebSocket or streaming HTTP connection is still open. The server calls `asyncio.Server.close()` but never completes because uvloop keeps the server alive until all existing persistent connections are fully closed — contrary to the asyncio spec, which states that `close()` stops accepting new connections but leaves existing connections open for the caller to drain.

## Analysis (Root Cause — 5 Whys)
uvloop's `Server.close()` implementation (in the affected version) waited for all active connections to finish before completing, deviating from the CPython asyncio specification. The asyncio spec explicitly says existing connections are "left open" after `close()` — it is the caller's responsibility to drain them. uvloop's deviation meant that graceful shutdown patterns relying on `server.close()` followed by independent connection draining would hang forever if any connection (WebSocket, SSE, long-poll) remained open. The fix was to conform to the asyncio spec and allow `close()` to return immediately.

## Corrective Actions
- Implement explicit connection drain before calling server close: track all active WebSocket connections in a registry, send close frames to each, and await their termination with a timeout before calling `server.close()`.
- Never rely on the server's `close()` to drain active connections — implement graceful shutdown at the application layer with a configurable drain timeout (e.g., 5 seconds).
- For ARIA's hub: the `IntelligenceHub.shutdown()` method should close all active WebSocket client connections and wait for their disconnect acknowledgment before stopping the uvicorn server process.
- Set a `--timeout-graceful-shutdown` flag in uvicorn invocation to enforce a maximum drain window and prevent indefinite hang on deployments.

## Key Takeaway
Never rely on the ASGI server's shutdown to drain persistent connections — implement explicit WebSocket close + drain with a timeout in the application's own shutdown handler before stopping the server.
