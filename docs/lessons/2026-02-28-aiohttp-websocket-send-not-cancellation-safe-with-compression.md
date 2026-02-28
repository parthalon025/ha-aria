# Lesson: aiohttp WebSocket send_str Is Not Cancellation-Safe When Compression Is Enabled
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** async
**Keywords:** aiohttp, websocket, compression, cancellation, asyncio.cancel, send_str, executor, data corruption
**Source:** https://github.com/aio-libs/aiohttp/issues/11725
---
## Observation (What Happened)
When aiohttp WebSocket compression is enabled and `asyncio.cancel()` interrupts a `send_str()` call while it is waiting for the compression executor to complete, the internal compression buffer is left in a corrupted intermediate state. The next `send_str()` call flushes both the residual bytes from the cancelled send and the new data, causing the receiver to see two messages concatenated into one receive event.

## Analysis (Root Cause — 5 Whys)
aiohttp offloads compression work to a thread executor to avoid blocking the event loop. If the awaiting coroutine is cancelled at the executor boundary, the exception surfaces in the calling coroutine but the executor-side compression state is not rolled back — it has partially consumed the data. The next send reads from the same stateful compressor object, which still holds the previous partial data, producing merged output. There is no `clear()` or `reset()` operation exposed on the compression buffer.

## Corrective Actions
- Use `asyncio.shield(ws.send_str(data))` for any WebSocket send that may be executed inside a task that can be cancelled, or guard the entire send with a dedicated cancellation-safe wrapper.
- If cancellation of in-flight sends cannot be prevented, reconnect the WebSocket rather than reusing it after a cancellation — the compression state is unrecoverable without reconnection.
- Add a comment at every `ws.send_*` call site documenting whether the enclosing task is cancellable; mark as `# cancellation-unsafe` if compression is on and shielding is not applied.
- For ARIA's HA WebSocket client: if reconnect logic can fire `cancel()` on an active send, switch to `asyncio.shield()` on the send or set `compress=False` on the connection.

## Key Takeaway
aiohttp WebSocket `send_str()` with compression is not cancellation-safe — cancelling the task mid-send corrupts the compressor state and causes data bleed into the next message.
