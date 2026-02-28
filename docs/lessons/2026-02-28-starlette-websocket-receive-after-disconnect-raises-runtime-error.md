# Lesson: Starlette WebSocket Raises RuntimeError When receive() Is Called After Disconnect Message
**Date:** 2026-02-28
**System:** community (encode/starlette)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** starlette, websocket, receive, disconnect, RuntimeError, websocket.disconnect, lifecycle, client disconnect
**Source:** https://github.com/encode/starlette/issues/2617
---
## Observation (What Happened)
When a WebSocket client disconnects, Starlette's `WebSocketEndpoint.dispatch()` raises `RuntimeError: Cannot call "receive" once a disconnect message has been received` on the next call to `websocket.receive()`. This happens in any loop that calls `receive()` without checking the message type first — the endpoint tries to receive another message after the disconnect message has already been delivered.

## Analysis (Root Cause — 5 Whys)
Starlette's WebSocket protocol state machine enters a terminal state when a `websocket.disconnect` ASGI message is received. Once in that state, any subsequent call to `websocket.receive()` is illegal — the underlying ASGI transport has no more data. The common pattern of `while True: message = await websocket.receive()` does not check whether the received message was a disconnect before looping back; the next iteration then calls `receive()` on a closed socket, raising the RuntimeError instead of a clean `WebSocketDisconnect` exception.

## Corrective Actions
- Always check the message type after receiving: `if message["type"] == "websocket.disconnect": break` before processing or looping.
- Use `websocket.receive_text()` / `websocket.receive_json()` helpers inside a `try/except WebSocketDisconnect` block rather than the raw `receive()` loop — these helpers handle disconnect detection internally.
- Wrap the entire WebSocket handler in `try/except (WebSocketDisconnect, RuntimeError)` to catch both clean and dirty disconnects without crashing the endpoint.
- For ARIA's hub WebSocket broadcast endpoint: verify the receive loop checks message type on each iteration and catches both `WebSocketDisconnect` and `RuntimeError`.

## Key Takeaway
Starlette WebSocket `receive()` raises `RuntimeError` if called after a disconnect message — always check the message type or use higher-level helpers inside a `try/except WebSocketDisconnect` block.
