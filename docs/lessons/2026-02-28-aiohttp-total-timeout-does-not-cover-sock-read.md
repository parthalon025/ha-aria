# Lesson: aiohttp ClientTimeout.total Does Not Prevent Indefinite epoll_wait During Response Read
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** reliability
**Keywords:** aiohttp, ClientTimeout, sock_read, timeout, epoll_wait, hang, connection, HTTP client
**Source:** https://github.com/aio-libs/aiohttp/issues/11740
---
## Observation (What Happened)
aiohttp's `ClientTimeout(total=5*60)` default does not prevent an HTTP request from hanging forever once the TCP connection is established and the request has been sent. After the request headers are sent, `epoll_wait` is called with a timeout of -1 (indefinitely), waiting for the server response. If the server stalls or connectivity is lost, the request never returns despite the `total` timeout being set.

## Analysis (Root Cause — 5 Whys)
`ClientTimeout.total` controls the overall wall-clock limit for the entire request lifecycle but does not propagate to the per-socket read wait — that is controlled separately by `ClientTimeout.sock_read`. When `sock_read=None` (the default), the underlying `epoll_wait` syscall receives no timeout argument and blocks forever. The `total` timer is only checked at higher-level coroutine suspension points, not the raw socket wait. This means a TCP connection that is open but unresponsive (no data, no RST) bypasses the total timeout entirely.

## Corrective Actions
- Always set both `total` and `sock_read` in `ClientTimeout` for any aiohttp session making outbound requests: `ClientTimeout(total=30, sock_read=10)`.
- For the ARIA hub's HA WebSocket and HTTP polling calls, audit every `aiohttp.ClientSession` construction — verify `sock_read` is explicitly set, not relying on `total` alone.
- Add a test that verifies an aiohttp request times out within `sock_read + slack` seconds when the server sends the TCP handshake but never sends response data (mock with a server that accepts then goes silent).

## Key Takeaway
`ClientTimeout.total` alone does not prevent aiohttp from hanging forever — always pair it with an explicit `sock_read` value.
