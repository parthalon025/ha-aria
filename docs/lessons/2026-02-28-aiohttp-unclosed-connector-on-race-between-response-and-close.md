# Lesson: aiohttp "Unclosed Connector" Warning Fires on a Race Between Keepalive Response and Session Close
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** aiohttp, ClientSession, TCPConnector, unclosed connector, keepalive, context manager, resource leak, warning
**Source:** https://github.com/aio-libs/aiohttp/issues/11221
---
## Observation (What Happened)
Even when using `async with aiohttp.ClientSession() as session:` correctly, intermittent "Unclosed connector" warnings appear in Sentry logs. The warning fires when the server returns a keepalive connection that is still alive in the connector's pool at the moment the session context manager exits, and the connector has not yet had a chance to drain/close those keepalive connections before the garbage collector sees the unclosed transport.

## Analysis (Root Cause — 5 Whys)
aiohttp's `TCPConnector` maintains a pool of keepalive connections. When `ClientSession.close()` is called (via context manager exit), it schedules connector closure but does not await the full drain of all keepalive transports before returning. If an event loop iteration is not given to process the closure, the transports report as unclosed. This is a known race in aiohttp's shutdown path — the connector needs a brief event loop pass to fully close all idle connections. The real fix is to use a long-lived session, but even correct per-request sessions can trigger this intermittently.

## Corrective Actions
- Use a single long-lived `aiohttp.ClientSession` for the lifetime of the application, not per-request sessions. This eliminates the teardown race entirely.
- If a short-lived session is unavoidable, call `await asyncio.sleep(0)` or `await session.connector.close()` explicitly after the context manager exits to allow the event loop to drain remaining transports.
- In tests, call `await asyncio.sleep(0.1)` after session close to allow connector cleanup before assertions.
- For ARIA's HA polling client: confirm the session is stored at module or class level and reused across all polling cycles — never instantiated per-call.

## Key Takeaway
Never create an `aiohttp.ClientSession` per request — even syntactically correct context-manager usage causes "Unclosed connector" warnings due to a keepalive teardown race.
