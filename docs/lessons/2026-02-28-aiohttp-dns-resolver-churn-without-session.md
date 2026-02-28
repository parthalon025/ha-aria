# Lesson: aiohttp Creates a New DNSResolver Per Request When No Session Is Reused
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** performance
**Keywords:** aiohttp, DNSResolver, c-ares, ClientSession, connection pool, resource churn, DNS, resolver
**Source:** https://github.com/aio-libs/aiohttp/issues/10847
---
## Observation (What Happened)
When `aiohttp` requests are made without a reused `ClientSession` (e.g., using the convenience `aiohttp.get()` shortcut or creating per-request sessions), a new `DNSResolver` object is allocated for every request. This churns c-ares resolver channels — objects explicitly designed to be shared across unlimited queries — instead of reusing a single shared instance.

## Analysis (Root Cause — 5 Whys)
The `DNSResolver` is owned by the `TCPConnector`, which is owned by the `ClientSession`. When a new session is created per request, a new connector is created, which spawns a new c-ares channel. c-ares's own documentation states "a single channel can accept unlimited queries" — the intention is always to have one resolver per process or long-lived context. Per-request session creation defeats this by spawning fresh resolver channels that are immediately torn down, adding DNS initialization overhead and file descriptor pressure on every request.

## Corrective Actions
- Always reuse a single `aiohttp.ClientSession` for the lifetime of a service. The `ClientSession` → `TCPConnector` → `DNSResolver` chain is designed for reuse, not instantiation per call.
- If a service makes high-frequency outbound requests (e.g., polling HA REST API, Telegram sends), measure DNS resolution latency — per-request sessions will show measurable overhead on each call.
- In integration tests, inject the shared session explicitly rather than allowing test helpers to create new sessions.

## Key Takeaway
aiohttp's DNSResolver is tied to the `ClientSession` lifecycle — per-request session creation wastes a c-ares resolver channel on every call; use a single shared session.
