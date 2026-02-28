# Lesson: aiohttp ClientTimeout(total=0) Raises ValueError on TLS Connections — Zero Is Not "No Timeout"
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** configuration
**Keywords:** aiohttp, ClientTimeout, total=0, TLS, ssl_handshake_timeout, ValueError, zero timeout, configuration semantics
**Source:** https://github.com/aio-libs/aiohttp/issues/11859
---
## Observation (What Happened)
Passing `ClientTimeout(total=0)` when connecting to a TLS endpoint raises `ValueError: ssl_handshake_timeout should be a positive number, got 0`. aiohttp forwards `timeout.total` to asyncio's `start_tls()` call, which treats zero as an invalid value — asyncio requires a positive number or `None` (for default timeout). This conflicts with a natural interpretation of `total=0` as "no timeout" (as `total=None` actually means in aiohttp).

## Analysis (Root Cause — 5 Whys)
aiohttp's `ClientTimeout` uses `None` to mean "no limit" and positive numbers for finite timeouts. However, when aiohttp passes the timeout to asyncio's internal TLS machinery, zero is not a valid sentinel — asyncio uses `None` for the default and requires positive values. The mapping between aiohttp's `total=0` and asyncio's requirement is not handled by aiohttp, so the raw zero value is passed directly, crashing the TLS handshake. This is a semantic inconsistency: `total=0` behaves differently for plain HTTP (where it may mean "immediate timeout") vs TLS (where it raises).

## Corrective Actions
- Never use `ClientTimeout(total=0)` — use `ClientTimeout(total=None)` for unlimited or `ClientTimeout(total=N)` for positive N seconds.
- When dynamically constructing `ClientTimeout` from user config, add a guard: `timeout_val = user_val or None` to convert falsy values to `None` before passing to `ClientTimeout`.
- Add a unit test that constructs `ClientTimeout` from a config that could be zero and verify it normalizes to `None` before passing to the session.

## Key Takeaway
`ClientTimeout(total=0)` raises `ValueError` on TLS connections because asyncio rejects zero as an SSL handshake timeout — always use `None` for "no timeout" in `ClientTimeout`, never `0`.
