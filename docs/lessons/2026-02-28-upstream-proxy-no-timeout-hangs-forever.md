# Lesson: Proxying to Upstream Services Without a Timeout Hangs the Entire Hub

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** timeout, aiohttp, proxy, Frigate, upstream, connection pool, hang, availability
**Files:** aria/hub/api.py (GET /api/presence/thumbnail)

---

## Observation (What Happened)

The Frigate thumbnail proxy at `GET /api/presence/thumbnail` forwards requests to Frigate with no timeout. If Frigate is unavailable (network hiccup, Frigate crash, restart), the request hangs indefinitely, holding a connection pool slot. Under concurrent thumbnail requests, the entire hub becomes unresponsive.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The proxy route makes an outbound HTTP call with no `timeout=` argument.

**Why #2:** Default `aiohttp.ClientTimeout` is no timeout — the connection waits forever unless explicitly bounded.

**Why #3:** Proxy routes are often written to "just forward the request" without considering the upstream availability contract; the caller assumes a fast response but gets a hang.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `timeout=aiohttp.ClientTimeout(total=5.0)` to the Frigate proxy call; return 504 Gateway Timeout on `asyncio.TimeoutError` | proposed | Justin | issue #296 |
| 2 | Enforce a policy: every outbound HTTP call in async hub code must have an explicit `timeout` argument | proposed | Justin | issue #296 |
| 3 | Apply the same fix to all other outbound calls in `presence.py`, `unifi.py`, `orchestrator.py` that may not have timeout guards | proposed | Justin | — |

## Key Takeaway

Every outbound HTTP call from an async service must have an explicit timeout — the default is no timeout, and one unavailable upstream service will hang the entire connection pool without it.
