# Lesson: aiohttp Silently Retries Requests Once on ServerDisconnectedError With No Way to Disable
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** reliability
**Keywords:** aiohttp, retry, ServerDisconnectedError, ClientOSError, idempotent, ClientSession, request, silent retry
**Source:** https://github.com/aio-libs/aiohttp/issues/10790
---
## Observation (What Happened)
`aiohttp.ClientSession._request()` contains hardcoded logic to retry the request once when a `ClientOSError` or `ServerDisconnectedError` is raised on a persistent connection. This retry is unconditional — it fires for any HTTP method including non-idempotent ones (POST, PATCH, DELETE) — and there is no API to opt out of it. Applications relying on exactly-once delivery semantics can silently perform duplicate operations.

## Analysis (Root Cause — 5 Whys)
The retry was added as a convenience to handle server-side keepalive connection expiry, which is a common scenario where the server closes an idle connection just as the client reuses it. While useful for simple GET requests, the code does not check HTTP method idempotency before retrying. The lack of a disable flag means consuming code has no way to enforce at-most-once semantics without wrapping all requests in workarounds.

## Corrective Actions
- For any non-idempotent aiohttp request (POST, PATCH, DELETE) where duplicate execution would cause data corruption, wrap the call and implement idempotency keys or deduplication at the server level — do not assume aiohttp will not retry.
- When debugging unexpected duplicate writes against HA or other APIs, check whether aiohttp's built-in retry is the cause before looking for application-level double invocations.
- Document the retry behavior in any service that uses aiohttp for POST/PATCH mutations, noting the upstream limitation.
- Track this as a risk in ARIA's `aria/hub/` outbound HTTP calls — particularly any write paths (e.g., calling HA services via REST API POST).

## Key Takeaway
aiohttp silently retries any request once on `ServerDisconnectedError` regardless of HTTP method — non-idempotent POST/PATCH/DELETE calls can be duplicated without any application-level indication.
