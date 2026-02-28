# Lesson: Token Expiry Validated Client-Side Only — Expired Tokens Accepted by Backend Until Explicit Logout

**Date:** 2026-02-28
**System:** community (fastapi/full-stack-fastapi-template)
**Tier:** lesson
**Category:** security
**Keywords:** JWT, token expiry, exp claim, client-side check, server-side validation, localStorage, 401 vs 403, interceptor, broken authentication
**Source:** https://github.com/fastapi/full-stack-fastapi-template/issues/1783

---

## Observation (What Happened)

A FastAPI + React application stored JWTs in `localStorage` but never validated the `exp` claim on the frontend before sending requests. The backend's `get_current_user()` dependency threw a generic 403 Forbidden for expired tokens (indistinguishable from a permissions error) rather than a 401 Unauthorized. Users received confusing errors rather than an automatic redirect to login. More critically, until the user manually clicked logout, the client kept sending expired tokens — and if backend validation was misconfigured, they might be silently accepted.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The auth hook stored the token but treated it as opaque — no code ever read the `exp` field from the JWT payload.

**Why #2:** Backend returned 403 (not 401) for expired tokens, so the frontend could not distinguish "expired" from "insufficient scope."

**Why #3:** The API client had no response interceptor — each 4xx error was handled inline without a centralized auth-failure path.

**Why #4:** The happy path (valid token, all works) was the only path exercised in development; token expiry only manifests after minutes/hours of real use.

**Why #5:** `localStorage` has no built-in expiry mechanism, unlike `httpOnly` cookies with `Max-Age`, so nothing forces a re-evaluation of the stored value.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | On the backend, return 401 for expired/invalid tokens, 403 only for insufficient permissions — this is the HTTP spec intention and enables clean frontend handling | proposed | community | issue #1783 |
| 2 | In the API client, add a response interceptor: on 401, clear the stored token and redirect to login | proposed | community | issue #1783 |
| 3 | On the frontend, decode the JWT (no verification needed client-side — just parse the payload) and check `exp` before each request; proactively redirect to login before sending an expired token | proposed | community | issue #1783 |
| 4 | Consider `httpOnly` cookies with `Max-Age` instead of `localStorage` — the browser enforces expiry automatically | proposed | community | issue #1783 |

## Key Takeaway

Return 401 (not 403) for expired tokens so the frontend can distinguish auth failure from permissions failure — and add a centralized response interceptor that clears credentials and redirects on 401, rather than handling expiry per-endpoint.
