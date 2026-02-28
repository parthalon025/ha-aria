# Lesson: JSON.parse Without try-catch Is a DoS Vector at HTTP Request Boundaries

**Date:** 2026-02-28
**System:** community (honojs/hono)
**Tier:** lesson
**Category:** error-handling
**Keywords:** JSON.parse, try-catch, DoS, request body, malformed JSON, error handling, HTTP, TypeScript, hono
**Source:** https://github.com/honojs/hono/issues/4711

---

## Observation (What Happened)

Hono's request body parsing called `JSON.parse(body)` without a surrounding `try-catch`. Any HTTP request with a malformed JSON body (missing closing brace, trailing comma, non-JSON content-type mismatch) threw an uncaught `SyntaxError` that propagated as an unhandled exception, crashing the handler and potentially the process.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `JSON.parse` throws `SyntaxError` on invalid input — it never returns a sentinel value.
**Why #2:** The body parsing code assumed valid JSON because the request had `Content-Type: application/json`.
**Why #3:** No defensive wrapper was added around the parse step despite being at the network boundary.
**Why #4:** The error did not surface in development because test clients always sent valid JSON.
**Why #5:** An attacker (or misconfigured client) can trivially send a POST body of `{` to crash any unprotected endpoint.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always wrap `JSON.parse` at HTTP request boundaries in `try-catch` — return 400 on SyntaxError | proposed | community | hono#4711 |
| 2 | Use a typed parse utility (`safeJsonParse`) that returns `{ ok: boolean, value }` instead of throwing | proposed | community | hono#4711 |
| 3 | Include a fuzz test in CI: POST random/malformed bodies to every JSON-consuming endpoint | proposed | community | hono#4711 |

## Key Takeaway

`JSON.parse` at any network boundary that accepts untrusted input must be wrapped in `try-catch` — a missing brace in a POST body is a one-line DoS; use a safe-parse wrapper that returns a result type, not an exception.
