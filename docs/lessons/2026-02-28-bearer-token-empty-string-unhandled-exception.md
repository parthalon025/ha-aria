# Lesson: `Authorization: Bearer` With No Token Value Raises Unhandled Exception Instead of 401

**Date:** 2026-02-28
**System:** community (django-oauth-toolkit)
**Tier:** lesson
**Category:** security
**Keywords:** bearer token, Authorization header, empty string, unhandled exception, 500, 401, input validation, token parsing, middleware
**Source:** https://github.com/jazzband/django-oauth-toolkit/issues/1496

---

## Observation (What Happened)

Sending `Authorization: Bearer` (with no token after the keyword) to a Django OAuth Toolkit protected endpoint raised an unhandled exception and returned a 500 Internal Server Error. The expected behavior is a clean 401 Unauthorized with `WWW-Authenticate: Bearer error="invalid_token"`. Returning 500 exposes stack traces and indicates an unguarded split/index operation on the header value.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The header parser split on whitespace and accessed index `[1]` without checking that at least two parts were present after splitting.

**Why #2:** Token extraction code assumed well-formed input — the "Bearer " prefix is always present and always followed by a non-empty value in legitimate clients.

**Why #3:** Security-layer input parsing was not subject to the same boundary-case coverage as business logic.

**Why #4:** The error surfaces as an IndexError or similar low-level exception that propagates to Django's 500 handler before the auth middleware can intercept it.

**Why #5:** Fuzzing or property-based tests on the Authorization header were not part of the test suite.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After splitting the Authorization header value, assert `len(parts) == 2 and parts[0].lower() == "bearer" and parts[1]` — return `None` (unauthenticated) immediately if any condition fails | proposed | community | issue #1496 |
| 2 | Wrap all header parsing in the auth middleware with a `try/except (IndexError, ValueError)` and return a 401 response rather than propagating the exception | proposed | community | issue #1496 |
| 3 | Add test cases for: `Authorization: Bearer` (no value), `Authorization: bearer`, `Authorization: ` (empty), `Authorization: NotBearer token123` | proposed | community | issue #1496 |

## Key Takeaway

Any code that parses security-critical request headers must guard against malformed input at the parse step — an IndexError from a split on `"Bearer "` should never become a 500; validate the parts count and token presence before accessing them.
