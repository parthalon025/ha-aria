# Lesson: JWT Encode Accepts Empty String Secret — Produces Trivially Spoofable Token

**Date:** 2026-02-28
**System:** community (pyjwt)
**Tier:** lesson
**Category:** security
**Keywords:** jwt, secret, empty string, HMAC, HS256, spoofable, broken authentication, configuration, env var
**Source:** https://github.com/jpadilla/pyjwt/issues/1009

---

## Observation (What Happened)

`jwt.encode(payload, '', algorithm='HS256')` succeeds silently and produces a valid-looking JWT. Any attacker who knows the secret is empty can forge tokens for any identity. The empty secret most commonly arrives from a misconfigured environment variable (`SECRET_KEY=` with no value) or a missing secret-store lookup that returns `""` instead of raising.

## Analysis (Root Cause — 5 Whys)

**Why #1:** PyJWT's `encode()` does not validate the secret argument before constructing the HMAC signature, so an empty bytes object passes the HMAC call and produces a syntactically valid JWT.

**Why #2:** The library treats secret validation as the caller's responsibility, but callers rarely add an explicit guard.

**Why #3:** Secrets typically arrive through environment variables or secret stores at runtime; a missing or empty value produces `""` (not `None`), which passes `if secret:` checks and looks set.

**Why #4:** Tests often use short placeholder strings and never exercise the empty-string path.

**Why #5:** There is no application-layer guard between secret loading and token generation, so an empty secret flows through undetected until a security audit.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After loading `SECRET_KEY` from env, assert `len(SECRET_KEY) >= 32` and raise `RuntimeError` if not — fail at startup, not at token issue time | proposed | community | issue #1009 |
| 2 | Wrap `jwt.encode()` in a project-level helper that rejects empty/short secrets before delegating | proposed | community | issue #1009 |
| 3 | Add a startup health-check that calls `jwt.encode({"probe": 1}, SECRET_KEY, ...)` and immediately decodes it — proves the key is non-empty and round-trips before the first real request | proposed | community | issue #1009 |

## Key Takeaway

Always assert that a JWT secret is non-empty (and meets a minimum length) immediately after loading it from the environment — an empty secret produces a cryptographically valid but trivially forgeable token with no runtime error.
