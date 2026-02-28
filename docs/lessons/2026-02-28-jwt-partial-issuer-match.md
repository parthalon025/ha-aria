# Lesson: JWT Issuer Validation Accepts Partial String Match — Any Prefix Passes

**Date:** 2026-02-28
**System:** community (pyjwt)
**Tier:** lesson
**Category:** security
**Keywords:** jwt, issuer, iss, partial match, validation, token forgery, PyJWT 2.10, algorithm confusion
**Source:** https://github.com/jpadilla/pyjwt/issues/1020

---

## Observation (What Happened)

In PyJWT 2.10.0, passing `issuer="https://auth.example.com"` to `jwt.decode()` accepted tokens with `"iss": "http"` — a four-character prefix. The validation checked substring containment rather than exact equality, meaning any token whose `iss` is a prefix of the expected issuer string passes verification. A token from a completely different issuer could be accepted if it shared a common URL prefix.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `_validate_iss` method treated a string issuer as a single-element container for an `in` check (`payload["iss"] in issuer`) rather than an equality check.

**Why #2:** The code path that runs when `issuer` is a `str` inverted the containment direction — checking whether the payload `iss` is "in" the issuer string instead of whether it equals it.

**Why #3:** A refactor introduced in 2.10.0 to unify list and scalar issuer handling broke the exact-match guarantee without a test catching the regression.

**Why #4:** Security-critical validation code was changed without a property-based or fuzz test that generates adversarial issuer values.

**Why #5:** The subtle Python behavior (`"http" in "https://..."` is `True`) is not immediately obvious during code review.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pass issuer as a list (`issuer=["https://auth.example.com"]`) rather than a bare string — forces PyJWT into the `__contains__` path which does exact element comparison | proposed | community | issue #1020 |
| 2 | Add a unit test that verifies a token with `iss="http"` is rejected when `issuer="https://auth.example.com"` | proposed | community | issue #1020 |
| 3 | Pin PyJWT version; when upgrading, run the issuer-mismatch test before promoting to production | proposed | community | issue #1020 |

## Key Takeaway

Pass JWT issuers as a list, not a bare string, to force exact-element matching — string membership semantics in Python make substring matches silently pass exact-issuer validation.
