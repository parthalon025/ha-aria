# Lesson: OAuth2 Refresh Token Not Revoked After Use — Replay Attack Window Remains Open

**Date:** 2026-02-28
**System:** community (oauthlib)
**Tier:** lesson
**Category:** security
**Keywords:** oauth2, refresh token, revocation, replay attack, token rotation, RFC 6749, grant, validator
**Source:** https://github.com/oauthlib/oauthlib/issues/770

---

## Observation (What Happened)

An OAuth2 provider built on oauthlib issued new access + refresh token pairs on refresh-grant requests but never invalidated the old refresh token. RFC 6749 §6 states "The authorization server MAY revoke the old refresh token after issuing a new refresh token." Many deployments skip this step, leaving the old token valid indefinitely. If it leaks (log line, network capture, compromised client), an attacker can exchange it for fresh tokens with no time constraint.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The oauthlib flow calls `validate_refresh_token` and then `save_bearer_token`, but neither method receives the old token as an argument in `save_bearer_token`, making revocation difficult to wire in.

**Why #2:** The RFC's "MAY revoke" language is treated as optional, so implementors skip it unless they read the security considerations section.

**Why #3:** The framework's `RequestValidator` interface does not have a dedicated `revoke_old_refresh_token(old_token, request)` hook, so developers are unsure where to add the revocation logic.

**Why #4:** Testing the refresh grant typically verifies that a new token is issued, not that the old one is rejected on a second attempt.

**Why #5:** Refresh token lifetime is often not monitored — expired tokens never cause issues in happy-path tests, so non-rotation goes unnoticed.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | In `validate_refresh_token`, store the old token's DB ID on the `request` object (`request.old_refresh_token_id = ...`) so it's available in downstream hooks | proposed | community | issue #770 |
| 2 | In `save_bearer_token`, call `self.revoke_token(request.old_refresh_token_id)` before returning | proposed | community | issue #770 |
| 3 | Add an integration test: use the same refresh token twice and assert the second exchange returns `invalid_grant` | proposed | community | issue #770 |
| 4 | Set a maximum refresh token lifetime (e.g., 30 days) and enforce it even without explicit revocation | proposed | community | issue #770 |

## Key Takeaway

Always revoke the old refresh token atomically when issuing a replacement — even though RFC 6749 says "MAY", leaving the old token valid means a single interception creates an unlimited replay window.
