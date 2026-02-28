# Lesson: OAuth2 Authorization Code Flow Without PKCE Is Vulnerable to Code Interception on Public Clients

**Date:** 2026-02-28
**System:** community (oauthlib)
**Tier:** lesson
**Category:** security
**Keywords:** oauth2, PKCE, authorization code, public client, code interception, RFC 7636, S256, code_verifier, code_challenge
**Source:** https://github.com/oauthlib/oauthlib/issues/774

---

## Observation (What Happened)

The oauthlib client did not implement PKCE (RFC 7636) for Authorization Code Grant flows. Public clients (SPAs, mobile apps) that cannot keep a `client_secret` safe are vulnerable to authorization code interception: an attacker who intercepts the redirect can exchange the code for tokens without possessing the original `code_verifier`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** PKCE was originally defined as an extension for mobile apps (RFC 7636, 2015) and not folded into the OAuth2 core spec until OAuth 2.1. Older libraries simply did not include it.

**Why #2:** Server-side (confidential) clients are not vulnerable without PKCE because they have a `client_secret`, so the risk only surfaces for public clients — which many backend libraries never tested.

**Why #3:** Integration tests for the Authorization Code Grant typically use a client that has a `client_secret`, masking the PKCE gap.

**Why #4:** Developers building SPAs follow OAuth2 library examples that predate PKCE, copying flows that are no longer considered secure.

**Why #5:** The security benefit of PKCE is not visible in normal flow — it only matters when a code is intercepted, which never happens in a test environment.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | For any public client (SPA, CLI, mobile), always use `code_challenge_method=S256` — generate a cryptographically random `code_verifier`, SHA-256 hash it to produce `code_challenge`, include both in the authorize request | proposed | community | issue #774, RFC 7636 |
| 2 | On the server side, make `code_challenge` a required parameter for public clients and reject auth requests without it | proposed | community | issue #774 |
| 3 | Add `WebApplicationClient.create_authorization_url()` call with `code_verifier` argument and assert the resulting URL contains `code_challenge` and `code_challenge_method=S256` | proposed | community | issue #774 |

## Key Takeaway

Authorization Code Grant without PKCE is insecure for public clients — always generate and include a `code_verifier`/`code_challenge` pair (S256 method) and require it server-side for any client that cannot hold a secret.
