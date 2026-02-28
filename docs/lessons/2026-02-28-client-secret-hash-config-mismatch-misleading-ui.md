# Lesson: UI Help Text Claims Secrets Are Hashed When Hash Setting Is Disabled — Operator Stores Plaintext Unaware

**Date:** 2026-02-28
**System:** community (django-oauth-toolkit)
**Tier:** lesson
**Category:** security
**Keywords:** client_secret, hash, plaintext, help_text, configuration mismatch, misleading UI, operator error, OAuth2, security assumption
**Source:** https://github.com/jazzband/django-oauth-toolkit/issues/1628

---

## Observation (What Happened)

The Django admin form for OAuth2 application management displayed "Hashed on Save. Copy it now if this is a new secret." for the `client_secret` field — regardless of the `OAUTH2_PROVIDER["HASH_CLIENT_SECRET"]` setting. When that setting is `False`, the secret is stored in plaintext, but the UI text told operators it would be hashed and therefore unrecoverable after saving. Operators who believed the UI might assume their DB is protected by hashing when it is not.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The `help_text` on the model field was hardcoded to describe the hashed behavior without being dynamically conditioned on the runtime setting.

**Why #2:** The setting's default changed between versions; the `help_text` was written when hashing was always enabled and was never updated for the configurable case.

**Why #3:** Model-level `help_text` is set at class definition time, not at render time, so it cannot read runtime settings without a custom `ModelAdmin`.

**Why #4:** Security-sensitive UI copy was not reviewed when the opt-out setting was introduced.

**Why #5:** Operators typically trust admin UI descriptions for security posture and do not cross-check them against application settings.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Override `get_form()` in `ApplicationAdmin` to set `help_text` dynamically based on `oauth2_settings.HASH_CLIENT_SECRET` | proposed | community | issue #1628 |
| 2 | When `HASH_CLIENT_SECRET=False`, display a prominent warning: "Secret stored as plaintext — ensure DB access is restricted" | proposed | community | issue #1628 |
| 3 | Apply the same principle project-wide: any UI that describes a security property (encrypted, hashed, signed) must read the live setting, not a hardcoded description | proposed | community | issue #1628 |

## Key Takeaway

UI copy that describes a security property (hashed, encrypted, signed) must be dynamically driven by the live configuration setting — static help text claiming "hashed on save" when hashing is disabled creates a dangerous false sense of security in the operator.
