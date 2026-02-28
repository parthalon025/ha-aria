# Lesson: Config History Endpoint Exposes Sensitive Key Values in Cleartext

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** security
**Keywords:** config history, sensitive values, redaction, cleartext, API, password, token, audit log
**Files:** aria/hub/api.py:1211-1222

---

## Observation (What Happened)

`GET /api/config/history` returns `old_value` and `new_value` for every config change as plaintext. If a user ever changed a sensitive config key (password, token, API key), both the old and new values are permanently exposed in the history log to any authenticated caller.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The history endpoint serializes the raw audit table rows without any redaction pass.

**Why #2:** The config system stores all values uniformly — it doesn't distinguish between safe settings (intervals, thresholds) and sensitive ones (credentials, keys).

**Why #3:** Audit logs are typically treated as internal data, so security review is sometimes skipped. But an authenticated REST endpoint making audit data accessible to all API clients has the same exposure as any other data endpoint.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `_is_sensitive_key(key)` check in the history serializer; replace `old_value`/`new_value` with `"***REDACTED***"` for sensitive keys | proposed | Justin | issue #180 |
| 2 | Define a `SENSITIVE_CONFIG_KEYS` set (keys containing "password", "token", "key", "secret") to drive redaction | proposed | Justin | issue #180 |
| 3 | Apply the same redaction to any single-key config endpoints that return current values | proposed | Justin | issue #153 |

## Key Takeaway

Any endpoint that exposes config change history must redact sensitive key values — the audit log is a security boundary, not just an internal diagnostic, once it becomes a REST API response.
