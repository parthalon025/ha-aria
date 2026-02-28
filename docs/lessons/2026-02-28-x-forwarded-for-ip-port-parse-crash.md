# Lesson: X-Forwarded-For Parser Crashes on IP:Port Format — Proxy-Specific Header Variations Cause 403

**Date:** 2026-02-28
**System:** community (django-allauth)
**Tier:** lesson
**Category:** security
**Keywords:** X-Forwarded-For, IP address, port, proxy, Azure, HTTP/2, ipaddress.ip_address, PermissionDenied, 403, header parsing, input validation
**Source:** https://github.com/pennersr/django-allauth/issues/4230

---

## Observation (What Happened)

Azure App Service's HTTP/2 proxy includes the client port in the `X-Forwarded-For` header (`94.252.75.68:22272` instead of `94.252.75.68`). Django allauth's `get_client_ip()` passed this value directly to `ipaddress.ip_address()`, which raised `ValueError: '94.252.75.68:22272' does not appear to be an IPv4 or IPv6 address`. The exception propagated as `PermissionDenied`, causing all login attempts to return 403 — silently blocking all users behind the Azure proxy.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `get_client_ip()` called `ipaddress.ip_address(ip_value)` directly without stripping the port component.

**Why #2:** The HTTP spec does not forbid `IP:port` in `X-Forwarded-For`, and some proxies (Azure, some CDNs) include it — but the code was written assuming IP-only format.

**Why #3:** The failure mode was a hard exception that Django's exception middleware converted to `PermissionDenied`, making it appear as an authorization failure rather than a parsing error.

**Why #4:** The behavior only manifests with specific proxy configurations — not in development environments or with standard proxies like nginx.

**Why #5:** There was no test covering non-standard `X-Forwarded-For` formats, and the error was not caught until a production deployment on Azure.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Strip the port before parsing: `ip_str = ip_value.split(":")[0]` for IPv4, or use a more robust parser that handles both `IP:port` and `[IPv6]:port` formats | proposed | community | issue #4230 |
| 2 | Wrap `ipaddress.ip_address()` in a `try/except ValueError` and return a safe fallback (e.g., `"0.0.0.0"`) rather than raising `PermissionDenied` on parse failure | proposed | community | issue #4230 |
| 3 | Add test cases for all `X-Forwarded-For` variants: plain IP, `IP:port`, comma-separated list, IPv6 in brackets, and empty string | proposed | community | issue #4230 |
| 4 | Apply the same defensive parsing to all places that read proxy headers (`X-Real-IP`, `CF-Connecting-IP`, etc.) | proposed | community | issue #4230 |

## Key Takeaway

Never pass `X-Forwarded-For` values directly to `ipaddress.ip_address()` — always strip the port component first and wrap the call in `try/except ValueError`, because proxy behavior varies and a parse crash should never surface as an auth failure.
