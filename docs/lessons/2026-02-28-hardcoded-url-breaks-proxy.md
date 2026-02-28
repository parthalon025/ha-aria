# Lesson: Hardcoded Absolute URL Bypasses Configured Base URL — Broken Under Proxy

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** frontend
**Keywords:** hardcoded URL, base URL, proxy, Tailscale, path prefix, absolute path, baseUrl, fetch, Preact, integration boundary
**Files:** aria/dashboard/spa/src/pages/Presence.jsx

---

## Observation (What Happened)

`Presence.jsx` constructed Frigate thumbnail URLs as absolute paths (`/api/frigate/thumbnail/...`) hardcoded in the component, bypassing the `baseUrl` config used by every other API call. When the dashboard was served via a Tailscale proxy with a path prefix (e.g., `/siri/`), thumbnails 404'd on every request because the absolute path skipped the prefix (issue #287).

## Analysis (Root Cause — 5 Whys)

**Why #1:** The thumbnail URL was constructed inline as a template literal with a hardcoded leading `/` rather than `${baseUrl}/api/frigate/...`.

**Why #2:** The developer was building a new feature and wrote the URL directly without checking how the rest of the codebase constructed API URLs.

**Why #3:** There was no enforced pattern or linting rule requiring all API URL construction to go through the central `baseUrl`/config — the contract was implicit.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace `/api/frigate/thumbnail/...` with `${baseUrl}/api/frigate/thumbnail/...` in Presence.jsx line 182 | proposed | Justin | Presence.jsx #287 |
| 2 | Audit all URL construction in the SPA for bare absolute paths that bypass `baseUrl` | proposed | Justin | — |
| 3 | Make `baseUrl` available via a shared utility/constant so every new fetch call is forced to import and use it | proposed | Justin | — |

## Key Takeaway

Every API URL in the SPA must be constructed using the configured `baseUrl` — hardcoded absolute paths silently break under any proxy or path-prefix deployment.
