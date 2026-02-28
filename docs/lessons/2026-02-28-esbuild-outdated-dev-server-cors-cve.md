# Lesson: esbuild Dev Server Accepts Cross-Origin Requests — Pinned Old Versions Are a Security Risk

**Date:** 2026-02-28
**System:** community (drizzle-team/drizzle-orm)
**Tier:** lesson
**Category:** security
**Keywords:** esbuild, dev server, CORS, CVE, GHSA-67mh-4wv8-2f99, transitive dependency, security, npm audit
**Source:** https://github.com/drizzle-team/drizzle-orm/issues/5198

---

## Observation (What Happened)

drizzle-kit depended on `@esbuild-kit/esm-loader` which pinned esbuild `<= 0.24.2`. esbuild versions up to 0.24.2 have GHSA-67mh-4wv8-2f99 (CVSS 5.3): the dev server accepts HTTP requests from any website on the same machine, allowing cross-origin data exfiltration. Projects using `drizzle-kit` in dev inherited this vulnerability through the transitive dependency chain.

## Analysis (Root Cause — 5 Whys)

**Why #1:** esbuild's dev server before 0.25.0 set no CORS or Host header restrictions — any website open in the developer's browser could read the dev server's responses.
**Why #2:** drizzle-kit's transitive dep (`@esbuild-kit/esm-loader`) pinned esbuild to an old range that predated the fix.
**Why #3:** The dependency was not regularly audited for security advisories.
**Why #4:** `npm audit` would flag it, but not all projects run audits on transitive deps in their dev toolchain.
**Why #5:** Dev-only dependencies are often treated as "safe" because they don't ship to production — but they run locally and can be targeted by malicious websites.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Run `npm audit` in CI for all dependency categories including `devDependencies` | proposed | community | drizzle-orm#5198 |
| 2 | Use `overrides` / `resolutions` to force vulnerable transitive deps to patched versions | proposed | community | drizzle-orm#5198 |
| 3 | Treat dev toolchain vulnerabilities as real attack surface — developers run these locally with browser open | proposed | community | drizzle-orm#5198 |

## Key Takeaway

Dev-only dependencies are a real attack surface: a vulnerable esbuild dev server can be read by any malicious website open in the developer's browser — run `npm audit` on all dependency tiers and enforce minimum versions with `overrides`.
