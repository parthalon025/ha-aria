# Lesson: Missing `id-token: write` Permission Silently Breaks OIDC Trusted Publishing

**Date:** 2026-02-28
**System:** community (pypa/gh-action-pypi-publish)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, OIDC, trusted-publishing, id-token, permissions, workflow-permissions, pypi
**Source:** https://github.com/pypa/gh-action-pypi-publish/issues/259

---

## Observation (What Happened)

After upgrading gh-action-pypi-publish from 0.9.0 to 0.10.0, workflows started failing with "Trusted publishing exchange failure: ACTIONS_ID_TOKEN_REQUEST_TOKEN environment variable was unset." The root cause was that `permissions: id-token: write` was missing at the job level. GitHub does not inject the OIDC token environment variables unless this permission is explicitly granted — the failure message is non-obvious and sends developers hunting for token configuration errors rather than workflow permissions.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Workflow worked with a PAT-based token but broke when switching to Trusted Publishing (OIDC).
**Why #2:** OIDC token exchange requires the GitHub Actions runner to hold a signed JWT, which is only issued when `id-token: write` is granted at the job (or workflow) level.
**Why #3:** If `permissions:` is absent entirely, GitHub defaults to a broad set that historically included id-token; newer GitHub policy changes tightened this, making it opt-in.
**Why #4:** The error message ("ACTIONS_ID_TOKEN_REQUEST_TOKEN environment variable was unset") describes a symptom, not the cause, so developers suspect OIDC configuration rather than missing permissions.
**Why #5:** The permissions block is easy to overlook when copying workflow snippets that predate OIDC trusted publishing.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always add `permissions: id-token: write` at the job level when using OIDC Trusted Publishing | proposed | community | https://github.com/pypa/gh-action-pypi-publish/issues/259 |
| 2 | Pair with minimum permissions (`contents: read`) to follow least-privilege even when adding id-token | proposed | community | — |
| 3 | Treat "ACTIONS_ID_TOKEN_REQUEST_TOKEN unset" as a permissions diagnostic, not an OIDC configuration error | proposed | community | — |

## Key Takeaway

OIDC Trusted Publishing requires explicit `id-token: write` at the job level — its absence silently prevents token injection and produces a misleading "environment variable unset" error rather than a permissions error.
