# Lesson: Unreferenced Repository Secrets Persist Indefinitely and Expand the Secret Exposure Surface

**Date:** 2026-02-28
**System:** community (iterative/cml)
**Tier:** lesson
**Category:** security
**Keywords:** github-actions, secrets, stale, unreferenced, least-privilege, audit, rotation, AZURE, GCP, SLACK_WEBHOOK
**Source:** https://github.com/iterative/cml/issues/1426

---

## Observation (What Happened)

A security audit of the CML repo flagged six repository secrets (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_SUBSCRIPTION_ID`, `AZURE_TENANT_ID`, `GOOGLE_APPLICATION_CREDENTIALS_DATA`, `SLACK_WEBHOOK`) that were not referenced in any workflow on the default branch. These secrets had been accumulating since previous CI configurations were deleted or refactored — the secrets themselves were not cleaned up. Each unreferenced secret remains a valid credential that can be extracted by any future workflow added to the repo.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Workflows were refactored or deleted, but the repository secrets they used were not removed.
**Why #2:** GitHub has no "warn on unused secret" mechanism — secrets sit silently in Settings → Secrets regardless of whether any workflow references them.
**Why #3:** Secret rotation policies (if any) are attached to usage context, not to the secret object itself — unused secrets drift out of rotation cycles.
**Why #4:** Any workflow file added to the repo (including from a PR, if `pull_request_target` is misused) can reference and exfiltrate these secrets.
**Why #5:** Developers treat secret cleanup as optional post-delete cleanup rather than as a required step of the workflow deletion process.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Treat secret deletion as a required step of workflow deletion — add it to the PR checklist for any workflow that is removed or refactored | proposed | community | https://github.com/iterative/cml/issues/1426 |
| 2 | Quarterly audit: enumerate all repository secrets and cross-reference against active workflow files; delete any with no matching `secrets.SECRET_NAME` reference | proposed | community | — |
| 3 | Rotate credentials before deleting secrets — an unreferenced secret may still be a live credential elsewhere | proposed | community | — |

## Key Takeaway

Unused repository secrets are live credentials with no automatic expiry — always delete secrets in the same PR that removes the workflow referencing them, and audit for orphaned secrets quarterly.
