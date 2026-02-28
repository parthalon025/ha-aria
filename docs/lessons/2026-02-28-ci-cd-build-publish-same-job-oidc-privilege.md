# Lesson: Building and Publishing in the Same CI Job Grants Build Dependencies OIDC Publish Privileges

**Date:** 2026-02-28
**System:** community (oauthlib)
**Tier:** lesson
**Category:** security
**Keywords:** CI/CD, OIDC, Trusted Publishing, PyPI, GitHub Actions, supply chain, privilege escalation, build deps, job isolation
**Source:** https://github.com/oauthlib/oauthlib/issues/913

---

## Observation (What Happened)

The oauthlib CI workflow ran `pip install build` (pulling transitive deps) and `python -m build` in the same GitHub Actions job that published to PyPI via OIDC Trusted Publishing. Any transitive build dependency — including a typosquatted or compromised package — ran code inside a job that held the OIDC token with publish permissions. A single malicious build dep could extract the token and push a backdoored release.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The workflow combined build and publish in one job for simplicity, without realizing that OIDC tokens are scoped per-job, not per-step.

**Why #2:** The OIDC Trusted Publishing design mints a token for the entire job, so all steps in that job inherit publish authority regardless of when in the job the token is first used.

**Why #3:** The `if:` condition controlling the publish step was always `true` due to a broken `${{ }}` expression — meaning publish ran on every push, not just on release tags.

**Why #4:** The pipeline was written before OIDC Trusted Publishing existed; when migrated away from a stored PyPI token, the job boundary was not reconsidered.

**Why #5:** Supply-chain attacks via build deps are not part of the threat model most developers keep in mind when writing CI workflows.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Split into two separate jobs: `build` (runs without OIDC publish permission) and `publish` (depends on `build`, requests `id-token: write`, uploads the pre-built artifacts) | proposed | community | issue #913, PyPA guide |
| 2 | In the `publish` job, use `actions/download-artifact` to retrieve the dist files built in the `build` job — no build tooling runs in the publish job | proposed | community | issue #913 |
| 3 | Set `permissions: id-token: write` only on the `publish` job, not on the entire workflow | proposed | community | issue #913 |
| 4 | Gate publish with `if: github.event_name == 'release' && github.event.action == 'published'` to prevent publishing on every push | proposed | community | issue #913 |

## Key Takeaway

Always separate build and publish into distinct CI jobs — OIDC tokens scope to the job, so build dependencies running in a publish job inherit the authority to push malicious releases to your package registry.
