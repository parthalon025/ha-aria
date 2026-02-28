# Lesson: GITHUB_TOKEN Expires After 24h — Long-Running Jobs Silently Lose Auth

**Date:** 2026-02-28
**System:** community (iterative/cml)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, GITHUB_TOKEN, expiry, long-running, timeout, authentication, ML-training
**Source:** https://github.com/iterative/cml/issues/1362

---

## Observation (What Happened)

ML training workflows that ran longer than 24 hours succeeded at the training stage but then failed at the CML report creation step with "Unable to extend GITHUB_TOKEN expiration time due to: GITHUB_TOKEN has expired." The token is valid at job start but GitHub does not allow extending it — once issued, it has a hard 24h ceiling.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The workflow used `GITHUB_TOKEN` (auto-generated per-job token) for CML's git operations and comment creation.
**Why #2:** `GITHUB_TOKEN` is issued at workflow trigger time and has a maximum lifetime of 24 hours with no renewal API.
**Why #3:** Long-running ML training jobs (model fitting, hyperparameter search) easily exceed 24h, especially with GPU queuing.
**Why #4:** The token expiry only manifests at the step that *uses* the token (report publish), not at training steps — the job appears healthy for 24+ hours before failing.
**Why #5:** Developers design workflows around job timeouts (6h GitHub-hosted default) without realizing self-hosted runners remove the job timeout ceiling but not the token ceiling.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use a PAT (Personal Access Token) or GitHub App token instead of `GITHUB_TOKEN` for workflows expected to exceed 24h | proposed | community | https://github.com/iterative/cml/issues/1362 |
| 2 | Split long-running jobs: run training as one job (no GH API calls needed), then trigger a downstream job for reporting using a fresh `GITHUB_TOKEN` | proposed | community | — |
| 3 | Add explicit `timeout-minutes` to long jobs as a kill switch — an expired token means the job result cannot be reported anyway | proposed | community | — |

## Key Takeaway

`GITHUB_TOKEN` has a hard 24-hour expiry that cannot be extended — jobs longer than 24h must use a PAT or GitHub App token, or be split so GH API calls happen in a fresh downstream job.
