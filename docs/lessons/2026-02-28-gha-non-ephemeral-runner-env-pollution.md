# Lesson: Non-Ephemeral Self-Hosted Runners Leak Environment State Between Jobs

**Date:** 2026-02-28
**System:** community (actions/actions-runner-controller, iterative/cml)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, self-hosted-runner, ephemeral, environment-pollution, state-leakage, ARC, cleanup
**Source:** https://github.com/actions/actions-runner-controller/issues/4258

---

## Observation (What Happened)

When ephemeral ARC runners fail mid-job, they enter a "Failed" state and are not automatically deleted — new runners cannot be created, all jobs queue indefinitely. Additionally, non-ephemeral self-hosted runners that persist between jobs accumulate filesystem state (tool caches, env vars set in previous jobs, leftover credentials) which bleeds into subsequent jobs from different workflows or authors, creating both correctness bugs and potential credential exposure.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Failed ephemeral runner pods were not cleaned up because the reconciliation loop did not handle the Failed pod phase.
**Why #2:** The controller expected runners to self-terminate, but pre-execution errors (before the runner binary starts) prevent the normal termination signal.
**Why #3:** Non-ephemeral runners (those not configured with `--ephemeral`) reuse the same workspace directory across jobs, meaning `$HOME`, tool caches, and credential files persist.
**Why #4:** Workflows that `export MY_SECRET=$(cat file)` or write to `~/.config/` during a job leave that state for the next job — across PRs, across authors.
**Why #5:** Self-hosted runner documentation emphasizes cost efficiency of persistent runners without surfacing the security model difference from GitHub-hosted ephemeral runners.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Configure ARC runners with `--ephemeral` flag or `ephemeral: true` in RunnerDeployment spec | proposed | community | https://github.com/actions/actions-runner-controller/issues/4258 |
| 2 | Add explicit cleanup steps (`rm -rf $HOME/.config/gcloud`, `unset credentials`) at job start and end | proposed | community | — |
| 3 | Monitor for runner pods stuck in Failed state; add controller-level GC for pods older than 10m | proposed | community | ARC issue #4258 |
| 4 | Treat non-ephemeral self-hosted runners as shared infrastructure — never pass `secrets.*` to `run:` steps that write to disk without explicit cleanup | proposed | community | — |

## Key Takeaway

Non-ephemeral self-hosted runners are shared mutable state — credentials, environment variables, and tool caches written in one job persist and leak to the next job unless explicit cleanup is enforced.
