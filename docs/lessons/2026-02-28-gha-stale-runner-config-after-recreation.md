# Lesson: Stale Listener Config Secrets After Runner Scale Set Recreation Cause Silent Job Queuing

**Date:** 2026-02-28
**System:** community (actions/actions-runner-controller)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, ARC, runner-controller, stale-config, scale-set-id, listener, kubernetes, helm
**Source:** https://github.com/actions/actions-runner-controller/issues/4322

---

## Observation (What Happened)

After deleting and recreating an AutoscalingRunnerSet via Helm (uninstall/reinstall), the runner scale set received a new ID from GitHub (e.g., 87→90). However, the listener config secret was not updated — it retained the old scale set ID (87). Listeners continued registering against the stale ID, causing jobs to queue indefinitely with no error in the workflow UI. No warning was logged by the controller.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Controller reconciliation uses the runner set ID from the secret, not from the live GitHub API response, when the ARS annotation has changed.
**Why #2:** The secret update path was only triggered on ARS creation, not on ID-change-after-recreation, because the controller's reconciler checked for the secret's existence, not its contents.
**Why #3:** GitHub assigns a new integer ID for each scale set registration — deletion + recreation is not an update, it is a new entity.
**Why #4:** The stale ID causes listener WebSocket sessions to connect to a non-existent scale set — GitHub silently drops the jobs rather than returning an error to the listener.
**Why #5:** Developers assume Helm uninstall/reinstall is idempotent with respect to GH-side state, but GitHub's scale set IDs are not stable across deletions.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After Helm reinstall, verify the listener secret contains the new scale set ID before expecting jobs to run | proposed | community | https://github.com/actions/actions-runner-controller/issues/4322 |
| 2 | Force-delete the listener config secret before reinstalling ARC to ensure the controller creates a fresh one | proposed | community | — |
| 3 | Treat Helm reinstall as a state-breaking operation for ARC — always validate runner registration against the GitHub API after reinstall | proposed | community | — |

## Key Takeaway

GitHub assigns a new scale set ID on every deletion+recreation — ARC listener config secrets that retain the old ID cause silent job queuing; always force-clean listener secrets during Helm reinstalls.
