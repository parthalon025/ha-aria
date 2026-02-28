# Lesson: Unpinned Third-Party Actions Silently Break on Major Version Upgrades (Cosign v2→v3 Case)

**Date:** 2026-02-28
**System:** community (goreleaser/goreleaser)
**Tier:** lesson
**Category:** security
**Keywords:** github-actions, supply-chain, unpinned-action, cosign, semver, breaking-change, pin-to-sha, sigstore
**Source:** https://github.com/goreleaser/goreleaser/issues/6195

---

## Observation (What Happened)

GoReleaser's signing configuration relied on `sigstore/cosign-installer` without pinning the cosign version. When cosign v3.0.0 released, it removed the `--output-certificate` and `--output-signature` flags, replacing them with `--bundle`. All existing `.goreleaser.yaml` sign configurations that referenced these flags silently broke — releases started failing mid-run during the sign step with no pre-release warning.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Workflows used `uses: sigstore/cosign-installer@v3` (tag pin) without specifying `cosign-release: v2.x.x` in the `with:` block.
**Why #2:** The tag `v3` of the installer action resolves to `latest cosign v3.x`, which drifted to 3.0.0 with breaking API changes.
**Why #3:** cosign v3 changed its CLI interface (removed flags, added `--bundle`), but the workflow had no mechanism to detect this drift — no integration test against the live signing step.
**Why #4:** The installer action allows pinning cosign independently of the action version, but this is opt-in documentation behavior — the default behavior fetches latest cosign.
**Why #5:** Developers assume semver minor/patch version bumps are safe but miss that the installer action's version and the tool's version are two independent semver axes.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pin actions to full SHA (`uses: sigstore/cosign-installer@faadad0c...`) and separately pin the tool version (`cosign-release: "v2.6.1"`) | proposed | community | https://github.com/goreleaser/goreleaser/issues/6195 |
| 2 | Use Dependabot or Renovate to get automated PRs for action SHA updates rather than drifting on floating tags | proposed | community | — |
| 3 | Test the full release pipeline (including signing) against a pre-release tag in a staging workflow before the real release — catches breaking CLI changes before they hit production | proposed | community | — |
| 4 | Audit all `uses:` entries for floating tags (`@main`, `@v3`, `@latest`) and replace with SHA pins | proposed | community | — |

## Key Takeaway

Floating action tags (`@v3`, `@main`) cause silent breakage when the referenced action or tool releases breaking changes — pin all third-party actions to a full commit SHA and use Dependabot to receive controlled update PRs.
