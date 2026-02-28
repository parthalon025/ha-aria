# Lesson: PyPI Artifact Upload is Non-Transactional — Partial Releases Are Visible

**Date:** 2026-02-28
**System:** community (pypa/gh-action-pypi-publish)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, pypi, upload, transactional, partial-release, artifact, publish, rollback
**Source:** https://github.com/pypa/gh-action-pypi-publish/issues/385

---

## Observation (What Happened)

A multi-artifact release (multiple wheel files + sdist) uploaded all binaries successfully but failed on the sdist with HTTP 400. PyPI had already accepted and published the binary wheels, so the release appeared as the "current" version in PyPI's UI with missing artifacts. There is no rollback API — the partially published release is permanent and users will try to install an incomplete distribution.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The workflow uploaded artifacts sequentially; twine stops on the first failure but does not attempt to delete already-uploaded artifacts.
**Why #2:** PyPI's legacy upload API is non-transactional — each file is an independent POST request with no batch semantics.
**Why #3:** A 400 on the sdist upload (likely a metadata validation issue) is unrelated to the binaries and occurs at a different upload step.
**Why #4:** The workflow treats the job as failed (correct) but leaves the partial release published (no cleanup step).
**Why #5:** Developers assume "job failed" = "nothing was published" — the non-transactional upload model violates this assumption.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Validate all artifacts with `twine check dist/*` in a prior job before any upload — catch metadata errors before any artifact is published | proposed | community | https://github.com/pypa/gh-action-pypi-publish/issues/385 |
| 2 | Use `--skip-existing` in upload config when re-running to avoid double-upload errors after partial failure | proposed | community | — |
| 3 | Pin version numbers only in release tags — never auto-increment on partial failure (new version = clean slate) | proposed | community | — |
| 4 | Document in release workflow that partial publication is a manual cleanup concern, not auto-rolled-back | proposed | community | — |

## Key Takeaway

PyPI artifact upload is non-transactional: a failure mid-upload leaves a partial release publicly visible — validate all artifacts before uploading any, because there is no automatic rollback.
