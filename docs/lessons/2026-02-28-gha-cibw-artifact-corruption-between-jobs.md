# Lesson: GHA Artifact Corruption Between Jobs Is Silent — Wheels Uploaded Successfully Can Differ From Wheels Built

**Date:** 2026-02-28
**System:** community (pypa/cibuildwheel)
**Tier:** lesson
**Category:** ci-cd
**Keywords:** github-actions, artifact, upload, download, corruption, integrity, wheel, hash, upload-artifact, download-artifact
**Source:** https://github.com/pypa/cibuildwheel/issues/2627

---

## Observation (What Happened)

A TestPyPI upload succeeded and a production PyPI upload of the same artifacts (uploaded/downloaded via `actions/upload-artifact` and `actions/download-artifact` between jobs) failed with a mysterious validation error. Investigation found no clues — the suspicion was that GHA artifact storage had corrupted the files during the inter-job transfer. Because cibuildwheel does not print wheel hashes at build time, there was no baseline to compare the downloaded artifacts against.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Artifacts are uploaded by one job and downloaded by a separate upload-to-pypi job — two distinct storage operations across potentially different runner hosts.
**Why #2:** `actions/upload-artifact` and `actions/download-artifact` use GHA's blob storage; storage corruption, though rare, is not detectable without a hash comparison.
**Why #3:** No integrity check (SHA256 or MD5) is performed on artifact download before re-use — the downloaded file is assumed identical to the uploaded file.
**Why #4:** The build job and the publish job use different tools and have different failure modes — a corrupted wheel passes `twine check` but fails PyPI validation.
**Why #5:** Developers treat GHA artifacts as a reliable in-memory pass-through rather than a persistent storage write + read with potential for corruption.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Capture SHA256 hashes of built wheels at build time (`sha256sum dist/*`), store as a separate artifact, and compare after download in the publish job | proposed | community | https://github.com/pypa/cibuildwheel/issues/2627 |
| 2 | Run `twine check dist/*` immediately after downloading artifacts, before attempting any upload | proposed | community | — |
| 3 | Use `actions/upload-artifact@v4` with `compression-level: 0` for wheel files to reduce corruption surface during zip/unzip operations | proposed | community | — |

## Key Takeaway

GHA artifacts are not guaranteed bit-identical after upload+download — always hash artifacts at build time and verify after download before publishing, because artifact corruption is silent and rare enough to be mistaken for a tool bug.
