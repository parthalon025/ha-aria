# Lesson: .env File Committed to Git History Exposes Secrets Permanently Even After File Deletion

**Date:** 2026-02-28
**System:** community (fastapi/full-stack-fastapi-template)
**Tier:** lesson
**Category:** security
**Keywords:** git history, .env, secrets, SECRET_KEY, git-filter-repo, gitignore, password, API key, supply chain, rewrite history
**Source:** https://github.com/fastapi/full-stack-fastapi-template/issues/1758

---

## Observation (What Happened)

A `.env` file containing `SECRET_KEY`, database passwords, and API keys was committed to the repository. Deleting the file in a subsequent commit did not remove it from history — the secrets remained accessible via `git log -p` or by checking out any earlier commit. A full history rewrite with `git-filter-repo` was required, invalidating all existing clones and changing every commit hash in the repository.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The developer committed the `.env` file during initial project setup when `.gitignore` had not yet been configured.

**Why #2:** The `.gitignore` entry for `.env` was added after the first commit, which has no retroactive effect on already-tracked files.

**Why #3:** No pre-commit hook (e.g., `gitleaks`, `detect-secrets`, `git-secrets`) was installed to block the commit at write time.

**Why #4:** The developer assumed "I deleted the file, so the secrets are gone" — a common and dangerous misconception about Git's immutable object model.

**Why #5:** The repository was shared publicly before the exposure was discovered, requiring secret rotation on top of the history rewrite.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Install `gitleaks` as a pre-commit hook on every repo — blocks commits containing recognized secret patterns | proposed | community | issue #1758 |
| 2 | Add `.env`, `.env.*`, `*.pem`, `client_secret*.json` to `.gitignore` before writing any code | proposed | community | issue #1758 |
| 3 | Run `git rm --cached .env` immediately when a secrets file is found to already be tracked — then commit the removal | proposed | community | issue #1758 |
| 4 | After any accidental commit of secrets: rotate all exposed values first, then rewrite history with `git filter-repo --path .env --invert-paths` | proposed | community | issue #1758 |
| 5 | Use `.env.example` with placeholder values for documentation; never commit `.env` with real values | proposed | community | issue #1758 |

## Key Takeaway

Deleting a secrets file in a new commit does not remove it from Git history — the only remediation is a full history rewrite with `git filter-repo` plus immediate rotation of all exposed values; prevention requires `.gitignore` and pre-commit secret scanning installed before the first commit.
