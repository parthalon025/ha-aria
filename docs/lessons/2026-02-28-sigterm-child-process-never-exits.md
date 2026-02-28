# Lesson: SIGTERM to a Child Process Does Not Guarantee the Parent Process Exits

**Date:** 2026-02-28
**System:** community (prisma/prisma)
**Tier:** lesson
**Category:** integration
**Keywords:** SIGTERM, child process, Node.js, Alpine, musl, process hang, CI/CD, schema engine, exit
**Source:** https://github.com/prisma/prisma/issues/29169

---

## Observation (What Happened)

`prisma migrate deploy` completed all migration work and printed success, but the Node.js process never exited. The Rust-based schema engine child process received SIGTERM but did not shut down on Alpine Linux (musl libc). The parent Node.js process stayed alive indefinitely because it was waiting for the child process, blocking CI/CD predeploy hooks.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The parent process held a reference to the child process handle — Node.js event loop stays alive while child process handles are open.
**Why #2:** The child process (Rust binary compiled for musl) did not handle SIGTERM on Alpine — different signal semantics under musl vs glibc.
**Why #3:** The parent code assumed SIGTERM would terminate the child, then called `child.kill('SIGTERM')` without a timeout fallback.
**Why #4:** No `child.unref()` or `process.exit()` was called after the migration work completed — the process relied on natural exit.
**Why #5:** The behavior differed only on Alpine (musl), so it passed all glibc-based CI runners and was invisible until production deployment.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After sending SIGTERM to a child process, set a timeout and send SIGKILL if it hasn't exited | proposed | community | prisma#29169 |
| 2 | Call `child.unref()` after child process work completes so the parent can exit independently | proposed | community | prisma#29169 |
| 3 | Test process teardown on Alpine/musl images — signal semantics differ from glibc | proposed | community | prisma#29169 |
| 4 | Add explicit `process.exit(0)` after cleanup to avoid zombie parent processes | proposed | community | prisma#29169 |

## Key Takeaway

SIGTERM to a child process is not a guarantee of exit — always pair it with a SIGKILL timeout, call `child.unref()` when done, and test on Alpine/musl if that's a deployment target.
