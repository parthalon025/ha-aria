# Lesson: Shell Entrypoint Without `exec` Swallows SIGTERM

**Date:** 2026-02-28
**System:** community (krallin/tini, docker/compose)
**Tier:** lesson
**Category:** reliability
**Keywords:** SIGTERM, PID 1, signal forwarding, entrypoint, shell, exec, graceful shutdown, Docker
**Source:** https://github.com/krallin/tini/issues/195

---

## Observation (What Happened)

A Dockerfile uses `ENTRYPOINT ["/sbin/tini", "--", "entrypoint.sh"]` where `entrypoint.sh` ends with `exec bash`. On `docker stop`, the container receives SIGKILL (exit 137) after the timeout instead of handling SIGTERM, because `exec bash` replaces the shell but bash itself does not forward signals properly in this context. Containers that do NOT use `exec` fare worse: the shell receives SIGTERM but never forwards it to the child process, which gets SIGKILL when the stop timeout expires.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `docker stop` sends SIGTERM to PID 1 — the process with that PID must handle or forward the signal.
**Why #2:** A shell script run as entrypoint IS PID 1. Shells (sh/bash) do not forward signals to child processes launched as background jobs or via `exec` in certain contexts.
**Why #3:** Without `exec yourapp` as the final line of the entrypoint script, the shell sits as PID 1 and the actual application runs as a child with a different PID. SIGTERM lands on the shell, which ignores or drops it, leaving the child process alive until SIGKILL.
**Why #4:** Even WITH `exec yourapp`, if the app itself doesn't register a SIGTERM handler, it still terminates uncleanly. The `exec` fixes signal routing but not signal handling in the app.
**Why #5:** Many developers write `CMD ["python", "app.py"]` (exec form — correct) but combine it with a shell entrypoint script that uses `ENTRYPOINT ["/bin/sh", "-c", "..."]` (shell form), which wraps everything in a shell and recreates the problem.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use exec form (`ENTRYPOINT ["cmd", "arg"]`) instead of shell form (`ENTRYPOINT cmd arg`) — exec form makes your process PID 1 directly | proposed | community | krallin/tini#195 |
| 2 | If entrypoint.sh is needed, end it with `exec "$@"` (pass-through) so the CMD argument becomes PID 1 | proposed | community | tini docs |
| 3 | Add `init: true` in docker-compose.yml or `--init` flag to insert tini as a proper PID 1 init that reaps zombies and forwards signals | proposed | community | docker docs |
| 4 | For Python: use `signal.signal(signal.SIGTERM, handler)` — Python does not forward SIGTERM to subprocesses by default | proposed | community | python signal docs |

## Key Takeaway

An entrypoint shell script that does not end with `exec "$@"` makes the shell PID 1 and silently swallows SIGTERM, forcing Docker to SIGKILL the container after the stop timeout — causing data loss and dirty shutdown in any service using SQLite, file locks, or in-flight requests.
