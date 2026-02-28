# Lesson: PM2 Does Not Propagate SIGKILL to Non-Node Child Processes — Orphans Survive
**Date:** 2026-02-28
**System:** community (Unitech/pm2)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** pm2, SIGKILL, process group, orphan, child process, fork mode, non-node, Go, Python, kill_timeout
**Source:** https://github.com/Unitech/pm2/issues/5937
---
## Observation (What Happened)
PM2 managed a Go binary alongside Node.js processes. When the operator sent `kill -9 -<pgid>` to terminate the process group, the two Node.js processes were killed but the Go process remained running. The Go process became an orphan — unmanaged by PM2, invisible in `pm2 list`, but consuming resources and potentially handling traffic.

## Analysis (Root Cause — 5 Whys)
PM2's Node.js runtime has native hooks into the Node.js process lifecycle; it can reliably signal and wait for Node.js children. For non-Node runtimes (Go, Python, native binaries) in fork mode, PM2 wraps the process but does not use cgroup-level or process-group-level kill operations by default. When PM2 itself receives a fatal signal (or is `kill -9`'d), its IPC channel with the child closes, but non-Node children may be in a signal handler or may not propagate the signal to their own children. The result is a partially-killed set of processes with orphaned non-Node children.

## Corrective Actions
- For non-Node processes managed by PM2: set `kill_timeout: 3000` (ms) so PM2 sends SIGKILL after SIGTERM timeout; also set `treekill: true` to signal the entire process subtree.
- Prefer systemd for managing mixed-runtime process trees — `KillMode=control-group` guarantees all processes in the cgroup are killed.
- Add post-stop verification: after `pm2 delete <app>`, run `pgrep -f <process-name>` to confirm no orphans remain.
- In containerized deployments: use `tini` or `dumb-init` as PID 1 to ensure proper signal propagation to all child processes regardless of PM2's behavior.

## Key Takeaway
PM2 does not reliably kill non-Node child processes — use `treekill: true` + `kill_timeout` for mixed runtimes, or prefer systemd with `KillMode=control-group` for non-Node process management.
