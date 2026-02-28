# Lesson: Supervisor Crashes at Startup if Log File Is Not Writable by the Running User
**Date:** 2026-02-28
**System:** community (Supervisor/supervisor)
**Tier:** lesson
**Category:** reliability
**Keywords:** supervisord, log file, permissions, PermissionError, startup crash, non-root, user, log rotation
**Source:** https://github.com/Supervisor/supervisor/issues/1648
---
## Observation (What Happened)
A supervisord instance running as a non-root user crashed with `PermissionError: [Errno 13] Permission denied` on its own log file at the very first line of startup — before any managed processes were launched. The log file was owned by root from a previous run under a different user context. Supervisor could not emit any logs to report the error, producing a bare traceback with no managed-process context.

## Analysis (Root Cause — 5 Whys)
supervisord opens its log file (`supervisord.log`) as the first action in `main()`, before any process supervision starts. If the log file (or its parent directory) is not writable by the current effective user, the `RotatingFileHandler` constructor raises `PermissionError`, which propagates up and kills the entire supervisor process. There is no pre-flight permission check or fallback to stderr — the process simply dies before it can supervise anything. The root cause is typically a log file left with stale ownership after a one-time root-executed setup step.

## Corrective Actions
- In deployment: `chown <supervisord-user> <log-dir>/supervisord.log` before starting supervisor under a non-root user.
- Add a pre-start check in the supervisor wrapper script: `test -w "$LOGFILE" || { echo "supervisord log not writable"; exit 1; }`.
- For systemd-managed supervisord: add `ExecStartPre=/bin/chown <user> <logfile>` to the unit file to self-heal ownership on restart.
- In ARIA context: any managed service with external log files (e.g., `/var/log/`) must have ownership verified as part of deployment runbooks.

## Key Takeaway
supervisord crashes before managing any process if its log file is unwritable — pre-flight `test -w` or `chown` in the start sequence prevents silent service manager failure.
