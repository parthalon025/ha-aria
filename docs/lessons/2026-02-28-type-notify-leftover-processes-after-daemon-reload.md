# Lesson: daemon-reload During Type=notify Service Startup Can Leave Orphan Processes
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** reliability
**Keywords:** systemd, Type=notify, daemon-reload, orphan process, leftover process, parallel start, KillMode, cgroup
**Source:** https://github.com/systemd/systemd/issues/37482
---
## Observation (What Happened)
A `Type=notify` service using a watchdog pattern (watchdog process sends `READY=1` on behalf of the main daemon) occasionally had two instances running simultaneously. The trigger was a `systemctl daemon-reload` issued at the moment the service transitioned from starting to started. systemd reported "Found left-over process in control group while starting unit" and restarted the service, resulting in two running instances competing for the same port.

## Analysis (Root Cause — 5 Whys)
When `daemon-reload` is received precisely as a `Type=notify` service reaches READY, systemd's unit state machine can lose track of the control-group state. The reload causes systemd to re-evaluate the unit, and in a race window, systemd sees a process in the cgroup that it believes belongs to a "previous run" (because the unit's internal start transaction was reset). It logs a warning but allows the service to start again, leaving the original process orphaned in the cgroup. With `KillMode=control-group` (the safe default), the orphan is killed when the unit stops — but during the window both instances handle requests.

## Corrective Actions
- Avoid issuing `systemctl daemon-reload` in automation (e.g., configuration management runs) when `Type=notify` services are mid-startup; use `--no-block` and stagger reloads.
- Use `KillMode=control-group` (the default) to ensure orphaned processes are reaped when the unit next stops/restarts.
- Add uniqueness enforcement in the service itself: acquire a lock file or bind the listening port at startup so a second instance fails fast rather than silently coexisting.
- Monitor for multiple processes: `systemctl show <service> --property=MainPID` and assert only one PID.

## Key Takeaway
A `daemon-reload` during a `Type=notify` service's startup window can leave orphan processes running — use `KillMode=control-group` and add application-level startup uniqueness guards (lock file or port bind).
