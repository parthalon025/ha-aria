# Lesson: ExecReload Using SIGHUP Can Cause Service Self-Termination
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** systemd, ExecReload, SIGHUP, KillSignal, service termination, reload, signal propagation, systemctl reload
**Source:** https://github.com/systemd/systemd/issues/34042
---
## Observation (What Happened)
A service used `ExecReload=/bin/kill -HUP $MAINPID` to trigger live configuration reload (standard for Prometheus node_exporter and many daemons). Upon calling `systemctl reload <service>`, systemd sent SIGHUP to the main process as expected — but systemd also interpreted the process's receipt of SIGHUP as a reason to deactivate the service, causing it to stop rather than reload.

## Analysis (Root Cause — 5 Whys)
On Debian 12 with systemd 252, when the service's `KillSignal=` is not explicitly set, systemd defaults to SIGTERM for kills. However, certain signal handling interactions cause systemd to treat SIGHUP received during the ExecReload phase as a service failure condition rather than an anticipated reload event. The issue arises because the signal is sent to MAINPID and systemd monitors that PID for unexpected exits; when the process's signal disposition causes an ambiguous state transition, systemd may decide to deactivate the unit. The canonical fix is to use `KillSignal=SIGTERM` explicitly and ensure `ExecReload=` uses a clean SIGHUP mechanism.

## Corrective Actions
- Always set `KillSignal=SIGTERM` explicitly when using `ExecReload=/bin/kill -HUP $MAINPID` to prevent signal ambiguity.
- Alternatively, replace `/bin/kill -HUP` with a proper reload command that the service provides (e.g., `ExecReload=/path/to/binary --reload`).
- After any `ExecReload=` change, run `systemctl reload <service>` and verify `systemctl is-active <service>` still shows `active`.
- Add integration test: issue reload during steady-state operation and confirm service remains up (not deactivated).

## Key Takeaway
Using `ExecReload=/bin/kill -HUP $MAINPID` without explicit `KillSignal=SIGTERM` can cause systemd to misinterpret the reload signal as a service failure — always set `KillSignal=SIGTERM` alongside SIGHUP-based reloads.
