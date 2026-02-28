# Lesson: WatchdogSec Changes Require Service Restart — daemon-reload Does Not Apply Them to Running Services
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** configuration
**Keywords:** systemd, WatchdogSec, daemon-reload, live update, watchdog, service restart, configuration drift
**Source:** https://github.com/systemd/systemd/issues/30922
---
## Observation (What Happened)
A systemd service was being killed by the watchdog timer under load. An operator increased `WatchdogSec=` in the unit file and ran `systemctl daemon-reload`. The service continued to be killed at the old, shorter watchdog interval. Even on subsequent watchdog-triggered restarts, the new `WatchdogSec=` value was not picked up — the service was restarted with the old timer.

## Analysis (Root Cause — 5 Whys)
`systemctl daemon-reload` re-reads unit file configurations but does not propagate runtime watchdog timer changes to currently running services. The watchdog timer is communicated to the child process at startup via the `WATCHDOG_USEC` environment variable and sd_notify protocol. A running service holds the original value and will not see updates until it stops and restarts. Furthermore, when the watchdog kills the service and systemd auto-restarts it, the restart picks up the cached (pre-reload) timer value, not the updated one — the new value only takes effect after a clean stop/start cycle.

## Corrective Actions
- After changing `WatchdogSec=`, always run `systemctl restart <service>` — `daemon-reload` alone is insufficient.
- Document in runbooks: "changing WatchdogSec requires restart, not just reload."
- For services that cannot be safely restarted while under load, use `systemctl set-property <service> WatchdogSec=X` (runtime property override) as a temporary measure, then schedule a planned restart.
- Add a monitoring check: after `daemon-reload`, verify that runtime `WatchdogUSec` matches the unit file via `systemctl show <service> --property=WatchdogUSec`.

## Key Takeaway
`WatchdogSec=` changes are not live-applied by `daemon-reload` — the service must be restarted for the new watchdog interval to take effect.
