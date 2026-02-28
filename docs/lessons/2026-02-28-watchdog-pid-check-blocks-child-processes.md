# Lesson: sd_watchdog_enabled() Returns 0 for Grandchild Processes Even With NotifyAccess=all
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** reliability
**Keywords:** systemd, WatchdogSec, sd_watchdog_enabled, NotifyAccess, grandchild, PID, daemonize, fork
**Source:** https://github.com/systemd/systemd/issues/25961
---
## Observation (What Happened)
A service using `Type=forking` and `WatchdogSec=15s` with `NotifyAccess=all` had its deeply-forked daemon process call `sd_watchdog_enabled()` to determine whether to send watchdog pings. The function consistently returned `0` even though `WATCHDOG_USEC` was present in the environment. The daemon never sent watchdog signals, causing the service to be killed by the watchdog timer.

## Analysis (Root Cause — 5 Whys)
`sd_watchdog_enabled()` internally compares `WATCHDOG_PID` against the current process PID (via `getpid_cached()`). When a shell script or multi-stage launcher forks to reach the final daemon, the running PID does not match `WATCHDOG_PID` (which was set to the original `ExecStart` PID). So the check returns `0` even though `NotifyAccess=all` would permit the notification. The fix is to bypass `sd_watchdog_enabled()` and instead check `WATCHDOG_USEC` directly, then call `sd_notify(0, "WATCHDOG=1")` unconditionally if the variable is non-empty.

## Corrective Actions
- Do not rely on `sd_watchdog_enabled()` in forking/daemonizing services where the final process PID differs from the ExecStart PID.
- Use `getenv("WATCHDOG_USEC")` directly to decide whether watchdog pings are expected; send via `sd_notify()` when non-empty.
- For Python asyncio services: schedule a periodic coroutine that calls `sd_notify(0, "WATCHDOG=1")` every `WATCHDOG_USEC/2` microseconds when the env var is set.
- Smoke test watchdog behavior after any service restructuring that changes fork depth.

## Key Takeaway
`sd_watchdog_enabled()` checks PID identity, not permission — in forking services, check `WATCHDOG_USEC` directly and call `sd_notify()` yourself rather than relying on the helper returning 1.
