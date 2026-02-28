# Lesson: TimeoutStopSec Is Overridden by DefaultTimeoutStopSec When Service Has ExecStop
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** systemd, TimeoutStopSec, ExecStop, SIGKILL, graceful shutdown, DefaultTimeoutStopSec, stop timeout, override
**Source:** https://github.com/systemd/systemd/issues/31288
---
## Observation (What Happened)
A service with `TimeoutStopSec=400` and an `ExecStop=` command was being SIGKILL'd after 90 seconds, not 400. The `ExecStop=` script needed ~173 seconds to complete a graceful shutdown. Operators set `TimeoutStopSec=400` expecting systemd to wait that long before sending SIGKILL. The service was killed prematurely.

## Analysis (Root Cause — 5 Whys)
When a service has both `ExecStop=` and `TimeoutStopSec=`, the timeout applies to the full stop sequence: first `ExecStop=` runs, then systemd sends SIGTERM to the main process, then waits again. However, the system-wide `DefaultTimeoutStopSec=90s` (the default in most distributions) acts as a cap if the per-unit value was not read correctly or if the system configuration overrides it. Additionally, on older systemd versions (pre-v239), per-unit `TimeoutStopSec=` was not always respected when `ExecStop=` was present. The root cause is ambiguity in which timeout governs which phase of the stop sequence.

## Corrective Actions
- Set `TimeoutStopSec=` explicitly in the unit AND verify it via `systemctl show <service> --property=TimeoutStopUSec`.
- Increase `DefaultTimeoutStopSec=` in `/etc/systemd/system.conf` if many services need longer shutdown windows.
- For graceful-shutdown services: set `TimeoutStopSec=` to at least 2x the expected max drain time; add `KillMode=mixed` so the main process gets SIGTERM first, worker processes get SIGKILL after timeout.
- Smoke test: `time systemctl stop <service>` — confirm duration matches expectation before deploying to production.

## Key Takeaway
`TimeoutStopSec=` must be verified at runtime via `systemctl show`, not just set in the unit file — system defaults can silently cap it, and the stop sequence timeout semantics differ when `ExecStop=` is present.
