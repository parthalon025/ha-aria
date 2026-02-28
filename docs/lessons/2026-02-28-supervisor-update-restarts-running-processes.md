# Lesson: supervisorctl update Restarts Running Processes Even on Non-Command Config Changes
**Date:** 2026-02-28
**System:** community (Supervisor/supervisor)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** supervisord, supervisorctl update, restart, config change, autorestart, in-place config, production disruption
**Source:** https://github.com/Supervisor/supervisor/issues/1637
---
## Observation (What Happened)
An operator changed only `autorestart=false` in a supervisord program config (no change to `command=`, `directory=`, or any execution parameter) and ran `supervisorctl update` to apply the change. The running process was immediately restarted, resetting uptime to 0. The operator expected that changing a supervisory policy setting (not the command itself) would not interrupt a running process, especially for a critical long-running service.

## Analysis (Root Cause — 5 Whys)
`supervisorctl update` compares the current and new config for any difference. If any config field changes — including supervisory-only settings like `autorestart`, `startretries`, or `priority` — supervisor treats the entire program as modified and performs a stop/start cycle. This is by design: supervisor has no facility to distinguish "live-applicable" config changes from "requires-restart" changes. The result is that even policy-only changes cause process disruption.

## Corrective Actions
- Before running `supervisorctl update` in production, audit what changed: if only `autorestart`, `priority`, or `startretries` changed, defer the update to a maintenance window.
- For live config changes that cannot cause restarts: use `supervisorctl pid <prog>` to get the PID and signal the process directly, bypassing supervisor's update path.
- Build a diff-check wrapper: `diff <(supervisorctl avail) <(cat new.conf)` to preview what `update` will change before executing.
- In ARIA: systemd's `systemctl daemon-reload` has the same class of issue for some directives — always verify which changes require restart vs. hot-apply.

## Key Takeaway
`supervisorctl update` restarts processes on any config diff, including policy-only changes — audit the diff before running in production and use maintenance windows for policy updates.
