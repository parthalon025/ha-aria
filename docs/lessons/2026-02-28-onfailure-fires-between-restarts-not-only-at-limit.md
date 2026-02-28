# Lesson: OnFailure= Fires Between Auto-Restarts, Not Just After StartLimitBurst Is Hit
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** reliability
**Keywords:** systemd, OnFailure, Restart, StartLimitBurst, failed state, notification, alert flooding, service dependency
**Source:** https://github.com/systemd/systemd/issues/27594
---
## Observation (What Happened)
A service was configured with `Restart=on-failure` and `OnFailure=alert-notify.service`. Operators expected the alert to fire only when the service exhausted its restart budget (`StartLimitBurst`). Instead, the alert fired on every single restart attempt — including the intermediate ones — resulting in alert flooding and false-incident escalation.

## Analysis (Root Cause — 5 Whys)
The systemd documentation states that a service with `Restart=` "enters the failed state only after start limits are reached." This was a known gap between documentation and implementation in versions prior to v254. `OnFailure=` was triggered each time the service transitioned through a failed state during restart cycling, not only at the terminal failure. The fix (merged in v254) corrected the transition logic so `OnFailure=` only fires after the start limit is exhausted. On older systemd versions this behavior is the actual runtime contract regardless of documentation.

## Corrective Actions
- On systemd < 254: do not use `OnFailure=` for high-signal alerting — use a separate watchdog or external health monitor that checks service state after a delay.
- Verify systemd version with `systemctl --version` before relying on documented `OnFailure=` semantics.
- As an alternative: use `ExecStopPost=` with `$SERVICE_RESULT` check (`if [ "$SERVICE_RESULT" = "start-limit-hit" ]`) to trigger alerts only on limit exhaustion.
- Add a test: after deploying alerting integrations based on `OnFailure=`, deliberately crash a non-critical service and verify alert count == 1, not N.

## Key Takeaway
On systemd < v254, `OnFailure=` fires on every restart cycle, not just at start-limit exhaustion — use `ExecStopPost=` with `$SERVICE_RESULT` for reliable once-per-limit-hit alerting.
