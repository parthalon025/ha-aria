# Lesson: Socket Activation Can Silently Break After systemd Version Upgrades
**Date:** 2026-02-28
**System:** community (systemd/systemd)
**Tier:** lesson
**Category:** reliability
**Keywords:** systemd, socket activation, regression, version upgrade, libvirtd, service start, timeout, silent hang
**Source:** https://github.com/systemd/systemd/issues/27953
---
## Observation (What Happened)
After upgrading from systemd 253.4 to 253.5, socket-activated services stopped responding after their idle timeout. Client processes connecting to the socket hung indefinitely with no errors in logs or any diagnostic output. Rolling back to 253.4 restored correct behavior. The regression was invisible — no failed units, no error messages, just silent hangs on client connects.

## Analysis (Root Cause — 5 Whys)
A regression introduced in systemd 253.5 broke the socket activation handoff path for services that use idle timeout + on-demand respawn (the standard socket activation pattern). After the service process exited due to inactivity, the socket unit was supposed to accept new connections and re-trigger the service unit. The regression caused the socket's activation state machine to stall without triggering a service restart or emitting a log entry. Silent failure is especially dangerous here because there is no negative signal — clients block on `connect()` which appears to be in progress.

## Corrective Actions
- Pin systemd version in production and test socket activation end-to-end after every OS upgrade before deploying.
- Add a socket activation smoke test: let the service idle-timeout, then issue a client request and assert it succeeds within N seconds.
- Monitor socket-activated services with a dedicated health probe on a fixed interval rather than relying on systemd's `is-active` (which reports the socket unit as active even when activation is broken).
- For Python asyncio services using socket activation: test the full idle→connect cycle in CI using a test socket and a child process.

## Key Takeaway
Socket activation regressions produce silent client hangs, not error logs — always include an end-to-end socket activation round-trip test after any systemd version upgrade.
