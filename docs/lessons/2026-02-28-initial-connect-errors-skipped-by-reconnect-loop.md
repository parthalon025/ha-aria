# Lesson: Initial Connection Errors Bypass Reconnect Logic — Cold Start Silently Fails

**Date:** 2026-02-28
**System:** community (miguelgrinberg/python-socketio)
**Tier:** lesson
**Category:** integration
**Keywords:** Redis, reconnect, initial connection, error handling, startup, cold start, queue subscription, constructor
**Source:** https://github.com/miguelgrinberg/python-socketio/issues/1534

---

## Observation (What Happened)

The python-socketio Redis manager had robust reconnection logic for failures that occurred during steady-state operation, but errors in the constructor-phase connection and the initial `listen()` subscription were not handled at all. If Redis was unavailable at startup, the service appeared to start but was silently non-functional — no retry, no error surface.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The service started successfully but never received any messages from the Redis pub/sub channel.
**Why #2:** The initial subscription in `listen()` raised an exception during startup, but there was no try/except at that call site.
**Why #3:** The reconnect loop only applied to failures after the first successful subscription, not before it.
**Why #4:** The initial connect path was written before the reconnect loop existed and was never updated to use the same resilience pattern.
**Why #5:** Cold-start error handling and steady-state reconnect logic were designed as separate code paths, creating a gap at the most vulnerable moment: first connection.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Treat the initial connection/subscription as the first iteration of the reconnect loop, not a separate code path | proposed | community | issue #1534 |
| 2 | Any service that subscribes to an external channel (Redis, MQTT, WebSocket) must wrap its `initialize()` / `listen()` startup in the same retry logic used during steady-state reconnection | proposed | community | issue #1534 |
| 3 | Add a startup health check that verifies the subscription is active before declaring the service ready | proposed | community | issue #1534 |

## Key Takeaway

Cold-start and steady-state failure handling must share the same retry path — any service that only handles reconnect failures after the first success has a silent gap exactly when it is most vulnerable: service startup.
