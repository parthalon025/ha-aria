# Lesson: MQTT Last Will Triggered on Graceful disconnect() Call — Not Just Unexpected Death

**Date:** 2026-02-28
**System:** community (wialon/gmqtt)
**Tier:** reliability
**Category:** reliability
**Keywords:** mqtt, will, last will, lwt, disconnect, graceful, retain, broker, gmqtt, MQTTv311
**Source:** https://github.com/wialon/gmqtt/issues/59

---

## Observation (What Happened)

A gmqtt client set a Will message (`payload="LOST"`, `retain=True`) and published `"OFFLINE"` before calling `await client.disconnect()`. The broker received all three messages in order: `ONLINE`, `OFFLINE`, `LOST`. The will message was delivered even though the client disconnected cleanly.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Will message (`LOST`) appeared in the broker log after a clean disconnect.
**Why #2:** The gmqtt library had a bug in the MQTT 3.1.1 DISCONNECT packet — it sent a malformed packet that the broker did not interpret as a clean disconnect.
**Why #3:** In MQTT, the broker only suppresses the Will message if it receives a valid DISCONNECT control packet before the connection closes. A malformed or missing DISCONNECT causes the broker to treat the close as an ungraceful termination and fire the Will.
**Why #4:** The `disconnect()` implementation did not correctly encode the DISCONNECT packet for MQTTv311 — fixed in gmqtt 0.4.1.
**Why #5:** Deeper: Will message semantics are easy to misunderstand — the Will fires on any close that is not preceded by a valid DISCONNECT packet, including crashes, `os.kill`, `SIGKILL`, network partition, and also bugs in the disconnect implementation.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Explicitly publish an `"offline"` status message *before* calling `disconnect()` as defense-in-depth — do not rely solely on the Will for presence tracking | proposed | community | https://github.com/wialon/gmqtt/issues/59 |
| 2 | Verify broker logs after implementing Will messages — confirm that Will fires only on unexpected termination, not on clean shutdowns | proposed | community | general pattern |
| 3 | Use `asyncio.CancelledError` handling to explicitly call `disconnect()` — if `CancelledError` is not caught, the context manager `__aexit__` may not run and Will may fire unexpectedly | proposed | community | aiomqtt issue #340 |
| 4 | In production presence tracking, use both Will message (offline detection) AND an explicit `"online"` heartbeat topic with a short retain TTL (MQTTv5) | proposed | community | general pattern |

## Key Takeaway

The MQTT Will message fires whenever the broker does not receive a valid DISCONNECT packet — including bugs in the client library's disconnect implementation; always verify Will behavior in broker logs after any client library update.
