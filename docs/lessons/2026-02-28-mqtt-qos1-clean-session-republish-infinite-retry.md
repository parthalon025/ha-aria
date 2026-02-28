# Lesson: QoS ≥ 1 + clean_session=False Creates Infinite Retry Storm on Broker-Rejected Messages

**Date:** 2026-02-28
**System:** community (eclipse-paho/paho.mqtt.python)
**Tier:** lesson
**Category:** reliability
**Keywords:** mqtt, paho, qos, clean_session, reconnect, retry, infinite loop, broker reject, out_messages queue
**Source:** https://github.com/eclipse/paho.mqtt.python/issues/760

---

## Observation (What Happened)

An IoT fleet using AWS IoT Core with QoS 1 and persistent sessions (`clean_session=False`) started consuming gigabytes of cellular upload per day. Root cause: one device sent a 200 KB message exceeding AWS's 128 KB quota. AWS returned a DISCONNECT. paho re-queued the unacknowledged message, reconnected, and retried — every ~3 seconds, indefinitely, until the device was manually restarted.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Cellular data exhausted in hours.
**Why #2:** Device was uploading the same oversized message in a tight loop.
**Why #3:** MQTT v3 has no NACK mechanism — the only way a broker can reject a message is to disconnect. paho interprets any disconnect as a transient error and retries all unacknowledged QoS ≥ 1 messages from the persistent outbound queue.
**Why #4:** No maximum message size check before `publish()`, and no logic to detect that a disconnect was caused by a specific message being rejected rather than a network failure.
**Why #5:** `clean_session=False` keeps the out_messages queue across reconnects by design — the retry is correct behavior for transient failures but catastrophic for permanent broker rejections.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Enforce payload size limit before `publish()` — e.g., `assert len(payload) < 128 * 1024` | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/760 |
| 2 | In `on_disconnect`, inspect `rc` and if abnormal, drain the out_messages queue to prevent infinite retry | proposed | community | issue #760 workaround |
| 3 | Consider MQTTv5 which supports PUBACK reason codes — broker can reject without disconnecting | proposed | community | MQTT v5 spec |
| 4 | Add a per-message retry counter; after N failures, drop the message and alert | proposed | community | general pattern |

## Key Takeaway

MQTT v3 QoS ≥ 1 + `clean_session=False` creates an infinite retry loop for any message the broker permanently rejects — validate payload constraints (size, schema) before publishing, not after.
