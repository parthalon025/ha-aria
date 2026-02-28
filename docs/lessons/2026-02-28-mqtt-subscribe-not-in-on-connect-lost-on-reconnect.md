# Lesson: MQTT Subscriptions Outside on_connect Are Lost on Reconnect

**Date:** 2026-02-28
**System:** community (eclipse-paho/paho.mqtt.python)
**Tier:** lesson
**Category:** reliability
**Keywords:** mqtt, paho, subscribe, reconnect, on_connect, clean_session, subscriptions, lost
**Source:** https://github.com/eclipse/paho.mqtt.python/issues/682

---

## Observation (What Happened)

A publisher and subscriber both set to auto-reconnect lost all subscriptions after the broker rebooted. Messages published after reconnect were never delivered to the subscriber, with no error or warning.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Subscriber stopped receiving messages after reconnect.
**Why #2:** Subscriptions were called once at startup, outside any `on_connect` callback.
**Why #3:** MQTT subscriptions are session-state on the broker. When the connection drops and `clean_session=True` (the default), the broker discards all session state including subscriptions.
**Why #4:** Paho's `client.subscribe()` sends a SUBSCRIBE packet to the broker. When the connection is re-established, the library reconnects the TCP layer but does not replay subscription commands — that is the application's responsibility.
**Why #5:** No distinction between "one-time initialization" and "per-connection setup" in the application code. `subscribe()` was treated as a fire-and-forget at startup.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Move all `client.subscribe()` calls into the `on_connect` callback | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/682 |
| 2 | Alternatively use `clean_session=False` + `session_expiry_interval` (MQTTv5) so the broker maintains subscriptions across reconnects | proposed | community | MQTT spec 3.2.2.2 |

## Key Takeaway

`client.subscribe()` at startup is silently ineffective after any reconnect — subscriptions belong in `on_connect`, which fires on every connection including reconnects.
