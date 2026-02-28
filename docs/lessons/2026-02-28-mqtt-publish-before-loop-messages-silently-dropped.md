# Lesson: MQTT publish() Returns Before Message Leaves the Socket — Exit Without loop.stop() Drops Messages

**Date:** 2026-02-28
**System:** community (eclipse-paho/paho.mqtt.python)
**Tier:** lesson
**Category:** reliability
**Keywords:** mqtt, paho, publish, loop_start, loop_stop, message loss, exit, async, queue
**Source:** https://github.com/eclipse/paho.mqtt.python/issues/741

---

## Observation (What Happened)

A script called `client.publish(topic, payload, qos=0)`, received the `on_publish` callback confirming the message was "sent," then exited. The broker never received the message. Adding `time.sleep(1)` after `publish()` fixed it consistently.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Messages were not reaching the broker despite `on_publish` firing.
**Why #2:** `on_publish` fires when the message is placed in the outbound queue, not when the TCP socket has flushed the bytes.
**Why #3:** The actual network I/O happens on paho's background network thread started by `loop_start()`. If the main thread exits before that thread flushes the queue, the OS closes the socket and the queued bytes are discarded.
**Why #4:** There is no built-in "drain" call between `publish()` and `disconnect()` or program exit.
**Why #5:** paho's API conflates "I accepted your message" (`on_publish` fires) with "your message left this host" (requires loop to flush).

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `mqtt.publish.single()` or `mqtt.publish.multiple()` for fire-and-exit patterns — they manage the loop lifecycle internally | proposed | community | paho docs |
| 2 | After `client.publish()`, call `client.loop_stop()` and wait for the message to be acknowledged before exiting | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/741 |
| 3 | For QoS > 0, use `MQTTMessageInfo.wait_for_publish()` to block until the broker acknowledges | proposed | community | paho 1.6+ API |

## Key Takeaway

`client.publish()` enqueues a message; the background thread sends it — program exit before `loop_stop()` silently discards every enqueued message regardless of what `on_publish` reported.
