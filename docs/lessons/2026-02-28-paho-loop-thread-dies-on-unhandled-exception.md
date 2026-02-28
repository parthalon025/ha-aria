# Lesson: paho loop_start() Thread Silently Dies on Any Unhandled Exception — All Future Messages Lost

**Date:** 2026-02-28
**System:** community (eclipse-paho/paho.mqtt.python)
**Tier:** lesson
**Category:** error-handling
**Keywords:** mqtt, paho, loop_start, thread, exception, uncaught, on_message, silent failure, thread death
**Source:** https://github.com/eclipse/paho.mqtt.python/issues/750, https://github.com/eclipse/paho.mqtt.python/issues/704

---

## Observation (What Happened)

A long-running paho client using `loop_start()` stopped receiving messages after a network outage or a malformed message payload. The process kept running with no errors logged. In a separate HA integration case, `on_message` raised `JSONDecodeError` on truncated data; the callback caught and logged the exception, but the paho background thread still died — all subsequent messages were silently dropped.

## Analysis (Root Cause — 5 Whys)

**Why #1:** MQTT client stopped receiving messages mid-run with no error.
**Why #2:** `loop_start()` runs paho's network loop in a daemon thread via `_thread_main → loop_forever`. An unhandled exception (SSL EOF, unexpected packet format, or any exception that escapes the socket read path) propagates up and kills that thread.
**Why #3:** Python's threading model does not propagate thread exceptions to the main thread. The thread simply exits with a stack trace to stderr — which is often redirected or ignored in production.
**Why #4:** The application only wrapped `on_message` in try/except, not the lower-level `loop_write`/`loop_read` path where socket-level exceptions escape.
**Why #5:** paho's internal exception handling in `_packet_write` does not catch all exception types (e.g., `ssl.SSLEOFError`), so network errors can bubble out of `loop_forever` entirely.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `threading.excepthook` or a watchdog that checks `client._thread.is_alive()` and calls `client.reconnect()` if the thread has died | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/750 |
| 2 | Prefer `loop_forever()` in a monitored main thread over `loop_start()` — exceptions from `loop_forever` are catchable in the calling thread | proposed | community | paho docs |
| 3 | Wrap any socket-error-prone code paths and call `client.disconnect()` + `client.reconnect()` in an outer retry loop | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/704 |
| 4 | Set up `client.on_disconnect` to log and trigger reconnect on `rc != 0` | proposed | community | paho docs |

## Key Takeaway

`loop_start()` thread death is fully silent — a liveness check on `client._thread.is_alive()` is the only way to detect it; there is no callback or exception for the main thread to catch.
