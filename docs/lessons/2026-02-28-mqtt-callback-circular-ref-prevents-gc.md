# Lesson: paho Callback Assignment Creates Circular Reference That Prevents Garbage Collection

**Date:** 2026-02-28
**System:** community (eclipse-paho/paho.mqtt.python)
**Tier:** lifecycle
**Category:** lifecycle
**Keywords:** mqtt, paho, callback, circular reference, garbage collection, loop_start, thread, memory leak, destructor
**Source:** https://github.com/eclipse/paho.mqtt.python/issues/812

---

## Observation (What Happened)

A wrapper class assigned `self._on_connect`, `self._on_message`, and `self._on_publish` as paho callbacks, then started the loop with `loop_start()`. After the wrapper's scope exited, `gc.collect()` showed the wrapper was still alive; `__del__` was never called and the background thread never stopped. The leak accumulated silently over thousands of API requests.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `__del__` was never called, background thread never stopped.
**Why #2:** Python's GC cannot collect the wrapper because a live reference chain exists: `loop_start()` thread → paho Client object → `client.on_message` (bound method) → `self` (the wrapper instance).
**Why #3:** Assigning a bound method to `client.on_message` creates a reference from the paho Client to the wrapper instance. Since the paho Client is held alive by the network thread, the wrapper can never be collected.
**Why #4:** Python's GC for objects with circular references or references from live threads does not guarantee timely collection — in CPython, `__del__` on objects in reference cycles is explicitly deferred.
**Why #5:** The design relied on `__del__` for resource cleanup (calling `loop_stop()`), which is explicitly discouraged by Python — `__del__` cannot be called when a reference cycle exists involving the instance.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Implement the wrapper as a context manager (`__enter__`/`__exit__`) and call `client.loop_stop()` and `client.disconnect()` in `__exit__` | proposed | community | https://github.com/eclipse/paho.mqtt.python/issues/812 |
| 2 | Never rely on `__del__` for MQTT cleanup — always use explicit `close()` / `shutdown()` called at a deterministic point | proposed | community | Python docs on object finalization |
| 3 | If using `loop_start()`, always pair it with `loop_stop()` in a `finally` block or context manager | proposed | community | paho docs |

## Key Takeaway

Assigning bound methods as paho callbacks creates a reference from the paho Client back to your object — never rely on `__del__` for cleanup; use a context manager or explicit `shutdown()` with `loop_stop()`.
