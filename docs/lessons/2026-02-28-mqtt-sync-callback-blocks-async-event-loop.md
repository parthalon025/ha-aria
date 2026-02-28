# Lesson: Synchronous Blocking Code in async on_message Callback Stalls the Entire Event Loop

**Date:** 2026-02-28
**System:** community (wialon/gmqtt, sabuhish/fastapi-mqtt)
**Tier:** async
**Category:** async
**Keywords:** mqtt, gmqtt, fastapi-mqtt, async, on_message, blocking, time.sleep, event loop, stall, uvicorn
**Source:** https://github.com/wialon/gmqtt/issues/146

---

## Observation (What Happened)

A FastAPI + gmqtt application processed MQTT messages with `time.sleep(0.4)` inside an `async def on_message` handler. At message rates above 2/second, all HTTP endpoints became unresponsive — requests piled up and timed out. The sleep was a placeholder for a database write.

## Analysis (Root Cause — 5 Whys)

**Why #1:** HTTP endpoints stopped responding while MQTT messages were being processed.
**Why #2:** `async def on_message` was called directly on the asyncio event loop thread. Despite being declared `async`, the function body contained `time.sleep(0.4)` — a blocking call that holds the GIL and yields no control to the event loop.
**Why #3:** asyncio's cooperative concurrency requires that every coroutine either `await` a non-blocking operation or return quickly. A blocking `time.sleep()` inside `async def` does not suspend the coroutine — it freezes the entire event loop for its duration.
**Why #4:** The developer assumed that `async def` automatically makes a function non-blocking. It does not — only `await asyncio.sleep()`, `await` on I/O primitives, or `asyncio.to_thread()` yield control.
**Why #5:** gmqtt/aiomqtt call `on_message` directly from the event loop if it is declared `async`, making callback blocking latency directly visible to all other coroutines sharing that loop.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace `time.sleep()` with `await asyncio.sleep()` for artificial delays | proposed | community | https://github.com/wialon/gmqtt/issues/146 |
| 2 | For CPU-bound or blocking I/O in `on_message`, use `await asyncio.to_thread(blocking_func, ...)` to run on a thread pool | proposed | community | Python asyncio docs |
| 3 | Use async database drivers (e.g., `asyncpg`, `motor`) instead of synchronous drivers inside async callbacks | proposed | community | issue #146 comment |
| 4 | For long-running handlers, enqueue to an `asyncio.Queue` in `on_message` and process in a separate task | proposed | community | general async pattern |

## Key Takeaway

`async def on_message` with any blocking call (`time.sleep`, sync DB writes, file I/O) freezes the entire event loop — use `await asyncio.to_thread()` or async equivalents for all non-trivial work inside MQTT message handlers.
