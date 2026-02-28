# Lesson: aiomqtt Client Instance Cannot Be Reused Across Reconnects — Must Reinitialize Inside Retry Loop

**Date:** 2026-02-28
**System:** community (empicano/aiomqtt)
**Tier:** reliability
**Category:** reliability
**Keywords:** mqtt, aiomqtt, reconnect, reentrant, context manager, client, async with, retry loop, reuse
**Source:** https://github.com/empicano/aiomqtt/issues/244

---

## Observation (What Happened)

The official aiomqtt reconnection example created a single `client = aiomqtt.Client(...)` outside a retry loop, then called `async with client:` in a loop on reconnect. After the first disconnection, every subsequent reconnect attempt failed immediately with `"Does not support reentrant"` — the client would never reconnect regardless of whether the broker was available.

## Analysis (Root Cause — 5 Whys)

**Why #1:** All reconnect attempts after the first failure immediately raised `"Does not support reentrant"`.
**Why #2:** `aiomqtt.Client` is implemented as a reusable but not reentrant context manager (per Python contextlib terminology). Calling `async with client:` a second time after `__aexit__` does not reset the internal lock state.
**Why #3:** When `__aexit__` runs after an error, an internal asyncio lock is released but the state machine is not reset to "ready for re-entry." A second `async with` call sees the lock as already held.
**Why #4:** The reconnection pattern in the docs placed `client = aiomqtt.Client(...)` outside the retry loop, intending to reuse the same instance — but the implementation does not support this.
**Why #5:** The distinction between "reusable" (can be entered again after exit) and "reentrant" (can be entered while already entered) was not clearly communicated in the API, making the connection pattern ambiguous.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Instantiate `aiomqtt.Client(...)` inside the retry loop so a fresh instance is created for each connection attempt | proposed | community | https://github.com/empicano/aiomqtt/issues/244 |
| 2 | The fix: `while True: try: async with aiomqtt.Client(...) as client: ...` — not `client = ...; while True: async with client:` | proposed | community | issue #244 |

## Key Takeaway

aiomqtt `Client` instances cannot be reused across reconnects — instantiate a new `Client` inside the retry loop, not outside it, or the second connection attempt will always raise `"Does not support reentrant"`.
