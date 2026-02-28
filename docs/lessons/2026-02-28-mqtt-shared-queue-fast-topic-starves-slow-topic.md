# Lesson: Single aiomqtt Message Queue With queue_maxsize Causes Fast Topic to Silently Drop Slow Topic Messages

**Date:** 2026-02-28
**System:** community (empicano/aiomqtt)
**Tier:** lesson
**Category:** reliability
**Keywords:** mqtt, aiomqtt, queue, backpressure, message loss, queue_maxsize, shared queue, topic, consumer, slow consumer
**Source:** https://github.com/empicano/aiomqtt/issues/250

---

## Observation (What Happened)

An application subscribed to a high-frequency `fast` topic and a low-frequency `slow` topic using `client.messages(queue_maxsize=10)`. The `slow` consumer reliably missed messages even though the slow topic published far less data than the queue capacity.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `slow` topic messages were silently dropped.
**Why #2:** `queue_maxsize` applies to the single global inbound queue shared by all subscriptions. When the `fast` topic fills the queue, the broker-side puts for `slow` topic messages hit the limit and are dropped.
**Why #3:** The queue is shared because aiomqtt (pre-2.0) used a single `asyncio.Queue` for all inbound messages regardless of topic. The `queue_maxsize` parameter is therefore a global constraint, not a per-topic constraint.
**Why #4:** When two consumers iterate the same queue, topic-matching filters inside each `async for` loop discard mismatched messages — but those discarded messages already consumed queue slots.
**Why #5:** The abstraction made `queue_maxsize` look like a per-consumer config, but it was actually a global side-effect — this is the kind of implicit shared state that creates non-deterministic message loss.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use a dispatcher pattern: one consumer reads all messages and routes to per-topic `asyncio.Queue` instances | proposed | community | https://github.com/empicano/aiomqtt/issues/250 comment |
| 2 | Do not set `queue_maxsize` when using topic-filter patterns across multiple concurrent consumers | proposed | community | issue #250 |
| 3 | In aiomqtt 2.0+, `Client.messages` was changed to a single client-wide queue — set `queue_maxsize` on the client, not the messages call | proposed | community | aiomqtt changelog |
| 4 | For slow consumers, use a separate client instance per consumer if per-topic queue isolation is required | proposed | community | issue #250 |

## Key Takeaway

`queue_maxsize` in MQTT client libraries is typically a global constraint on the single inbound queue — a high-frequency topic will starve slow-topic consumers by filling the shared queue, causing silent message drops.
