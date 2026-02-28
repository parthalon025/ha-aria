# Lesson: MQTT Client in Multi-Worker Process Deployment Duplicates Every Subscription — N Workers = N Deliveries

**Date:** 2026-02-28
**System:** community (sabuhish/fastapi-mqtt)
**Tier:** integration
**Category:** integration
**Keywords:** mqtt, fastapi, gunicorn, uvicorn, multiprocess, multiple workers, subscription, duplicate, broker, fanout
**Source:** https://github.com/sabuhish/fastapi-mqtt/issues/24

---

## Observation (What Happened)

A FastAPI application with an MQTT subscription worked correctly with a single worker. After deploying in a Docker container using the standard `tiangolo/uvicorn-gunicorn-fastapi` image (which starts 8 worker processes automatically), each incoming MQTT message triggered `on_message` 8 times — once per worker process.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Every MQTT message was processed 8 times.
**Why #2:** Each worker process independently instantiated the MQTT client and called `subscribe()` at startup. The broker treats each connection as a distinct client with the same subscription.
**Why #3:** `gunicorn`/`uvicorn` multi-worker mode forks the process N times. Each fork is an independent OS process with its own network connections — there is no shared MQTT client state between workers.
**Why #4:** MQTT subscription is per-client-connection at the broker level. When N clients all subscribe to the same topic, the broker delivers to all N — this is correct MQTT behavior, not a bug.
**Why #5:** The application was designed as if MQTT subscriptions were a global singleton, but they are per-process in a forking deployment model.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Run MQTT in a dedicated single-process service separate from the HTTP workers | proposed | community | https://github.com/sabuhish/fastapi-mqtt/issues/24 |
| 2 | Use a message queue (Redis pub/sub, in-process asyncio.Queue) as the bridge — single MQTT subscriber fans out to HTTP workers via the queue | proposed | community | general pattern |
| 3 | Pin MQTT subscription to a single named worker using a coordination primitive (file lock, Redis lock) — only the worker holding the lock subscribes | proposed | community | general pattern |
| 4 | Run uvicorn with `--workers 1` for MQTT-heavy services | proposed | community | uvicorn docs |

## Key Takeaway

MQTT subscriptions in multi-process deployments are not singletons — every forked worker creates an independent broker subscription, delivering N copies of every message; MQTT clients must run in a single-process context or behind an explicit coordination layer.
