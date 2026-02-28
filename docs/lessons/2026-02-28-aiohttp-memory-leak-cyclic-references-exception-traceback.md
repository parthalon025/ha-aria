# Lesson: aiohttp Server and Client Objects Accumulate in Memory Due to Exception Traceback Cyclic References
**Date:** 2026-02-28
**System:** community (aio-libs/aiohttp)
**Tier:** lesson
**Category:** reliability
**Keywords:** aiohttp, memory leak, cyclic reference, traceback, gc, ClientResponse, Request, garbage collector
**Source:** https://github.com/aio-libs/aiohttp/issues/10548
---
## Observation (What Happened)
Long-running aiohttp servers and clients show steadily growing memory usage traceable to `Request` and `ClientResponse` objects that are never released by the garbage collector. `objgraph` reveals that these objects form cyclic reference chains through exception tracebacks, preventing CPython's reference-count-based collector from releasing them. Memory growth is proportional to request rate and is most visible in image-processing or large-body handlers.

## Analysis (Root Cause — 5 Whys)
Python's exception mechanism stores local variable frames in the traceback. When an exception is raised inside a request handler, the traceback object holds a reference to the local frame, which holds a reference to `request`, which holds references back through the response cycle. This creates a cycle that CPython's reference counter alone cannot collect — it requires the cyclic garbage collector (`gc.collect()`). In production, `gc.collect()` runs infrequently, so these cycles accumulate. The fundamental issue is that aiohttp's request objects participate in exception frames without breaking the cycle on cleanup.

## Corrective Actions
- In long-running aiohttp server handlers that process large request bodies, explicitly delete local references after use: `del request, data` before the return, or use `try/finally` to clear `request._payload` after reading.
- Enable `gc.collect()` at regular intervals (e.g., via a periodic asyncio task) for high-throughput services processing large bodies.
- In ARIA's hub, any route that reads large JSON payloads from HA (e.g., history bulk fetch) should clear local response objects after use to avoid cyclic accumulation.
- Monitor with `tracemalloc` in staging — filter for `aiohttp/web_request.py` growth as an early warning.

## Key Takeaway
Exception tracebacks in Python create cyclic references through local frames — in aiohttp handlers that raise exceptions, `Request` and `ClientResponse` objects can accumulate indefinitely without periodic `gc.collect()`.
