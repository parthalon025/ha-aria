# Lesson: HTTP Client Libraries Log Successful Requests at INFO Level — Floods Production Logs

**Date:** 2026-02-28
**System:** community (ollama/ollama-python)
**Tier:** lesson
**Category:** performance
**Keywords:** logging, httpx, httpcore, INFO, DEBUG, noise, production-logs, ollama, http-client, log-level
**Source:** https://github.com/ollama/ollama-python/issues/540

---

## Observation (What Happened)

Every Ollama API call generated an INFO-level log entry from httpx: `HTTP Request: POST http://<host>:11434/api/generate "HTTP/1.1 200 OK"`. In production this flooded logs with noise, obscuring actual application-level INFO events and making log-based alerting unreliable.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Production logs were full of low-value HTTP success lines.
**Why #2:** `httpx` logs successful requests at INFO level by default.
**Why #3:** The ollama-python library did not suppress or downgrade httpx logging on initialization.
**Why #4:** The library author assumed callers would configure logging as needed — but most users don't know which internal logger to suppress.
**Why #5:** No guidance was provided in the docs on expected log volume or how to reduce it.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | In application code, set `logging.getLogger("httpx").setLevel(logging.WARNING)` to suppress success lines | proposed | community | issue #540 |
| 2 | Similarly suppress `logging.getLogger("httpcore").setLevel(logging.WARNING)` | proposed | community | issue #540 |
| 3 | Libraries wrapping HTTP clients should downgrade the client's log level on init unless the user opts into verbose logging | proposed | community | issue #540 |
| 4 | Add to standard project logging setup: silence known-noisy third-party loggers by name | proposed | community | operational pattern |

## Key Takeaway

`httpx` and `httpcore` log every successful HTTP request at INFO level — any application making frequent LLM API calls must explicitly set `logging.getLogger("httpx").setLevel(WARNING)` and `logging.getLogger("httpcore").setLevel(WARNING)` to prevent these from flooding production logs.
