# Lesson: Application Shutdown Is Not Idempotent — Concurrent Calls Execute Lifecycle Hooks Multiple Times

**Date:** 2026-02-28
**System:** community (nestjs/nest)
**Tier:** lesson
**Category:** integration
**Keywords:** idempotent, shutdown, lifecycle hooks, onModuleDestroy, SIGTERM, concurrent close, NestJS, TypeScript
**Source:** https://github.com/nestjs/nest/issues/16439

---

## Observation (What Happened)

`NestApplicationContext.close()` was not idempotent. When `enableShutdownHooks()` triggered a shutdown via OS signal AND application code also called `close()` concurrently (e.g., from a health check timeout), all lifecycle hooks (`onModuleDestroy`, `beforeApplicationShutdown`, `onApplicationShutdown`) executed twice. This could cause double-flushing of queues, double-closing of DB connections, or duplicate Telegram/webhook notifications on shutdown.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `NestApplicationContext.close()` had no guard against concurrent calls — each call triggered the full lifecycle chain independently.
**Why #2:** `enableShutdownHooks()` set a `receivedSignal` flag inside its own closure, but a direct `.close()` call bypassed that flag.
**Why #3:** The `NestMicroservice` version of `close()` already guarded with an `isTerminated` flag — the pattern existed but wasn't applied to `NestApplicationContext`.
**Why #4:** Concurrent shutdown is common in production: process signals race with graceful shutdown calls in request handlers or health checks.
**Why #5:** The lifecycle hooks had external side effects (DB disconnect, queue flush) that are dangerous to run twice.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add an `isTerminated` (or `isClosing`) boolean flag to any close/shutdown method — set it on first call, return early on subsequent calls | proposed | community | nestjs#16439 |
| 2 | Use a single Promise (`this._closing = this._closing ?? this._doClose()`) to coalesce concurrent calls | proposed | community | nestjs#16439 |
| 3 | Make lifecycle hooks idempotent where possible (check connection state before disconnecting) | proposed | community | nestjs#16439 |

## Key Takeaway

Any `close()` or `shutdown()` method with external side effects must be idempotent — use a guard flag or a cached Promise to ensure lifecycle hooks run exactly once even when called concurrently from signals and application code.
