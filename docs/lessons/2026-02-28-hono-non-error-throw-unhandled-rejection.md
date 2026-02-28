# Lesson: Throwing Non-Error Values Bypasses instanceof-Gated Error Handlers

**Date:** 2026-02-28
**System:** community (honojs/hono)
**Tier:** lesson
**Category:** error-handling
**Keywords:** unhandled promise rejection, throw, non-Error, instanceof, middleware, error handler, TypeScript, hono
**Source:** https://github.com/honojs/hono/issues/4708

---

## Observation (What Happened)

Hono's middleware composition only caught exceptions that were `instanceof Error`. When middleware threw a string, number, or plain object (`throw "auth failed"` or `throw { status: 401 }`), the framework's `onError` handler was bypassed entirely, causing an unhandled promise rejection that could crash the Node.js/Bun process.

## Analysis (Root Cause — 5 Whys)

**Why #1:** JavaScript allows throwing any value — `throw "string"`, `throw 42`, `throw {code: 400}` are all valid.
**Why #2:** The error handler used `if (err instanceof Error)` to gate processing — this silently passes non-Error throws through uncaught.
**Why #3:** TypeScript's `catch (e)` types `e` as `unknown` since TS 4.0, but the handler narrowed without an exhaustive fallback.
**Why #4:** Framework authors assumed "good citizens throw Error instances" — a reasonable assumption violated by third-party middleware.
**Why #5:** The unhandled rejection is only fatal on Node.js 15+ (previously just a warning), so this was invisible on older runtimes.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Error handlers must catch `unknown`, not `Error` — use `if (err instanceof Error) ... else { /* handle non-Error */ }` | proposed | community | hono#4708 |
| 2 | In TypeScript, always type catch bindings as `unknown` and narrow explicitly | proposed | community | hono#4708 |
| 3 | Never rely on framework error boundaries for non-Error throws — add lint rule `@typescript-eslint/no-throw-literal` | proposed | community | hono#4708 |

## Key Takeaway

`catch (e) { if (e instanceof Error) }` silently swallows non-Error throws as unhandled rejections — error handlers must treat the caught value as `unknown` and handle both Error and non-Error cases explicitly.
