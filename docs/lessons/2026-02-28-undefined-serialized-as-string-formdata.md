# Lesson: FormData.append() Serializes undefined as the String "undefined"

**Date:** 2026-02-28
**System:** community (honojs/hono)
**Tier:** lesson
**Category:** integration
**Keywords:** FormData, append, undefined, string serialization, RPC client, query params, TypeScript, hono
**Source:** https://github.com/honojs/hono/issues/4731

---

## Observation (What Happened)

Hono's RPC client had inconsistent `undefined` handling: query params correctly skipped `undefined` values via an explicit guard (`if (v === void 0) { continue; }`), but form data used `FormData.append(key, value)` without the same guard. Result: `undefined` values in form submissions were sent to the server as the literal string `"undefined"` instead of being omitted.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `FormData.append(key, value)` calls `String(value)` internally — `String(undefined)` returns `"undefined"`.
**Why #2:** The query param and form data serialization paths were implemented independently, each adding its own `undefined` skip logic — but the form path missed it.
**Why #3:** No integration test verified that an `undefined` form value was omitted rather than serialized.
**Why #4:** TypeScript types allowed passing `undefined` to the form object at call sites — the type system did not enforce "no undefined values in form data."
**Why #5:** The difference between "undefined means skip" (HTTP convention) and "undefined means string 'undefined'" (FormData behavior) is a footgun nowhere documented at the API boundary.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add explicit `undefined` guard in form data loop: `if (v === undefined) continue;` | proposed | community | hono#4731 |
| 2 | Write a cross-param-type test: same undefined value → query skips, form skips, JSON omits | proposed | community | hono#4731 |
| 3 | At API boundaries that accept user data, serialize through a single utility that handles undefined consistently | proposed | community | hono#4731 |

## Key Takeaway

`FormData.append(key, undefined)` sends the literal string `"undefined"` to the server — always guard undefined values before appending to FormData, just as you would before constructing query strings.
