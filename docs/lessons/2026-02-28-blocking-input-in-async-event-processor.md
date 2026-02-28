# Lesson: Blocking I/O in Async Event Handler Halts Entire Event Loop

**Date:** 2026-02-28
**System:** community (ag2ai/ag2)
**Tier:** lesson
**Category:** async
**Keywords:** async, blocking I/O, event loop, input(), getpass, run_in_executor, asyncio, multi-agent, agent framework
**Source:** https://github.com/ag2ai/ag2/issues/2110

---

## Observation (What Happened)

`AsyncConsoleEventProcessor.process_event` called the blocking `input()` and `getpass.getpass()` functions directly inside an `async def` method. While the application waited for user input, the entire asyncio event loop froze — all other concurrent agent tasks stopped until the console read completed.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `async def process_event()` contains synchronous blocking calls (`input()`, `getpass.getpass()`).
**Why #2:** Python asyncio does not automatically offload blocking calls in `async def` — they run on the event loop thread directly.
**Why #3:** The method was likely ported from a synchronous version without auditing I/O boundaries.
**Why #4:** No integration test existed that ran a concurrent async task alongside the event processor to detect the freeze.
**Why #5:** The assumption was that `async def` alone provides concurrency, when in reality it only provides cooperation — blocking calls opt out of that cooperation.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Replace `input()` / `getpass()` with `await asyncio.get_event_loop().run_in_executor(None, input, prompt)` | proposed | community | https://github.com/ag2ai/ag2/issues/2110 |
| 2 | Add a concurrent task test that verifies other coroutines continue executing during input waits | proposed | community | issue |

## Key Takeaway

Any blocking call inside `async def` — including `input()`, `getpass()`, file reads, or `subprocess.run()` without `asyncio.create_subprocess_exec()` — must be wrapped in `run_in_executor`; the `async def` keyword does not protect the event loop from blocking.
