# Lesson: Retry Loop Enters Infinite Spin When Initialize Is Not Idempotent

**Date:** 2026-02-28
**System:** community (python-telegram-bot/python-telegram-bot)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** bootstrap, retry, infinite-loop, idempotent, initialize, already-initialized, python-telegram-bot, polling
**Source:** https://github.com/python-telegram-bot/python-telegram-bot/issues/4966

---

## Observation (What Happened)

A bot using `run_polling(bootstrap_retries=-1)` (retry forever) entered an infinite loop with "This application is already initialized" errors and never started processing commands. The issue was introduced in version 22.4 when the initialization path changed.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The bot looped infinitely with "already initialized" errors and never polled.
**Why #2:** The retry loop called `application.initialize()` on each attempt, but `initialize()` was not idempotent — calling it twice on an already-initialized object raised an error instead of being a no-op.
**Why #3:** A version bump changed the retry loop path such that `initialize()` was called before checking whether initialization had already succeeded.
**Why #4:** The retry loop did not distinguish between "startup failed, retry" and "startup succeeded, do not retry."
**Why #5:** The `bootstrap_retries=-1` sentinel (retry forever) was only safe when initialize was idempotent; the library's internal contract changed without that invariant being enforced by a test.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Make `initialize()` idempotent — check `self._initialized` and return early if already done | proposed | community | issue #4966 |
| 2 | Retry loops must have an explicit success sentinel that exits the loop unconditionally | proposed | community | issue #4966 |
| 3 | Version bump regression test: run the exact `bootstrap_retries=-1` pattern in CI | proposed | community | regression |
| 4 | Log startup state transitions (initializing → initialized → polling) to make infinite loops visible immediately | proposed | community | operational |

## Key Takeaway

Any `initialize()` method called inside a retry loop must be idempotent — calling it on an already-initialized object must be a safe no-op, because a retry loop cannot distinguish "not yet initialized" from "already initialized and working" without that guarantee.
