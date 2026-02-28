# Lesson: Telegram FloodWait State Persists Across Process Restarts

**Date:** 2026-02-28
**System:** community (Rongronggg9/RSS-to-Telegram-Bot)
**Tier:** lesson
**Category:** integration
**Keywords:** telegram, flood-wait, rate-limit, restart, telethon, python-telegram-bot, persistent-ban, MTProto
**Source:** https://github.com/Rongronggg9/RSS-to-Telegram-Bot/issues/575
**Source:** https://github.com/Rongronggg9/RSS-to-Telegram-Bot/issues/542

---

## Observation (What Happened)

A Telegram bot was rate-limited by the API with a `FloodWaitError` requiring a 43,837-second wait. After restarting the Docker container, the flood ban immediately reappeared and blocked all outgoing messages — the restart did not reset the ban state.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Restarting the process did not fix the flood wait.
**Why #2:** The `FloodWaitError` ban is server-side on Telegram's infrastructure, not stored locally in the bot process.
**Why #3:** The bot sent messages to many users in a short burst (e.g., batch RSS delivery) without per-chat rate limiting.
**Why #4:** The code used the same Telegram client/token for high-frequency bulk sends with no delay between messages.
**Why #5:** Telegram's MTProto flood control operates per-token per-chat globally — it cannot be cleared by restarting the client.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Respect `FloodWaitError.seconds` — sleep exactly that long before resuming | proposed | community | issue #575 |
| 2 | Implement per-chat send queues with minimum inter-message delay (1–3s recommended by Telegram) | proposed | community | issue #542 |
| 3 | On startup, check for active flood bans before sending any queued messages | proposed | community | issue #542 |
| 4 | Never use batch-send loops without `asyncio.sleep()` between sends | proposed | community | Telegram Bot API docs |

## Key Takeaway

Telegram flood bans are server-side and survive process restarts — the only fix is to honor the `retry_after` value in `FloodWaitError` and rate-limit outgoing sends at the application layer before the API enforces it.
