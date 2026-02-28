# Lesson: Chat History Sliding Window Evicts System Prompt — Bot Loses Role Identity

**Date:** 2026-02-28
**System:** community (TheR1D/shell_gpt)
**Tier:** lesson
**Category:** integration
**Keywords:** chat-history, sliding-window, system-prompt, role-loss, truncation, LLM, CHAT_CACHE_LENGTH, context
**Source:** https://github.com/TheR1D/shell_gpt/issues/633

---

## Observation (What Happened)

`shell_gpt` uses a `CHAT_CACHE_LENGTH` config to trim old chat messages when the history grows too long. After enough turns, the initial system prompt (which defines the bot's role) was evicted by the sliding window. The tool then threw "Could not determine chat role" — a silent functional break that only manifested after many messages.

## Analysis (Root Cause — 5 Whys)

**Why #1:** After many messages, the bot could not determine its operating role.
**Why #2:** The `CHAT_CACHE_LENGTH` trim removed the oldest messages — including the first system/role message.
**Why #3:** The trimming logic used a simple `messages[-N:]` slice without distinguishing role/system messages from conversational messages.
**Why #4:** The developer assumed role messages would always be present since they were added first.
**Why #5:** No invariant enforcement prevented the system prompt from being in the eviction zone.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Pin the system/role message (index 0) and apply the sliding window only to messages[1:] | proposed | community | issue #633 |
| 2 | When trimming, always preserve messages where `role == "system"` regardless of position | proposed | community | issue #633 |
| 3 | Add a guard on load: if no system message found, re-inject the default role before sending | proposed | community | defensive pattern |
| 4 | Test chat history trim behavior at `CHAT_CACHE_LENGTH` boundary in CI | proposed | community | regression prevention |

## Key Takeaway

Sliding-window chat history truncation must treat system/role messages as pinned and never evict them — trimming `messages[-N:]` without filtering on `role` is guaranteed to silently corrupt the bot's identity after enough turns.
