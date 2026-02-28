# Lesson: Unbounded MongoDB Document Growth Causes Silent CPU Spike and Server Lock-Up

**Date:** 2026-02-28
**System:** community (father-bot/chatgpt_telegram_bot)
**Tier:** lesson
**Category:** performance
**Keywords:** mongodb, unbounded-growth, cpu-spike, memory, chat-history, document-size, index, query, performance
**Source:** https://github.com/father-bot/chatgpt_telegram_bot/issues/488

---

## Observation (What Happened)

A Telegram bot storing conversation history in MongoDB caused server load average to spike to 25+ on a single-CPU machine, making it unresponsive. The root cause was unbounded chat history documents growing with every new message, causing MongoDB to do constant read/write at 45-50 MB/s even for a single user in low-traffic conditions.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The server became unresponsive with MongoDB consuming nearly all CPU.
**Why #2:** MongoDB was doing constant high-rate I/O with no apparent trigger.
**Why #3:** Each new message was appended to an array field in the conversation document without any size cap, causing the document to grow unboundedly.
**Why #4:** MongoDB must rewrite entire documents on update — a large array update on a growing document triggers disk I/O proportional to document size on every message.
**Why #5:** No retention policy, TTL index, or document size cap was applied when the DB storage pattern was implemented.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Cap message history arrays to a fixed length (e.g., last N messages) before each write | proposed | community | issue #488 |
| 2 | Use a TTL index on the conversation collection to expire old documents automatically | proposed | community | MongoDB docs |
| 3 | Store messages as individual documents (not appended arrays) with an index on conversation_id | proposed | community | schema redesign |
| 4 | Add a monitoring alert on MongoDB collection document size — alert if median doc > 100KB | proposed | community | operational |
| 5 | Load test with 1000 messages per conversation before shipping a persistence-backed bot | proposed | community | regression |

## Key Takeaway

Appending to an array field in a single MongoDB document without a size cap will cause the document to grow unboundedly and trigger O(N) disk rewrites on every insert — always cap array fields or use separate document-per-message schema with a TTL index.
