# Lesson: Two-Step DB Mutations Without Idempotency Guard Create Concurrent Corruption

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** data-model
**Keywords:** database, idempotency, race condition, concurrent, mark_reviewed, duplicate insert, TOCTOU, face, label
**Files:** aria/faces/store.py:184, aria/hub/routes_faces.py:99

---

## Observation (What Happened)

`label_face` uses a two-step approach: (1) fetch a queue item, (2) `mark_reviewed` then `add_embedding`. Between steps 1 and 2, a second concurrent request can fetch the same pending item. Both calls proceed to `mark_reviewed` and `add_embedding`, producing conflicting duplicate embeddings for the same event (e.g., one labeled "alice", one labeled "bob"). The `mark_reviewed` UPDATE has no `AND reviewed_at IS NULL` guard, so it succeeds even when the item was already reviewed by the concurrent request.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `mark_reviewed`'s SQL UPDATE does not include `AND reviewed_at IS NULL`, so it overwrites an already-reviewed item.

**Why #2:** The fetch-then-mutate pattern has an inherent race window — without a database-level atomicity mechanism, two concurrent callers can both complete step 1 before either completes step 2.

**Why #3:** Single-user development testing doesn't expose this race — it only manifests when two review sessions (browser tabs, concurrent API calls) are active simultaneously.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `AND reviewed_at IS NULL` to `mark_reviewed`'s UPDATE; return a bool indicating whether the update affected a row | proposed | Justin | issue #192 |
| 2 | In `label_face`, skip `add_embedding` if `mark_reviewed` returned False (row already reviewed) | proposed | Justin | issue #192 |
| 3 | Apply the same "check rowcount after UPDATE" pattern to any other two-step DB mutations that must be idempotent | proposed | Justin | — |

## Key Takeaway

Any two-step "fetch then mutate" DB operation must use a conditional UPDATE (`AND status = 'pending'`) and check rowcount — `rowcount == 0` means the race was lost; skip the second step to avoid conflicting duplicates.
