# Lesson: Notion API Silently Truncates Results at 100 Items — Pagination Cursor Must Be Followed

**Date:** 2026-02-28
**System:** community (ramnes/notion-sdk-py)
**Tier:** lesson
**Category:** integration
**Keywords:** notion, pagination, cursor, 100 items, truncation, iterate_paginated_api, next_cursor, start_cursor, silent data loss
**Source:** https://github.com/ramnes/notion-sdk-py/issues/224

---

## Observation (What Happened)
A developer using `iterate_paginated_api(notion.databases.query, ...)` after a library upgrade (2.2.0) saw the generator stuck in an infinite loop returning the same first 100 items repeatedly. The `start_cursor` parameter was not being forwarded to subsequent requests, so the pagination cursor was lost and the same first page was fetched indefinitely.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Notion's API returns a maximum of 100 items per request regardless of any `page_size` argument — callers must follow `next_cursor` to retrieve subsequent pages.

**Why #2:** A library refactor removed the `pick()` function call that forwarded `start_cursor` from the paginated response back into the next request parameters.

**Why #3:** The `iterate_paginated_api` wrapper continued to call the API in a loop, but every iteration started from the beginning because `start_cursor` was missing.

**Why #4:** The loop did not detect infinite repetition — it re-fetched page 1 indefinitely, producing an infinite generator of the same data.

**Why #5:** The regression was caught only by users with databases >100 items; users with smaller databases saw no behavior change.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always use `iterate_paginated_api()` (or equivalent cursor-following wrapper) for any Notion API call that may return >100 items — never call the API endpoint directly for unbounded lists | proposed | community | issue #224 |
| 2 | Assert that pagination terminates within expected bounds (e.g., `len(list(pages)) <= expected_count`) in integration tests | proposed | community | issue #224 |
| 3 | After library upgrades that touch the HTTP layer, add a regression test that verifies cursor forwarding with a database containing >100 items | proposed | community | issue #224 |

## Key Takeaway
The Notion API silently returns only 100 items per page — always use a cursor-following wrapper, and add a regression test verifying that pagination terminates correctly after library upgrades touch the HTTP client layer.
