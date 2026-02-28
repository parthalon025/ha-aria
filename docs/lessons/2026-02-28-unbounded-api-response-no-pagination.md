# Lesson: API Endpoints That Return All Records Become Unusable Over Time

**Date:** 2026-02-28
**System:** ARIA (ha-aria)
**Tier:** lesson
**Category:** error-handling
**Keywords:** pagination, limit, offset, unbounded, events, response size, memory, browser OOM
**Files:** aria/hub/api.py (GET /api/events)

---

## Observation (What Happened)

`GET /api/events` returns all stored events with no `limit` or `offset` parameter. On systems with months of accumulated event history, this returns tens of thousands of records in a single response — causing slow renders, memory pressure on the hub, and potential browser OOM on the dashboard.

The same issue affects audit/timeline queries (noted in issue #8 and addressed for unbounded DB queries) but was not applied to the REST API surface.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The endpoint was built without pagination — it worked fine during development with small datasets.

**Why #2:** Event accumulation is unbounded: a production deployment generates hundreds of events per day; the dataset grows silently until performance degrades.

**Why #3:** Pagination is a forward-design concern — the cost of adding it later is a breaking API change that requires frontend updates.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Add `?limit=500&offset=0` parameters to `/api/events` with a hard cap of 10,000 | proposed | Justin | issue #295 |
| 2 | Audit all list endpoints for unbounded returns; any that touch a DB table need a default limit | proposed | Justin | issues #295, #8 |
| 3 | Apply pagination at API design time — add `limit`/`offset` or cursor to every new list endpoint | proposed | Justin | — |

## Key Takeaway

Every list endpoint that reads from a growing store must have a default `limit` from day one — retrofitting pagination requires a breaking API change; adding it upfront costs nothing.
