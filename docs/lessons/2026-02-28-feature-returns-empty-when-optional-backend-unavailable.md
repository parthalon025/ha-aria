# Lesson: Feature Returns Empty Results When Optional Backend Unavailable — No Fallback Path

**Date:** 2026-02-28
**System:** lessons-db
**Tier:** lesson
**Category:** reliability
**Keywords:** optional dependency, fallback, graceful degradation, LanceDB, SQLite, keyword search, semantic search, empty results, feature flag, silent empty

---

## Observation (What Happened)

`lessons-db search` required LanceDB for semantic vector search. When LanceDB was not installed, the command returned zero results with no error or explanation, rather than falling back to a SQLite `LIKE` keyword search. Users saw an empty result set and had no signal that the search infrastructure was missing.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The search function was written to call LanceDB and return its results; when LanceDB was absent, the import failed silently and the function returned an empty list.
**Why #2:** The author designed for the "LanceDB present" happy path only, treating it as a required dependency rather than an optional enhancement.
**Why #3:** SQLite `LIKE` search already existed in the codebase for other queries but was not wired into the search fallback path.
**Why #4:** No startup check or capability flag exposed whether semantic search was available, so callers had no way to know what mode they were in.
**Why #5:** No test exercised `search()` with LanceDB absent.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Wrap LanceDB import in `try/except ImportError`; fall back to SQLite `LIKE` on failure | proposed | Justin | lessons-db #1 |
| 2 | Log a INFO/WARNING at startup when operating in keyword-only mode | proposed | Justin | — |
| 3 | Add a test: uninstall/mock LanceDB, run `search()`, assert keyword results returned (not empty) | proposed | Justin | — |

## Key Takeaway

A feature that silently returns empty when its optional backend is unavailable is indistinguishable from "no matching results" — always define and wire the degraded path before treating the optimal path as the only path.
