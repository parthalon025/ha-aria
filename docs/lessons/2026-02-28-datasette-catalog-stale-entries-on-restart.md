# Lesson: Persistent Internal DB Retains Stale Catalog Entries Across Restarts — Causes KeyError on Lookup

**Date:** 2026-02-28
**System:** community (simonw/datasette)
**Tier:** lesson
**Category:** data-model
**Keywords:** sqlite, internal db, catalog, stale entries, restart, KeyError, 500 error, persistent state, housekeeping
**Source:** https://github.com/simonw/datasette/issues/2605

---

## Observation (What Happened)
When `internal.db` was set to a persistent file (not in-memory), restarting Datasette with different databases attached caused a `500 KeyError` on the index page. The internal `catalog_databases` table still contained entries for databases that were no longer attached in the current session.

## Analysis (Root Cause — 5 Whys)
**Why #1:** On startup, the catalog population code inserted entries for currently-attached databases but did not first clear entries from the previous session.

**Why #2:** The index page code looked up each `catalog_databases` entry in the in-memory registry of live databases — a key present in the table but absent from the live registry caused a `KeyError`.

**Why #3:** The internal DB was designed for in-memory use where it is always empty on startup; persistent mode was added later without updating the startup housekeeping path.

**Why #4:** The bug only manifested when: (a) `internal.db` was persistent AND (b) the set of attached databases changed between restarts — a combination not covered by the test suite.

**Why #5:** No error was raised during startup — the inconsistency only surfaced at request time, making it hard to associate the error with the startup sequence.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | On application startup, truncate or DELETE all catalog tables that derive from the current runtime state before repopulating them | proposed | maintainer | issue #2605 — fix: wipe catalog on startup |
| 2 | Treat any persistent internal/catalog table as requiring a startup reconciliation pass: DELETE WHERE (key not in current live set) | proposed | community | issue #2605 |
| 3 | Add an integration test that starts with a persistent internal DB, removes an attached database, restarts, and verifies no 500 on the index page | proposed | community | issue #2605 |

## Key Takeaway
Persistent internal databases retain stale catalog entries across restarts — always truncate or reconcile derived catalog tables at startup before repopulating from current runtime state.
