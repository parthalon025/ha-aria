# Lesson: Migration Command Not Idempotent — Inserts Duplicates on Re-Run

**Date:** 2026-02-28
**System:** lessons-db
**Tier:** lesson
**Category:** data-integrity
**Keywords:** migration, idempotency, duplicate insert, re-run, unique constraint, upsert, database, migrate command, lessons

---

## Observation (What Happened)

`lessons-db migrate` scanned a directory of markdown lesson files and inserted each file into the database. Running it twice against the same directory inserted every lesson twice, producing duplicate records with no error or warning. The fix was adding a unique constraint check (or `INSERT OR IGNORE`) before each insert.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The migrate command iterated files and issued a plain `INSERT` for each one without checking whether the record already existed.
**Why #2:** The author assumed `migrate` would only ever be run once on a fresh database, so idempotency was not a design requirement.
**Why #3:** Operational practice diverged from that assumption: operators run `migrate` after adding new lessons to an already-populated database, expecting it to be safe to re-run.
**Why #4:** No unique constraint existed on the underlying table to reject the second insert at the DB layer, so the duplicate was silently accepted.
**Why #5:** There was no test for the "run migrate twice, count records" scenario.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `INSERT OR IGNORE` (SQLite) or add a `WHERE NOT EXISTS` guard before each insert | proposed | Justin | lessons-db #4 |
| 2 | Add a unique constraint on the natural key (e.g., source file path or title hash) | proposed | Justin | — |
| 3 | Add a test: run `migrate` twice on the same directory, assert record count equals file count, not 2× | proposed | Justin | — |

## Key Takeaway

Any command that populates a database from external files must be idempotent by default — assume it will be re-run, and enforce uniqueness at the DB layer, not just at the call site.
