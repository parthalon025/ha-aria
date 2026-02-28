# Lesson: SQLite Upsert with NOT NULL Constraint Silently Writes No Rows When Using INSERT OR IGNORE Pattern

**Date:** 2026-02-28
**System:** community (simonw/sqlite-utils)
**Tier:** lesson
**Category:** data-model
**Keywords:** sqlite, upsert, not_null, INSERT OR IGNORE, constraint, silent failure, no rows written
**Source:** https://github.com/simonw/sqlite-utils/issues/538

---

## Observation (What Happened)
A developer called `table.upsert_all([...], pk="id", not_null=["name"])` and zero rows appeared in the table. No exception was raised. The table schema was created correctly (NOT NULL on `name`), but the upsert silently wrote nothing.

## Analysis (Root Cause — 5 Whys)
**Why #1:** sqlite-utils implements upsert as `INSERT OR IGNORE INTO table(id) VALUES(?)` followed by `UPDATE table SET col=? WHERE id=?`.

**Why #2:** `INSERT OR IGNORE INTO table(id)` inserts a row with `id` only, leaving `name` as NULL — but the table has `name NOT NULL`.

**Why #3:** The NOT NULL constraint causes the INSERT to fail. `INSERT OR IGNORE` suppresses the `IntegrityError` silently — the row is never inserted.

**Why #4:** The subsequent `UPDATE ... WHERE id=?` finds no row (since the insert was ignored) and updates zero rows — also silently.

**Why #5:** The caller has no indication anything went wrong: no exception, no rowcount check, no warning.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use `INSERT OR REPLACE` or true `INSERT ... ON CONFLICT DO UPDATE` (SQLite 3.24+) instead of the INSERT OR IGNORE + UPDATE split when NOT NULL columns are involved | proposed | community | issue #538 |
| 2 | After any upsert operation, assert `cursor.rowcount > 0` or verify row existence when NOT NULL constraints are present | proposed | community | issue #538 |
| 3 | Treat `INSERT OR IGNORE` as a code smell for tables with NOT NULL non-PK columns — the pattern can silently swallow valid data | proposed | community | issue #538 |

## Key Takeaway
`INSERT OR IGNORE` + `UPDATE` upsert pattern silently discards all data when a NOT NULL column prevents the initial INSERT — use `INSERT ... ON CONFLICT DO UPDATE` (UPSERT syntax) for tables with NOT NULL constraints.
