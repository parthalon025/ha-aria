# Lesson: SQLite Schema Migration via Table Recreate Silently Drops Existing Indexes

**Date:** 2026-02-28
**System:** community (simonw/sqlite-utils)
**Tier:** lesson
**Category:** data-model
**Keywords:** sqlite, migration, index, foreign key, table recreate, silent drop, add_foreign_key, schema change
**Source:** https://github.com/simonw/sqlite-utils/issues/633

---

## Observation (What Happened)
A developer created a table with a unique index on `name`, then called `db.add_foreign_keys()`. After the call, the unique index silently disappeared. The table schema (columns, constraints) was correct but all user-defined indexes were gone.

## Analysis (Root Cause — 5 Whys)
**Why #1:** SQLite does not support `ALTER TABLE ADD CONSTRAINT` — adding foreign keys requires dropping and recreating the table.

**Why #2:** The recreate path copies columns, primary key, and NOT NULL constraints but does not automatically re-execute the original `CREATE INDEX` statements.

**Why #3:** The `CREATE INDEX` statements are stored in `sqlite_master` and are discoverable, but the migration code did not query them before dropping the old table.

**Why #4:** No error or warning is raised — the indexes simply vanish.

**Why #5:** Developers assume schema migrations are additive-only (ADD COLUMN pattern); they do not expect a constraint addition to remove unrelated artifacts.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Before any operation that triggers table recreate, snapshot existing indexes from `sqlite_master WHERE type='index'` and re-apply them after recreate | proposed | community | issue #633 PR #634 |
| 2 | Warn loudly (or raise) when a migration silently drops non-PK indexes rather than re-creating them | proposed | community | issue #633 comment |
| 3 | Write an integration test that verifies index count before and after every schema migration call | proposed | community | issue #633 |

## Key Takeaway
Any SQLite operation that internally drops and recreates a table (add foreign key, transform, rename column) will silently discard all non-PK indexes — snapshot and restore them explicitly.
