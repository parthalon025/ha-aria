# Lesson: Raw `db.execute()` Does Not Auto-Commit in sqlite3/sqlite-utils

**Date:** 2026-02-28
**System:** community (simonw/sqlite-utils)
**Tier:** lesson
**Category:** data-model
**Keywords:** sqlite, execute, commit, autocommit, isolation_level, transaction, silent data loss
**Source:** https://github.com/simonw/sqlite-utils/issues/641

---

## Observation (What Happened)
A developer called `db.execute(INSERT ...)` as the last operation before process exit. The rows were never persisted. The table appeared empty when queried afterward. Higher-level helpers like `.upsert()` did auto-commit, masking the issue in mixed-usage code.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Python's sqlite3 module does not auto-commit DML executed through `connection.execute()` — the connection is in implicit transaction mode by default.

**Why #2:** When the process exits, the open transaction is rolled back (connection close without explicit commit).

**Why #3:** Higher-level functions in sqlite-utils (`.upsert()`, `.insert()`) call `conn.commit()` internally, so callers of those functions never observe this behavior.

**Why #4:** The developer mixed high-level helper calls (which commit) and raw `execute()` calls (which do not), making the bug intermittent and hard to diagnose.

**Why #5:** No error or warning is raised — data is silently lost.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always call `db.conn.commit()` explicitly after raw `db.execute()` DML | proposed | community | issue #641 |
| 2 | Use `with db.conn:` context manager (auto-commits on success, rolls back on exception) | proposed | community | issue #641 comment |
| 3 | Set `isolation_level=None` on the connection for true autocommit mode when explicit transaction control is not needed | proposed | community | issue #641 comment |

## Key Takeaway
Raw `sqlite3.execute()` does not commit — any DML without a following `conn.commit()` or a context-manager block is silently discarded when the process exits.
