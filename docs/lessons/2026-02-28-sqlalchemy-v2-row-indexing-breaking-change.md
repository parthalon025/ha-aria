# Lesson: SQLAlchemy v2 Returns `Row` Objects — String Index Access Breaks Silently at Runtime

**Date:** 2026-02-28
**System:** community (slackapi/bolt-python)
**Tier:** lesson
**Category:** integration
**Keywords:** SQLAlchemy, v2, Row, tuple, string index, breaking change, ORM, query result, integer index, migration
**Source:** https://github.com/slackapi/bolt-python/issues/822

---

## Observation (What Happened)

Code using SQLAlchemy to query an OAuth state store worked correctly with SQLAlchemy v1.x but silently failed after upgrading to v2.x. The v2 `Row` object does not support string-key indexing (`row['column_name']`); only integer index or attribute access (`row.column_name`) is supported. This caused a `TypeError: tuple indices must be integers or slices, not str` at runtime, which was surfaced to the user as `invalid_state` — the wrong error, masking the root cause.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Users got `invalid_state` errors on OAuth even when state was valid.
**Why #2:** The state lookup function raised `TypeError: tuple indices must be integers or slices, not str` — silently caught as a generic failure, returning "not found".
**Why #3:** SQLAlchemy v2 changed `Row.__getitem__` to raise on string keys; v1 allowed `row['col']`.
**Why #4:** No CI test matrix covered SQLAlchemy v2; the upgrade passed silently.
**Why #5:** String key access to `Row` objects is idiomatic in v1-era SQLAlchemy code and becomes a runtime bomb on v2 upgrade.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Migrate all `row['column']` access to `row.column` or `row[0]` for SQLAlchemy v2 compatibility | proposed | community | issue #822 |
| 2 | Include SQLAlchemy v2 in CI test matrix for any code that accesses query result rows | proposed | community | issue #822 |
| 3 | When upgrading SQLAlchemy, grep for `row['` and `result['` patterns — these are all v1 breakage points | proposed | community | issue #822 |

## Key Takeaway

SQLAlchemy v2 removed string-key indexing on `Row` objects — any `row['column']` pattern breaks silently (caught as a generic query failure) on upgrade; audit and migrate all row access patterns before bumping the dependency version.
