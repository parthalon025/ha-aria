# Lesson: ORM Model Metadata Is Shared Global State — Race Condition Under DB Switching in Multithreaded Apps

**Date:** 2026-02-28
**System:** community (coleifer/peewee)
**Tier:** lesson
**Category:** integration
**Keywords:** peewee, ORM, thread-safe, model metadata, database binding, race condition, multithreaded, bind_ctx, table name null
**Source:** https://github.com/coleifer/peewee/issues/3013

---

## Observation (What Happened)
A Flask/gunicorn app with multiple threads used `bind_ctx(ALL_MODELS)` to switch models between a master and read-only replica database within a request context. Intermittently, SQL queries contained `NULL."field_name"` — the table was replaced by `None` — causing a Postgres syntax error. The bug was non-deterministic and appeared only under concurrency.

## Analysis (Root Cause — 5 Whys)
**Why #1:** Model metadata in peewee stores the `_table` (table reference object) on the model's `Meta` class — shared state across all threads.

**Why #2:** `Metadata.set_database()` deletes and resets `_table`; `Metadata.table()` reads `_table`. Without a lock, thread A can read `_table = None` between thread B's delete and re-creation.

**Why #3:** `bind_ctx()` calls `set_database()` under the hood, creating a write/read race on every context switch.

**Why #4:** The model metadata class `ThreadSafeDatabaseMetadata` existed but did not protect the table object itself — only the database reference.

**Why #5:** Developers assume context managers are inherently thread-safe; ORM context managers that modify shared class-level state are not.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Use a threading.Lock around `set_database` + table invalidation, and move `_table` into a `threading.local()` | resolved | maintainer | issue #3013 — fix committed by coleifer |
| 2 | Prefer separate model subclasses for read replicas (explicit `Meta.database`) over dynamic binding in multithreaded apps | proposed | community | issue #3013 comment |
| 3 | Test ORM model operations under concurrent thread access when using `bind_ctx` or any DB-switching context manager | proposed | community | issue #3013 |

## Key Takeaway
ORM model metadata stored at the class level is shared global state — any method that mutates it (database binding, table invalidation) must be protected by a per-model lock and/or stored in thread-local storage.
