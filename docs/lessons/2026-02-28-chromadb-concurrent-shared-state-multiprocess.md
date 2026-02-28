# Lesson: ChromaDB PersistentClient Not Safe for Concurrent Access from Multiple Processes

**Date:** 2026-02-28
**System:** community (chroma-core/chroma)
**Tier:** lesson
**Category:** integration
**Keywords:** chromadb, PersistentClient, multiprocess, Celery, prefork, concurrent, data inconsistency, metadata None, shared state
**Source:** https://github.com/chroma-core/chroma/issues/4897

---

## Observation (What Happened)
A Celery worker running in prefork mode (multiple worker processes) shared a `PersistentClient` instance via a process-level dictionary for reuse. Under load, `collection.query()` returned `metadatas=None` on results that should have had metadata. The developer identified that accessing the same collection from multiple processes simultaneously caused data inconsistency.

## Analysis (Root Cause — 5 Whys)
**Why #1:** `chromadb.PersistentClient` holds a file-system-based SQLite database that is not designed for concurrent multi-process write access.

**Why #2:** In Celery prefork mode, the parent process creates the client object; forked child processes inherit the connection state but operate on separate OS process contexts with independent file descriptors and SQLite locks.

**Why #3:** Multiple child processes reading and writing the same SQLite file simultaneously can produce torn reads where metadata is partially written during query execution.

**Why #4:** The client was stored in a process-level dictionary initialized before forking — a pattern that works for stateless clients but is unsafe for stateful database connections.

**Why #5:** No exception is raised — the metadata simply returns `None` for affected queries, mimicking a "document without metadata" rather than a concurrency error.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Create a fresh `PersistentClient` instance inside each Celery task (after fork) rather than sharing a pre-fork instance | proposed | community | issue #4897 |
| 2 | For multi-process workloads, use `HttpClient` pointing to a dedicated ChromaDB server process instead of `PersistentClient` | proposed | community | issue #4897 |
| 3 | Never share SQLite-backed client objects across fork boundaries — always initialize after the fork | proposed | community | issue #4897 |

## Key Takeaway
`PersistentClient` (SQLite-backed) is not fork-safe — never share it across Celery prefork workers or any multi-process boundary; initialize a new client instance per process after forking.
