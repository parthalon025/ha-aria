# Lesson: ChromaDB Query Returns Incorrect Results After Document Deletion in Small Collections

**Date:** 2026-02-28
**System:** community (chroma-core/chroma)
**Tier:** lesson
**Category:** data-model
**Keywords:** chromadb, delete, query, incorrect results, HNSW, vector index, small collection, data integrity
**Source:** https://github.com/chroma-core/chroma/issues/4275

---

## Observation (What Happened)
After deleting a document from a ChromaDB collection and querying with `n_results=1`, the result pointed to a different document than querying a collection that had never contained the deleted document. The discrepancy appeared only when: the collection had fewer than ~100 items AND more than ~20% of data had been deleted. ChromaDB versions 1.0.0 and later were affected; 0.6.x was not.

## Analysis (Root Cause — 5 Whys)
**Why #1:** The internal vector index (HNSW via Rust backend in 1.0.x) did not correctly handle deletion tombstones when the collection was small — deleted rows were still being considered during ANN search.

**Why #2:** The 1.0.0 release switched from hnswlib 0.7.6 (Python) to hnswlib 0.8.1 (Rust), which has different soft-delete semantics for small graphs.

**Why #3:** The regression was not caught by the test suite because most tests used larger collections or did not verify post-delete result ordering.

**Why #4:** The bug manifested only at a specific intersection: small collection + high deletion ratio — a combination rarely tested by library authors.

**Why #5:** The application assumed that `delete(ids=[...])` guaranteed those vectors were excluded from all future queries immediately.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | After bulk deletes, call `collection.count()` and compare to expected count before relying on query results | proposed | community | issue #4275 |
| 2 | For small, high-churn collections, prefer `collection.delete()` + recreate over in-place deletion when data integrity is critical | proposed | community | issue #4275 |
| 3 | Pin ChromaDB version in production and test upgrade paths specifically with delete-heavy workloads at small scale (<100 items) | proposed | community | issue #4275 |

## Key Takeaway
Vector database deletions are soft-deletes internally — query correctness after deletion can regress between versions, especially in small collections; always test delete + query round-trips as part of version upgrade validation.
