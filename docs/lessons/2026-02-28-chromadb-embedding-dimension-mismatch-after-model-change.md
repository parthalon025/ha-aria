# Lesson: ChromaDB Embedding Dimension Mismatch After Default Embedding Function Change

**Date:** 2026-02-28
**System:** community (chroma-core/chroma)
**Tier:** lesson
**Category:** data-model
**Keywords:** chromadb, embedding, dimension, mismatch, default embedding function, InvalidDimensionException, collection, model change
**Source:** https://github.com/chroma-core/chroma/issues/4368

---

## Observation (What Happened)
A developer created a ChromaDB collection with Google Gemini embeddings (768 dimensions), persisted the database, then later called `client.get_collection(name)` without specifying the embedding function. ChromaDB defaulted to its built-in model (384 dimensions), producing `InvalidDimensionException: Embedding dimension 384 does not match collection dimensionality 768`.

## Analysis (Root Cause — 5 Whys)
**Why #1:** ChromaDB stores collection dimensionality at creation time; all subsequent operations must use an embedding function that produces the same dimension.

**Why #2:** `get_collection()` without `embedding_function=` uses the default embedding function (currently `all-MiniLM-L6-v2`, 384 dims).

**Why #3:** The developer assumed the embedding function used at `create_collection` time was persisted and would be re-applied automatically on `get_collection`.

**Why #4:** ChromaDB only persists the embedding function configuration if `api_key_env_var` is used (not a raw `api_key=` value) — an undocumented constraint.

**Why #5:** The error fires at query time, not at `get_collection` time, making the root cause harder to trace.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always pass `embedding_function=` explicitly to both `create_collection()` and `get_collection()` — never rely on default | proposed | community | issue #4368 |
| 2 | Store the embedding model name alongside the collection in application metadata (e.g., a separate config record); verify on reconnect | proposed | community | issue #4368 |
| 3 | Use `api_key_env_var=` (not `api_key=`) when initializing embedding functions that need to be persisted across sessions | proposed | community | issue #4368 comment |

## Key Takeaway
ChromaDB does not automatically restore the original embedding function on `get_collection()` — always pass `embedding_function=` explicitly, and store the model name in your own config to catch dimension drift before it hits production.
