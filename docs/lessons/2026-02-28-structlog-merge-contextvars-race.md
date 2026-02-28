# Lesson: structlog merge_contextvars Race Condition When Iterating contextvars.copy_context()

**Date:** 2026-02-28
**System:** community (hynek/structlog)
**Tier:** lesson
**Category:** async
**Keywords:** structlog, contextvars, merge_contextvars, race condition, KeyError, copy_context, threading, Gunicorn
**Source:** https://github.com/hynek/structlog/issues/591

---

## Observation (What Happened)

In a Flask + Gunicorn app, upgrading structlog from 20.2.0 to 21.1.0 introduced intermittent `KeyError` on `merge_contextvars`. The error appeared when a log call was made while another thread was concurrently modifying the context. The stack trace pointed to `ctx[k] is not Ellipsis` — a key present in the `copy_context()` snapshot no longer existed by the time the loop body ran.

## Analysis (Root Cause — 5 Whys)

The 21.1.0 implementation changed from copying a private dict (`_get_context().copy()`) to calling `contextvars.copy_context()` and then iterating it. Under the old approach, the dict was fully materialized at copy time. Under the new approach, `copy_context()` returns a `Context` object that may not be stable across threads if the context itself is being mutated concurrently — a key visible in the snapshot can be absent when dereferenced during iteration in a different thread. The fix required re-copying the context or iterating a fully materialized dict snapshot.

## Corrective Actions

- When iterating `contextvars.copy_context()` output in a multi-threaded environment, materialize to a dict first: `{k: v for k, v in ctx.items()}` before the loop.
- Treat any `KeyError` from `copy_context()` iteration as a threading race, not a logic error.
- In Gunicorn/threaded environments, prefer `structlog.contextvars.clear_contextvars()` at request boundaries over managing raw `ContextVar`s to avoid stale state from worker thread reuse.

## Key Takeaway

`contextvars.copy_context()` returns a live snapshot that can raise `KeyError` on access in multi-threaded code — always materialize to a plain dict before iterating.
