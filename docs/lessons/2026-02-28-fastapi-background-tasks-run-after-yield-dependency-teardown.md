# Lesson: FastAPI yield Dependencies With Background Tasks — Teardown Order Reversed in 0.116+
**Date:** 2026-02-28
**System:** community (tiangolo/fastapi)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** FastAPI, yield dependency, background task, teardown order, transaction, database, response lifecycle
**Source:** https://github.com/tiangolo/fastapi/issues/14988
---
## Observation (What Happened)
In FastAPI 0.116+ (and 0.128+, 0.133+), when a route uses both a `yield` dependency and `BackgroundTasks`, the dependency's post-yield cleanup now runs *before* the background tasks complete — specifically: response is sent, dependency tears down, then background task runs. In FastAPI 0.115.12 and earlier, the order was: dependency tears down, response sends, background task runs. If the `yield` dependency manages a database transaction, the transaction commits before the response is sent; in newer versions it commits before the background task runs but after the response, meaning background tasks that read from the committed transaction data now see it, but the behavior change is a breaking semantic shift.

## Analysis (Root Cause — 5 Whys)
FastAPI's dependency resolution and background task execution order changed across versions as the response lifecycle was refactored. The `yield` dependency cleanup (the code after `yield`) is now deferred to run as part of the response finalization, which is after the response bytes are sent but interleaved differently with background task scheduling. Application code that relies on the relative order of "DB transaction committed → background task reads result" is brittle to this version coupling.

## Corrective Actions
- Never rely on execution order between `yield` dependency teardown and `BackgroundTasks` — the order is not guaranteed stable across FastAPI versions.
- If a background task must observe database changes made in the same request, use explicit commit inside the handler before adding the background task, rather than relying on the dependency's post-yield commit.
- Add a changelog check step when upgrading FastAPI to verify dependency/background task ordering if the service uses this pattern.
- In ARIA's hub: any FastAPI route that uses a `yield` dependency (DB session, resource lock) and also registers background tasks should be audited — move explicit resource commits/releases before the background task addition.

## Key Takeaway
FastAPI's execution order of `yield` dependency teardown relative to `BackgroundTasks` is not stable across versions — never rely on teardown ordering for correctness; commit resources explicitly before adding background tasks.
