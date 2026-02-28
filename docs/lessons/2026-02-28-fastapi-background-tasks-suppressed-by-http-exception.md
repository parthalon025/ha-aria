# Lesson: FastAPI BackgroundTasks Added Before HTTPException Are Silently Dropped
**Date:** 2026-02-28
**System:** community (tiangolo/fastapi)
**Tier:** lesson
**Category:** reliability
**Keywords:** FastAPI, BackgroundTasks, HTTPException, background task, silent drop, error handling, task lifecycle
**Source:** https://github.com/tiangolo/fastapi/issues/13116
---
## Observation (What Happened)
When a FastAPI route calls `background_tasks.add_task(...)` and then raises `HTTPException`, the registered background task is silently dropped — it never runs. The task is discarded because `HTTPException` short-circuits the normal response path before FastAPI's background task runner executes.

## Analysis (Root Cause — 5 Whys)
FastAPI attaches background tasks to the `Response` object that is returned through the normal handler path. When `HTTPException` is raised, it is caught by the exception handler middleware, which constructs a new `JSONResponse` independently — it has no knowledge of the `BackgroundTasks` object that was being built in the route function. The `BackgroundTasks` instance is simply garbage collected along with the route's local scope, and the tasks never fire.

## Corrective Actions
- Never register background tasks before a potential `HTTPException` raise if the task must run regardless of outcome. Use a `try/finally` block to schedule guaranteed work, or use a lifespan-level task queue that accepts work units independently of the request/response cycle.
- If background tasks must fire even on error paths (e.g., audit logging, cleanup), use an explicit task queue (like ARIA's `hub.publish()` event pattern) rather than `BackgroundTasks`.
- Add a lint rule or code review check: any route that calls `background_tasks.add_task()` followed by a conditional `raise HTTPException` should be flagged for review.

## Key Takeaway
`BackgroundTasks` registered in a FastAPI route are silently dropped if `HTTPException` is raised — tasks that must run on error paths belong in a separate task queue, not in `BackgroundTasks`.
