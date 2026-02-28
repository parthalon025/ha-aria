# Lesson: MLflow Recreates SQLAlchemy Engine per Request — Leaks Connection Pool Under PostgreSQL

**Date:** 2026-02-28
**System:** community (mlflow/mlflow)
**Tier:** lesson
**Category:** reliability
**Keywords:** mlflow, sqlalchemy, connection-pool, postgresql, engine-recreation, leak, pg-connections, cloud-sql, pooling, tracking-server
**Source:** https://github.com/mlflow/mlflow/issues/19379

---

## Observation (What Happened)

An MLflow tracking server configured with a PostgreSQL backend (Cloud SQL) accumulated idle PostgreSQL connections over time during normal UI usage, eventually exhausting the database connection limit. The server log showed repeated `INFO: Create SQLAlchemy engine with pool options {...}` entries — a new connection pool was created per UI request rather than being reused. The previously created pools were not disposed, causing pool accumulation and connection exhaustion. SQLite backends were unaffected because SQLite uses file-level locking rather than a connection pool.

## Analysis (Root Cause — 5 Whys)

The MLflow tracking store instantiated a new `create_engine()` call in a code path that was not guarded by a singleton or module-level cache. Because `create_engine` creates a new connection pool each time, and the old engine object was not explicitly `.dispose()`-d, the connection pool remained alive in memory holding open PostgreSQL connections until Python garbage collected the engine — which under long-lived server processes is effectively never. The bug was triggered by a missing module-level engine cache.

## Corrective Actions

- When building any service that wraps a database ORM: instantiate the engine exactly once at module or application startup, store it as a module-level singleton, and access it everywhere via that singleton — never create engines inside request handlers.
- Add a health check that monitors active DB connection count against `max_connections`; alert when > 80% utilization.
- In ARIA's hub.db SQLite access: use `engine = create_engine(url, pool_pre_ping=True)` once in `__init__` and reuse across all queries — already following the pattern, but verify no code path calls `create_engine` in a request handler.

## Key Takeaway

`create_engine()` must be called exactly once at startup and reused as a singleton — calling it per request creates a new connection pool each time, leaking connections until the process runs out.
