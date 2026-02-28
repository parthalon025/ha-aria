# Lesson: Autologging Assumes Single fit() Call per Run — Multiple fit() Calls Cause UNIQUE Constraint Crash

**Date:** 2026-02-28
**System:** community (mlflow/mlflow)
**Tier:** lesson
**Category:** integration
**Keywords:** mlflow, autolog, tensorflow, keras, fit-multiple-times, unique-constraint, sqlite, metrics-table, incremental-training, run-id
**Source:** https://github.com/mlflow/mlflow/issues/19144

---

## Observation (What Happened)

`mlflow.tensorflow.autolog()` failed with `sqlite3.IntegrityError: UNIQUE constraint failed: metrics.key, metrics.timestamp, metrics.step, metrics.run_uuid, metrics.value, metrics.is_nan` starting on the second call to `model.fit()` within the same MLflow run. The autologger logged epoch-level metrics using the epoch number as `step`. On the second call to `fit()`, epoch numbering restarted at 0, producing duplicate `(key, step, run_uuid)` tuples that violated the unique constraint. The first fit call succeeded; all subsequent calls failed.

## Analysis (Root Cause — 5 Whys)

The autologger's step counter was not offset by the total steps already logged in the run. MLflow's metric schema uses `(key, timestamp, step, run_uuid, value)` as a compound unique key — if `step` repeats within a run, it only survives if `timestamp` or `value` differ. Keras resets its epoch counter per `fit()` call, so both timestamp collisions (fast machine) and exact value collisions (stable loss) could trigger the constraint. The autologger was designed for the single-fit training paradigm and does not track cumulative epoch offsets across fit calls.

## Corrective Actions

- When training in batches with multiple `fit()` calls, wrap each call in its own `mlflow.start_run(run_name=f"batch_{i}")` rather than reusing a single run, or manually log metrics with an incrementing `step` offset.
- Alternatively, disable `log_every_n_steps` in autolog and log epoch metrics manually with a cumulative step counter maintained by the caller.
- In ARIA's ML engine: if mlflow autolog is added, never call `fit()` more than once per active run; structure training loops as a single fit call or as per-batch runs with explicit parent run nesting.

## Key Takeaway

ML framework autologgers assume a single `fit()` call per run and reset step counters per call — calling `fit()` multiple times in one run causes UNIQUE constraint violations on step-keyed metrics.
