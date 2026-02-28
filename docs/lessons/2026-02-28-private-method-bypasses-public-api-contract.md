# Lesson: Internal Code Bypassing Public API via Private Method Skips Validation and Audit

**Date:** 2026-02-28
**System:** ollama-queue
**Tier:** lesson
**Category:** architecture
**Keywords:** private method, public API, bypass, validation, audit logging, rate limiting, _connect, internal access, contract violation, API boundary

---

## Observation (What Happened)

`rebalance()` in ollama-queue called the private `_connect()` helper directly to insert recurring jobs into the database, bypassing `insert_job()` — the public API method that enforces validation, rate limiting, and audit logging. Recurring jobs were added silently with no record in the audit log and no rate-limit check applied.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `rebalance()` needed a DB connection and used the private `_connect()` helper it already knew about, then issued raw SQL inserts.
**Why #2:** The developer treated `_connect()` as a convenience (get a connection handle) without recognizing that `insert_job()` exists to enforce invariants, not just write SQL.
**Why #3:** There is no enforcement mechanism — Python does not prevent calling underscore-prefixed methods, and no linter rule flagged the call site.
**Why #4:** The distinction between "get a resource" helpers and "enforce a contract" APIs is not documented or tested, so internal callers can accidentally (or casually) skip the contract layer.
**Why #5:** No integration test verified that the audit log was populated after a `rebalance()` run, so the bypass was invisible until manual inspection.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Rewrite `rebalance()` to call `insert_job()` instead of `_connect()` + raw SQL | proposed | Justin | ollama-queue #2 |
| 2 | Add an integration test that verifies audit log entries exist after `rebalance()` | proposed | Justin | — |
| 3 | Document in the module docstring which methods are the public contract vs. private helpers | proposed | Justin | — |

## Key Takeaway

When a public API method exists to enforce validation and auditing, all internal callers must use it — private helpers give access to the resource, not permission to skip the contract.
