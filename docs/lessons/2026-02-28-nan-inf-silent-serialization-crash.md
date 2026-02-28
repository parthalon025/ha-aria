# Lesson: NaN/Infinity Values Silently Crash Serialization and Drop WebSocket Connections

**Date:** 2026-02-28
**System:** community (miguelgrinberg/python-socketio)
**Tier:** lesson
**Category:** error-handling
**Keywords:** NaN, Infinity, math.nan, math.inf, JSON, serialization, silent crash, WebSocket disconnect, float, emit
**Source:** https://github.com/miguelgrinberg/python-socketio/issues/1438

---

## Observation (What Happened)

Emitting any payload containing `math.nan` or `math.inf` from the python-socketio server caused the underlying thread to crash silently with no logged error. The client observed an unexpected disconnect and attempted reconnection in a loop. No exception was surfaced to the application layer.

## Analysis (Root Cause — 5 Whys)

**Why #1:** The client disconnected unexpectedly on every emit containing `math.nan` or `math.inf`.
**Why #2:** The serialization step (JSON encoding) crashed because standard JSON does not support NaN or Infinity (`json.dumps(math.nan)` raises `ValueError` by default).
**Why #3:** The exception was swallowed at the transport layer rather than propagated to the caller, causing a silent thread death.
**Why #4:** No validation of float values in emitted payloads before serialization — the assumption was that Python `float` values are always JSON-serializable.
**Why #5:** `math.nan` and `math.inf` are valid Python floats but invalid JSON — the boundary was not enforced.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Validate all outbound payloads for NaN/Infinity before serialization: `if not math.isfinite(v): raise ValueError(...)` or substitute with `None` | proposed | community | issue #1438 |
| 2 | Use `json.dumps(data, allow_nan=False)` in any serialization path where silent NaN corruption is unacceptable | proposed | community | issue #1438 |
| 3 | In any async service emitting ML scores or sensor readings (which can produce NaN/Inf), add a `sanitize_floats()` pass before serialization | proposed | community | issue #1438 |

## Key Takeaway

`math.nan` and `math.inf` are valid Python floats but illegal JSON — any serialization path that can receive ML or sensor output must explicitly guard for non-finite values, or a silent transport crash will look like a network reconnection loop.
