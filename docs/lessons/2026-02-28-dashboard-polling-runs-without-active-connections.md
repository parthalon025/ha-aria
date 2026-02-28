# Lesson: Dashboard Data Collection Must Gate on Active Client Connections

**Date:** 2026-02-28
**System:** community (MauriceNino/dashdot)
**Tier:** lesson
**Category:** performance
**Keywords:** dashboard, polling, WebSocket, socket.io, idle CPU, connection tracking, Raspberry Pi, server-push, resource usage
**Source:** https://github.com/MauriceNino/dashdot/issues/1050

---

## Observation (What Happened)

A Node.js dashboard server (dashdot) continuously polled system metrics and pushed data over socket.io regardless of whether any clients were connected. With zero active connections, the container consumed 6–14% CPU on a Raspberry Pi — comparable to its load under 10 active connections. On resource-constrained hardware, this caused the process to eventually crash under memory pressure during peak periods. The fix (shipped in v6.3.0) was to stop data collection when no socket.io connections exist and resume when the first client connects.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Dashboard process consumes significant CPU even when no browser tab is open.

**Why #2:** The polling loop and socket.io emission run on a fixed timer, independent of client connection state.

**Why #3:** The server architecture treated data collection and data delivery as the same pipeline — separating "is anyone watching?" from "what data to collect" was not a design concern.

**Why #4:** The default pattern for real-time dashboards (setInterval → emit) is straightforward to implement but does not account for the zero-client case.

**Why #5:** No connection lifecycle hook was wired to pause/resume the collection timers; the server has no concept of "idle" state.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Track active client count via connection/disconnect events. Start data collection timers on first connect; stop them when count reaches zero. | resolved | MauriceNino/dashdot v6.3.0 | https://github.com/MauriceNino/dashdot/issues/1050 |
| 2 | For dashboard services on constrained hardware (Pi, low-spec VPS): always gate metric collection on client presence — the push model should be demand-driven, not time-driven. | proposed | community | https://github.com/MauriceNino/dashdot/issues/1050 |
| 3 | Expose a configuration flag to allow operators to opt into always-on collection if they need pre-warmed data at first connect (e.g., for historical graphs that need backfill). | proposed | community | https://github.com/MauriceNino/dashdot/issues/1050 |
| 4 | Audit all `setInterval`/`setTimeout` loops in server-push services — any loop that produces data for clients should be connection-aware, not global. | proposed | community | https://github.com/MauriceNino/dashdot/issues/1050 |

## Key Takeaway

Real-time dashboard data collection must be demand-driven — start polling timers when the first client connects and stop them when the last client disconnects, or the server wastes CPU on data no one is reading, which crashes resource-constrained hardware under load.
