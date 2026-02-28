# Lesson: PM2 Built-in Monitoring Agent (pmx) Keeps Node.js Event Loop Alive After uncaughtException
**Date:** 2026-02-28
**System:** community (Unitech/pm2)
**Tier:** lesson
**Category:** reliability
**Keywords:** pm2, pmx, uncaughtException, unhandledRejection, event loop, process hang, monitoring agent, Node.js
**Source:** https://github.com/Unitech/pm2/issues/6000
---
## Observation (What Happened)
Node.js processes managed by PM2 that hit `uncaughtException` or `unhandledRejection` would hang indefinitely instead of exiting and being restarted by PM2. `strace` revealed that the event loop was held alive by PM2's embedded monitoring agent (`@pm2/io` / pmx) continuously polling with `epoll_wait` and writing metric data. The process appeared running to PM2 and the OS but was functionally dead — requests were not handled and the restart mechanism never triggered.

## Analysis (Root Cause — 5 Whys)
PM2 injects its monitoring agent into managed Node.js processes via an IPC channel. The agent uses `setInterval` loops for metric collection and inter-process communication that are registered as unref'd or ref'd timers. When an `uncaughtException` reaches Node.js's default handler, execution stops but the agent's event sources keep the event loop from draining, preventing `process.exit()` from being called. The process is alive (responding to `kill -0`) but dead to the application. Setting `pmx: false` in the ecosystem config disables the monitoring agent, eliminating the event loop leak.

## Corrective Actions
- Set `pmx: false` in `ecosystem.config.js` for any process that must cleanly exit on unhandled exceptions.
- Alternatively: add an explicit `process.exit(1)` in `uncaughtException` and `unhandledRejection` handlers so the application forces exit regardless of event loop state.
- For Python asyncio equivalents: ensure `loop.stop()` is called in top-level exception handlers so the service exits and is restarted by systemd.
- In ARIA: any background task that registers timers must have its error path call `loop.stop()` or `sys.exit()` — never leave the process alive but non-functional.

## Key Takeaway
PM2's monitoring agent can keep a crashed Node.js process's event loop alive indefinitely — always add an explicit `process.exit(1)` in unhandled error handlers, or set `pmx: false` in the ecosystem config.
