# Lesson: supervisord Sends SIGTERM to Managed Processes Sequentially, Not in Parallel
**Date:** 2026-02-28
**System:** community (Supervisor/supervisor)
**Tier:** lesson
**Category:** lifecycle
**Keywords:** supervisord, SIGTERM, shutdown, sequential, process group, signal propagation, stopwaitsecs, timeout
**Source:** https://github.com/Supervisor/supervisor/issues/1649
---
## Observation (What Happened)
A supervisord instance managing multiple services, one of which could not exit on SIGTERM (process A), received SIGTERM to shut down. All other processes were eventually killed, but only after the entire `stopwaitsecs` timeout elapsed for process A. The shutdown was blocked until process A's timer expired. Investigation confirmed that supervisor sends SIGTERM to each process sequentially, not in parallel — so one unresponsive process delays termination of all subsequent processes.

## Analysis (Root Cause — 5 Whys)
supervisord's default shutdown behavior iterates through managed programs and sends SIGTERM to each, waiting up to `stopwaitsecs` for each before proceeding to the next. For a program that ignores SIGTERM, supervisor waits the full timeout before moving on. This sequential design means N unresponsive processes multiply the shutdown time by N, and in containerized environments this can cause supervisor (PID 1) to be SIGKILL'd by the container runtime's stop timeout before all children are properly reaped.

## Corrective Actions
- Set `stopwaitsecs` to a tight value (e.g., 5-10s) for processes known to need SIGKILL anyway; use the default (10s) only for processes with graceful shutdown.
- Use `killasgroup=true` and `stopasgroup=true` in the program config to send the signal to the entire process group, catching processes that fork or ignore signals.
- For containerized deployments: ensure the container `stop_timeout` is set to `total_stopwaitsecs * num_processes + 5` to give supervisor enough time.
- Diagnose unresponsive-to-SIGTERM processes: use `strace -p <pid>` to determine if the process is stuck in a blocking syscall that can be resolved without SIGKILL.

## Key Takeaway
supervisord stops processes sequentially — one SIGTERM-ignoring process blocks all subsequent shutdowns, so set `stopasgroup=true` and tune `stopwaitsecs` per process, not globally.
