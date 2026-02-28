# Lesson: Docker Default json-file Log Driver Has No Rotation — Disk Fills Unboundedly

**Date:** 2026-02-28
**System:** community (deviantony/docker-elk)
**Tier:** lesson
**Category:** reliability
**Keywords:** Docker logs, json-file, log rotation, disk full, logging driver, max-size, max-file
**Source:** https://github.com/deviantony/docker-elk/issues/1073

---

## Observation (What Happened)

A Docker-deployed ELK stack accumulates ~5TB of log files weekly. The user must manually `truncate -s 0` log files before restarting containers. The root cause is that Docker's default `json-file` logging driver writes logs to `/var/lib/docker/containers/<id>/<id>-json.log` with no size limit and no rotation configured. High-throughput log sources (Windows AD, Exchange, M365 feeding Logstash) fill the file indefinitely.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Docker's `json-file` logging driver defaults to unlimited log file size with no rotation.
**Why #2:** The `max-size` and `max-file` options exist but are not set by default and are not documented prominently in compose reference.
**Why #3:** Log-heavy services (Logstash, application containers receiving high event volumes) produce logs proportional to input throughput — no ceiling without explicit configuration.
**Why #4:** Log growth is invisible until disk is nearly full; no default alerting exists for container log file size.
**Why #5:** Many production deployments run for months before hitting the limit; by then the log files are enormous and truncating them loses diagnostic history.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Set log rotation globally in `/etc/docker/daemon.json`: `{"log-driver": "json-file", "log-opts": {"max-size": "100m", "max-file": "3"}}` — apply to all containers | proposed | community | Docker logging docs |
| 2 | Override per service in compose: `logging: {driver: json-file, options: {max-size: "50m", max-file: "5"}}` — service-level override takes precedence | proposed | community | docker-elk#1073 |
| 3 | For high-throughput services, switch to `local` log driver (compressed, rotated) or forward to dedicated logging infra (Loki, Splunk, CloudWatch) | proposed | community | Docker local driver docs |
| 4 | Add disk usage alerting: `df -h /var/lib/docker` should be monitored; add alert at 70% usage | proposed | community | ops best practice |

## Key Takeaway

Docker's default log driver writes without size limits — without `max-size` and `max-file` options, high-throughput containers will fill the disk unboundedly; configure log rotation before first production deployment.
