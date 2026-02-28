# Lesson: `depends_on` Checks Container Start, Not Application Readiness

**Date:** 2026-02-28
**System:** community (deviantony/docker-elk, docker/compose)
**Tier:** lesson
**Category:** reliability
**Keywords:** depends_on, healthcheck, readiness, startup ordering, race condition, Docker Compose
**Source:** https://github.com/deviantony/docker-elk/issues/992

---

## Observation (What Happened)

A docker-compose stack with `depends_on: [elasticsearch]` fails intermittently at startup. The dependent service (Kibana, Logstash) starts before Elasticsearch has finished bootstrapping — even though Elasticsearch's container is "running." The dependent service exits with a connection error, and with `restart: unless-stopped` it spins in a retry loop. The issue is consistent on cold starts where Elasticsearch takes 30-60 seconds to become responsive.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `depends_on` in Docker Compose only waits for the dependent container to enter the "running" state (i.e., the container process has started). It does NOT wait for the application inside the container to be ready to accept connections.
**Why #2:** Elasticsearch takes 30-60 seconds after the Java process starts to finish bootstrapping (keystore, cluster formation, security setup). During this window the port is not listening.
**Why #3:** Without `condition: service_healthy` and a `healthcheck:` defined on the dependency, compose has no way to know when the application inside is ready.
**Why #4:** Defining `healthcheck:` is optional and non-obvious. Many public images don't include it, so users must add it themselves in compose.
**Why #5:** `restart: unless-stopped` masks the root cause — the service crashes and restarts until the dependency is ready, which works but wastes startup time and generates error log noise.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Define `healthcheck:` on every stateful service (databases, message brokers, APIs) in docker-compose.yml | proposed | community | docker-elk#992 |
| 2 | Use `depends_on: {service: {condition: service_healthy}}` to block dependent service startup until the healthcheck passes | proposed | community | Docker Compose docs |
| 3 | For Elasticsearch: `healthcheck: {test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -v red"], interval: 10s, retries: 20}` | proposed | community | docker-elk templates |
| 4 | For databases (Postgres/MySQL): `healthcheck: {test: ["CMD", "pg_isready", "-U", "${POSTGRES_USER}"]}` | proposed | community | postgres Docker Hub |

## Key Takeaway

`depends_on` guarantees container start order, not application readiness — without `condition: service_healthy` and a `healthcheck:`, dependent services will race against a not-yet-ready dependency and fail on cold start.
