# Lesson: Docker Port Binding Defaults to 0.0.0.0, Silently Exposing Services to the Public Internet

**Date:** 2026-02-28
**System:** community (coollabsio/coolify)
**Tier:** lesson
**Category:** security
**Keywords:** port binding, 0.0.0.0, localhost, public exposure, database, UFW bypass, Docker networking
**Source:** https://github.com/coollabsio/coolify/issues/8581

---

## Observation (What Happened)

A user configured a PostgreSQL database port mapping as `5432:5432` in Coolify, expecting the database to be available only inside the Docker network (i.e., bound to `127.0.0.1:5432`). Instead, Docker bound the port on `0.0.0.0:5432`, making the database accessible from the public internet. UFW rules were also bypassed because Docker manages iptables rules directly, at a layer below UFW. The user confirmed external connectivity to the database from outside the VPS.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Docker's `ports` syntax `"5432:5432"` binds to all interfaces (`0.0.0.0`) by default — it is equivalent to `"0.0.0.0:5432:5432"`.
**Why #2:** Specifying only `HOST_PORT:CONTAINER_PORT` without a host IP gives the impression it's an internal mapping. The Docker documentation does state the default is `0.0.0.0` but this is not surfaced visually in UI tools.
**Why #3:** Docker modifies iptables PREROUTING/FORWARD rules directly, bypassing UFW. A UFW `deny 5432` rule has no effect because Docker's iptables rules are evaluated before UFW's rules in the FORWARD chain.
**Why #4:** Developers assume firewalls provide a safety net. They don't for Docker-exposed ports unless `DOCKER_OPTS="--iptables=false"` or explicit `127.0.0.1` binding is used.
**Why #5:** Platforms that manage compose files on behalf of users (Coolify, Portainer) may not enforce safe binding defaults.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | For services that should NOT be publicly accessible, bind explicitly to localhost: `"127.0.0.1:5432:5432"` — never bare `"5432:5432"` | proposed | community | coolify#8581 |
| 2 | Services that need only internal container-to-container communication should use no `ports:` at all — expose via Docker network directly | proposed | community | Docker networking docs |
| 3 | For production hosts with UFW: add `/etc/docker/daemon.json` with `{"iptables": false}` and manage port exposure via UFW (but this disables Docker DNS for container names) — or use `ufw-docker` tool | proposed | community | chaifeng/ufw-docker |
| 4 | Audit all `ports:` entries in compose files: any that don't have `127.0.0.1:` or a specific IP prefix are public-facing on all interfaces | proposed | community | security review |

## Key Takeaway

`ports: "5432:5432"` in docker-compose binds on ALL interfaces including public ones, and Docker bypasses UFW by writing iptables rules directly — always prefix with `127.0.0.1:` for any service not intended to be publicly accessible.
