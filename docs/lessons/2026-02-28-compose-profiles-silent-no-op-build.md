# Lesson: `docker compose build` Silently Does Nothing When All Services Have Profiles

**Date:** 2026-02-28
**System:** community (docker/compose)
**Tier:** lesson
**Category:** reliability
**Keywords:** Docker Compose, profiles, build, silent failure, CI, no-op, COMPOSE_PROFILES
**Source:** https://github.com/docker/compose/issues/13466

---

## Observation (What Happened)

A developer adds `profiles: ["dev", "prod"]` to a service in docker-compose.yml that previously had no profiles. Running `docker compose build` immediately returns to the prompt with no output and no error. The build did not run. The service is excluded from the default profile set, so with no active profile specified, compose considers there are no services to build and silently exits successfully (exit code 0). In CI, this produces a green build that built nothing.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Docker Compose profiles gate which services are active. Services with a `profiles:` key are only included when one of their profiles is explicitly activated via `--profile` or `COMPOSE_PROFILES`.
**Why #2:** When ALL services have profiles (or the target service has a profile and no default profile service exists), `docker compose build` has nothing to build and exits 0 with no warning.
**Why #3:** Exit code 0 + no output is indistinguishable from "built successfully with nothing to report" vs "no services matched" in automated pipelines.
**Why #4:** The `COMPOSE_PROFILES` environment variable is not always inherited from the shell in Docker Desktop WSL2 integration (separate issue #13552), making profile-based workflows unreliable in mixed environments.
**Why #5:** Developers adding profiles to an existing compose file don't expect existing build commands to stop working — the profile addition is a non-breaking-looking change.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | When adding profiles to a compose service, always update all build/run scripts and CI config to pass `--profile <name>` or set `COMPOSE_PROFILES=<name>` | proposed | community | docker/compose#13466 |
| 2 | Add a canary check in CI: after `docker compose build`, verify at least one image exists with `docker images --filter label=com.docker.compose.project=<name> --format "{{.ID}}" | wc -l` | proposed | community | docker/compose#13466 |
| 3 | Consider keeping a profile-free "base" service as a sentinel so `docker compose build` never silently no-ops | proposed | community | Docker Compose profiles docs |
| 4 | Do not rely on `COMPOSE_PROFILES` env var in Docker Desktop WSL2 — use `--profile` flag explicitly | proposed | community | docker/compose#13552 |

## Key Takeaway

`docker compose build` exits 0 with no output when no services match the active profile set — adding `profiles:` to all services without updating build scripts causes silent no-op builds in CI that look like successes.
