# Lesson: `.env` File Used for Variable Interpolation Gets Injected Into All Containers via `env_file:`

**Date:** 2026-02-28
**System:** community (coollabsio/coolify)
**Tier:** lesson
**Category:** reliability
**Keywords:** env_file, .env, Docker Compose, variable interpolation, environment injection, secret leakage, compose semantics
**Source:** https://github.com/coollabsio/coolify/issues/8219

---

## Observation (What Happened)

Coolify automatically appends `env_file: [.env]` to every service in a managed docker-compose deployment. This causes every variable in the `.env` file (intended only for `${VAR}` interpolation in the compose YAML) to become a runtime environment variable inside every container. A variable like `DOCKER_HOST` set for one specific service is leaked into all other services, breaking infrastructure-level tools like Docker-in-Docker and GitLab Runner that are sensitive to `DOCKER_HOST` being set unexpectedly.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Docker Compose distinguishes two scopes: (a) `.env` at compose project root — used only for variable interpolation in the compose file itself, NOT injected into containers; (b) `env_file:` in a service definition — injects variables into the container runtime environment.
**Why #2:** When a platform or tool adds `env_file: [.env]` to all services, it conflates these two scopes and injects interpolation-only variables as runtime environment variables.
**Why #3:** Users who add variables to `.env` for compose interpolation don't expect those variables to appear inside their containers.
**Why #4:** There is no validation or warning when `env_file:` and `.env` contain overlapping variable names — the last writer wins silently.
**Why #5:** Some variables are infrastructure-level (host paths, Docker socket addresses, internal IPs) — injecting them into app containers can change app behavior in unpredictable ways.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Never add `env_file: [.env]` to all services globally — `.env` is for compose interpolation only; use explicit `environment:` blocks to inject variables into specific containers | proposed | community | coolify#8219 |
| 2 | Separate interpolation variables (compose-level) from container runtime variables (service-level) into different files: `.env` for compose interpolation, `app.env` for container injection | proposed | community | Docker Compose docs |
| 3 | Audit any platform (Coolify, Portainer) for automatic `env_file` injection behavior before deploying infrastructure-level services | proposed | community | coolify#8219 |
| 4 | For sensitive interpolation variables (tokens, passwords), prefer Docker Secrets or a secrets manager over `.env` files | proposed | community | Docker secrets docs |

## Key Takeaway

`.env` at the compose project root is for variable interpolation in the YAML only — it is NOT injected into containers unless explicitly referenced via `env_file:`; conflating these two scopes leaks interpolation-only variables (including infrastructure secrets) into all containers.
