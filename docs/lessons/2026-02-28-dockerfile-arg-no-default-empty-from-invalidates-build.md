# Lesson: `ARG` Without Default Value in `FROM` Produces Empty Image Name — Silent Build Failure

**Date:** 2026-02-28
**System:** community (deviantony/docker-elk)
**Tier:** lesson
**Category:** reliability
**Keywords:** ARG, FROM, Dockerfile, build args, default value, empty image name, build cache, multi-stage build
**Source:** https://github.com/deviantony/docker-elk/issues/1015

---

## Observation (What Happened)

A Dockerfile contains `FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION}` where `ELASTIC_VERSION` is passed as a build arg with no default value. Running `docker build` without passing `--build-arg ELASTIC_VERSION=...` produces a BuildKit warning: `InvalidDefaultArgInFrom: Default value for ARG results in empty or invalid base image name`. The build may fail or produce a useless image. In CI pipelines that don't set build args, this produces a confusing failure with no clear indication that a missing variable is the cause.

## Analysis (Root Cause — 5 Whys)

**Why #1:** `ARG` before `FROM` can be used to parameterize the base image, but if the arg has no default and is not passed at build time, it expands to an empty string, making the `FROM` clause invalid.
**Why #2:** BuildKit warns but may attempt to proceed, producing an image based on an empty or invalid base — the resulting container fails at runtime in non-obvious ways.
**Why #3:** In multi-service compose setups (like ELK where all services share the same version), it is easy to forget to pass the arg when building a single service in isolation.
**Why #4:** The `ARG` → `FROM` pattern is correct for parameterized builds but requires a safe default so builds don't silently use the wrong or an empty base image.
**Why #5:** The warning text `InvalidDefaultArgInFrom` is precise but only visible if BuildKit output is read carefully — it doesn't fail the build by default in older Docker versions.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Always provide a default value for `ARG` used in `FROM`: `ARG ELASTIC_VERSION=8.17.3` followed by `FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION}` | proposed | community | docker-elk#1015 |
| 2 | Treat `InvalidDefaultArgInFrom` BuildKit warning as a build error in CI: `docker build ... 2>&1 | grep -E "error|WARNING" && exit 1` | proposed | community | Docker build checks docs |
| 3 | For multi-stage builds, ensure the `--build-arg` is passed to ALL stages that use the ARG — ARG scope resets after each `FROM` stage | proposed | community | Dockerfile ARG docs |

## Key Takeaway

`ARG` used in a `FROM` clause must have a default value; without one, builds without explicit `--build-arg` silently expand to an empty base image name, and the resulting failure is non-obvious and may only appear as a BuildKit warning rather than a hard error.
