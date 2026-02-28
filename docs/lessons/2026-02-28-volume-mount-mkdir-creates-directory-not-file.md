# Lesson: Docker Creates Mount Target as Directory When the Host Path Does Not Exist

**Date:** 2026-02-28
**System:** community (coollabsio/coolify)
**Tier:** lesson
**Category:** reliability
**Keywords:** bind mount, volume, directory vs file, Docker Compose, container filesystem, host path
**Source:** https://github.com/coollabsio/coolify/issues/8107

---

## Observation (What Happened)

A docker-compose file contains a file bind mount: `./volumes/logs/vector.yml:/etc/vector/vector.yml:ro`. The file `volumes/logs/vector.yml` exists in the repository as a valid YAML file. On deployment via Coolify, the container fails with `ERROR vector::cli: Configuration error. error=Is a directory (os error 21)`. Inspection of the host filesystem shows `volumes/logs/vector.yml` was created as a directory instead of a file. Every file bind mount in the compose file — 14 total — was silently created as a directory.

## Analysis (Root Cause — 5 Whys)

**Why #1:** When Docker (or a platform that runs docker compose) creates the host-side of a bind mount and the host path does not exist, it creates a directory at that path by default — regardless of whether the mount is intended for a file.
**Why #2:** Docker cannot infer from the compose file whether a non-existent host path should be a file or a directory — it defaults to directory because directories are the common case.
**Why #3:** Coolify's deployment pipeline cloned the repo to a working directory but the `volumes/` subdirectory was not pre-created from the repository contents before Docker processed the compose file.
**Why #4:** The container then tries to mount this newly-created directory as a file inside the container, producing `Is a directory (os error 21)`.
**Why #5:** The error message in the container log (`Is a directory`) does not mention the host path or the bind mount configuration, making diagnosis non-obvious.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | For file bind mounts, pre-create the host file before `docker compose up`: `touch ./volumes/logs/vector.yml` in a setup script or Makefile | proposed | community | coolify#8107 |
| 2 | In deployment pipelines, ensure the repository's `volumes/` directory structure is copied to the target host before compose runs — don't rely on Docker to create source paths | proposed | community | coolify#8107 |
| 3 | When debugging "Is a directory" errors on volume mounts: inspect the HOST path with `ls -la` — the source path being a directory when a file was expected is the most common cause | proposed | community | coolify#8107 |
| 4 | For config files that must exist, use Docker configs (`configs:` in compose) instead of bind mounts — configs are managed by Docker and have well-defined creation semantics | proposed | community | Docker configs docs |

## Key Takeaway

Docker auto-creates missing host paths for bind mounts as directories — for file bind mounts, the source file must already exist on the host before `docker compose up` runs, or the mount will silently become a directory, failing with a cryptic "Is a directory" error inside the container.
