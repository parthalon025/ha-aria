# Lesson: `COPY . .` Before `RUN pip install` Invalidates Build Cache on Every Code Change

**Date:** 2026-02-28
**System:** community (docker best practices)
**Tier:** lesson
**Category:** performance
**Keywords:** Docker build cache, COPY, pip install, layer caching, Dockerfile, build performance
**Source:** https://github.com/fastapi/full-stack-fastapi-template/issues/1493

---

## Observation (What Happened)

A Python service Dockerfile places `COPY . .` before `RUN pip install -r requirements.txt`. Every time any source file changes (even a comment in a `.py` file), Docker invalidates the COPY layer, which invalidates all subsequent layers including the pip install layer. A full dependency install (potentially 5-10 minutes) runs on every build even when `requirements.txt` has not changed.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Docker cache works layer-by-layer. Any change to a layer's inputs invalidates that layer and all layers below it.
**Why #2:** `COPY . .` copies all files, so any file change invalidates the layer. This includes `.py` source files that have nothing to do with dependencies.
**Why #3:** `RUN pip install` appears after `COPY . .`, so it re-runs on every code change regardless of whether `requirements.txt` changed.
**Why #4:** The correct pattern — copy only the dependency manifest first, install, then copy source — is not intuitive. Developers naturally write "copy everything, then install."
**Why #5:** CI build times silently balloon. No error occurs; builds just become slow.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Restructure Dockerfile: `COPY requirements.txt .` → `RUN pip install` → `COPY . .` — dependencies cache independently of source changes | proposed | community | Docker best practices |
| 2 | For pyproject.toml/Poetry: `COPY pyproject.toml poetry.lock ./` → `RUN poetry install --no-root` → `COPY . .` → `RUN poetry install --only-root` | proposed | community | poetry docs |
| 3 | Add `.dockerignore` with `*.pyc`, `__pycache__/`, `.git/`, `tests/` — reduces COPY layer inputs and prevents accidental cache busting | proposed | community | docker docs |

## Key Takeaway

`COPY . .` before `RUN pip install` guarantees cache miss on every code change; always copy only the dependency manifest file first, install dependencies, then copy source code.
