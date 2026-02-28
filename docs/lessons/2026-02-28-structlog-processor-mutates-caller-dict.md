# Lesson: structlog Renderer Mutating Caller-Supplied Style Dict Causes Accumulating State

**Date:** 2026-02-28
**System:** community (hynek/structlog)
**Tier:** lesson
**Category:** reliability
**Keywords:** structlog, ConsoleRenderer, mutation, shared state, level_styles, ANSI escape, processor, configuration
**Source:** https://github.com/hynek/structlog/issues/643

---

## Observation (What Happened)

`structlog.dev.ConsoleRenderer` mutated the `level_styles` dict passed to it at construction time, appending extra ANSI escape codes on every call to `structlog.configure()`. Code that called `configure()` multiple times in tests (a common pattern for resetting state) accumulated extra escape sequences each call, causing malformed terminal output. The caller's dict was modified in-place by the renderer's `__init__` instead of taking a defensive copy.

## Analysis (Root Cause — 5 Whys)

The renderer assumed ownership of the passed dict and mutated it to add a reset code. The API accepted the dict as a parameter with no documented "takes ownership" contract, so callers reasonably passed a module-level constant. Any code that calls a processor constructor more than once (test setup, hot-reload, multi-logger config) triggers the mutation on each call. The underlying rule — constructors that receive mutable arguments must copy them unless ownership transfer is explicit in the API contract.

## Corrective Actions

- In any class that stores a mutable argument (`dict`, `list`, `set`) on `self`, copy defensively at construction: `self._styles = dict(level_styles)`.
- When configuring structlog in tests, use `structlog.reset_defaults()` and reconstruct renderers from scratch rather than passing module-level shared dicts.
- Audit any processor/renderer subclass for in-place mutations of constructor arguments — this is a class of bug that only appears when constructors are called multiple times.

## Key Takeaway

Constructor arguments that are mutable containers must be copied unless the API explicitly documents ownership transfer — callers cannot be expected to know a dict will be mutated.
