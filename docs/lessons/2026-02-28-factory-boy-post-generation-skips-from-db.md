# Lesson: DjangoModelFactory post_generation Hooks Skip from_db — Model Instance Lacks Tracking State

**Date:** 2026-02-28
**System:** community (FactoryBoy/factory_boy)
**Tier:** lesson
**Category:** testing
**Keywords:** factory_boy, DjangoModelFactory, post_generation, from_db, _old_values, save, model state, Django ORM
**Source:** https://github.com/FactoryBoy/factory_boy/issues/1048

---

## Observation (What Happened)
A Django model overrode `from_db()` to attach `_old_values` for change-tracking in `save()`. A factory with a `@post_generation` hook triggered `_after_postgeneration()` which called `instance.save()`. This save raised `AttributeError: '_old_values'` because the instance was created via `factory.create()` (which calls `Model(**kwargs)` then `.save()`, not `from_db()`). The `_old_values` attribute that `from_db()` attaches was never set on the fresh instance.

## Analysis (Root Cause — 5 Whys)
**Why #1:** `DjangoModelFactory.create()` constructs the model via `Model(**kwargs)` and calls `.save()` — it does not go through `from_db()`.
**Why #2:** `from_db()` is only called when Django loads an existing record from the database, not when creating new instances in memory.
**Why #3:** `@post_generation` hooks run after the initial `.save()` and trigger a second `.save()` via `_after_postgeneration()`. This second save calls the model's `save()` method which assumes `_old_values` is present.
**Why #4:** The model's `save()` method did not guard for missing `_old_values` on freshly-created (non-DB-loaded) instances.
**Why #5:** This only surfaces when `@post_generation` is used — without it, the single factory `.save()` does not trigger the problem path.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | In models that use `from_db()` for tracking state, add a `__init__` override or `_old_values = {}` default so fresh instances have the attribute even if loaded outside of `from_db()` | proposed | community | issue #1048 |
| 2 | In factories with `@post_generation` hooks, explicitly call `instance.refresh_from_db()` at the start of the hook to ensure `from_db()` is invoked and tracking state is populated | proposed | community | issue #1048 |
| 3 | Guard change-detection code in `save()` with `hasattr(self, '_old_values')` to handle the fresh-instance case: `if hasattr(self, '_old_values') and any(...)` | proposed | community | issue #1048 |

## Key Takeaway
`DjangoModelFactory` creates model instances via the constructor, not `from_db()` — models that attach state only in `from_db()` will be missing that state in factory-created instances, causing `@post_generation`-triggered saves to fail; always initialize tracking attributes in `__init__` as well as `from_db()`.
