# Lesson: Sentry before_send Receives Inconsistent breadcrumbs Type Across Event Types

**Date:** 2026-02-28
**System:** community (getsentry/sentry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** sentry, before_send, breadcrumbs, event type, type inconsistency, cron monitor, check_in, silent drop
**Source:** https://github.com/getsentry/sentry-python/issues/4951

---

## Observation (What Happened)

A `before_send` hook that assumed `event["breadcrumbs"]` was always a dict with a `"values"` key worked correctly for errors and spans but silently dropped all cron monitor `check_in` events. Enabling `SENTRY_DEBUG=1` revealed an `AttributeError: 'list' object has no attribute 'get'` — the SDK set `breadcrumbs` to an empty list `[]` for check_in events, not the expected `{"values": [...]}` dict structure.

## Analysis (Root Cause — 5 Whys)

Different Sentry event types (error, transaction, check_in) are built by different internal code paths, and the SDK does not enforce a uniform schema for all fields before calling `before_send`. The `before_send` contract says the hook receives "an event dict" but the shape of optional fields varies. Any SDK version bump can change field shapes for non-error event types without a schema-breaking changelog entry.

## Corrective Actions

- Treat every field in a `before_send` hook defensively: `breadcrumbs = event.get("breadcrumbs") or {}; values = breadcrumbs if isinstance(breadcrumbs, list) else breadcrumbs.get("values", [])`.
- Write an integration test for `before_send` that exercises all event types (error, transaction, check_in) — not just error events.
- Never assume schema stability for SDKs that evolve across minor versions; pin and review changelogs before upgrades.

## Key Takeaway

`before_send` receives event dicts whose field types vary by event type — always guard with `isinstance` before accessing nested fields.
