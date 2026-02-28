# Lesson: pytest-django settings Fixture Changes Are Not Visible to Django Signal Handlers

**Date:** 2026-02-28
**System:** community (pytest-dev/pytest-django)
**Tier:** lesson
**Category:** testing
**Keywords:** pytest, pytest-django, settings, signal, post_save, django, fixture, settings override, SUSPEND_SIGNALS
**Source:** https://github.com/pytest-dev/pytest-django/issues/1026

---

## Observation (What Happened)
A test fixture modified `settings.SUSPEND_SIGNALS = True` using the pytest-django `settings` fixture. A `print(settings.SUSPEND_SIGNALS)` in the fixture showed `True`. However, inside a Django `post_save` signal handler triggered by test actions, the same `print(settings.SUSPEND_SIGNALS)` showed `False`. The signal handler could not see the override.

## Analysis (Root Cause — 5 Whys)
**Why #1:** The pytest-django `settings` fixture creates a Django override_settings context around the test. Signal handlers are registered at module import time and hold references to the settings module, not to the fixture-wrapped override.
**Why #2:** The signal handler reads `from django.conf import settings` which returns the original settings object (a `LazySettings` wrapper). The fixture's override is a `UserSettingsHolder` stacked on top, but signals may be reading settings values at a different proxy level.
**Why #3:** Django's signal dispatch is synchronous but happens in the context of the ORM save, which may use a different settings accessor than the test thread's override context.
**Why #4:** The settings fixture wraps `override_settings` which patches `django.conf.settings._wrapped`. Code that imports `settings` at module load and caches the attribute (e.g., `SUSPEND = settings.SUSPEND_SIGNALS`) will never see the override.
**Why #5:** The anti-pattern is caching settings values at import time rather than reading them fresh from `settings.SUSPEND_SIGNALS` at call time.

## Corrective Actions
| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Signal handlers and any code that must respect test settings overrides must read `from django.conf import settings; settings.SUSPEND_SIGNALS` at call time — never cache settings values in module-level variables | proposed | community | issue #1026 |
| 2 | Use `@override_settings(SUSPEND_SIGNALS=True)` as a decorator on the test class/method if the pytest `settings` fixture does not propagate to signal handlers in your Django version | proposed | community | issue #1026 |
| 3 | For signal suppression in tests, use Django's `Signal.disconnect()` / `connect()` pattern in a fixture rather than relying on a settings flag | proposed | community | issue #1026 |

## Key Takeaway
The pytest-django `settings` fixture overrides settings for the test thread, but Django signal handlers executing in ORM context may read settings from a different accessor level — signal handlers must read settings at call time (not module import time), and signal suppression via settings flag requires the handler to re-read the setting on every invocation.
