# Lesson: Eagerly Applying str.format() to Untrusted Log Templates Crashes on Brace-Containing Strings

**Date:** 2026-02-28
**System:** community (getsentry/sentry-python)
**Tier:** lesson
**Category:** error-handling
**Keywords:** string formatting, template, KeyError, format(), log capture, structured logging, sentry logger
**Source:** https://github.com/getsentry/sentry-python/issues/4975

---

## Observation (What Happened)

Sentry's structured logger called `template.format(**kwargs)` on the raw log message string to capture it as a structured event. Any log message containing literal curly braces — JSON strings, regex patterns, format strings passed as data — raised `KeyError` because `str.format()` interpreted the content as unfilled template placeholders.

## Analysis (Root Cause — 5 Whys)

The SDK conflated two distinct use cases: structured logging with named placeholders (intentional template) vs. passing an arbitrary string that happens to contain `{`. Calling `str.format()` unconditionally without sanitizing the string, or without a `try/except KeyError`, causes any user-logged JSON or regex to raise uncaught exceptions inside the capture path. Because the exception is raised inside the SDK, not in user code, the error can crash log capture and lose the event.

## Corrective Actions

- Never call `str.format(**user_data)` on arbitrary user-supplied strings without a fallback: wrap in `try/except (KeyError, IndexError): body = template` to preserve the raw string on interpolation failure.
- When building a structured log capture pipeline, treat the template and the formatted body as separate fields — store the raw template regardless of whether formatting succeeds.
- Test the capture path with log messages containing `{`, `}`, `{{`, `{key}` without any kwargs.

## Key Takeaway

Any code that calls `str.format()` on untrusted input must handle `KeyError` and `IndexError` — log pipelines that crash on malformed templates silently lose log events.
