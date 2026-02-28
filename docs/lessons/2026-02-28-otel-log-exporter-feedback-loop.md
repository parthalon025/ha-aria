# Lesson: OTEL Log Exporter Logging Its Own Failures Creates an Infinite Feedback Loop

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** opentelemetry, LoggingHandler, feedback loop, infinite loop, recursion, OTLP exporter, root logger, export failure
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4688

---

## Observation (What Happened)

When the OTLP collector was unreachable, the `OTLPLogExporter` logged its export failure via the standard Python `logging` module. Because `LoggingHandler` was attached to the root logger, that failure log triggered another export attempt, which failed again and logged again — creating an infinite loop that hung the application.

## Analysis (Root Cause — 5 Whys)

The exporter mixin used `logging.getLogger(__name__).warning(...)` to report export failures. The application had added `LoggingHandler` to the root logger, which intercepts all log calls including those from the exporter itself. This is a classic feedback loop: the component responsible for delivering logs uses the same channel it is delivering to in order to report its own errors. The same pattern caused infinite recursion during shutdown when `atexit` handlers fired and the provider was already shut down.

## Corrective Actions

- Use a dedicated internal logger that bypasses the OTEL handler: `_internal_logger = logging.getLogger("opentelemetry._internal"); _internal_logger.propagate = False` and only attach a `StreamHandler` directly.
- When setting up OTEL logging in your application, add a `logging.Filter` or `logging.Manager.loggerDict`-based guard on the root logger to prevent OTEL-internal loggers from re-entering the handler.
- Test the failure path: start the app with no collector running and verify the process does not hang.

## Key Takeaway

Any component that uses `logging` to report its own failures must not attach to a logger that feeds back through the component — use a non-propagating internal logger with a direct stream sink.
