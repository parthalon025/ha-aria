# Lesson: OTEL gRPC Exporter Enters Permanent Broken State After Collector Restart

**Date:** 2026-02-28
**System:** community (open-telemetry/opentelemetry-python)
**Tier:** lesson
**Category:** reliability
**Keywords:** opentelemetry, gRPC, exporter, reconnect, collector restart, channel state, TRANSIENT_FAILURE, permanent failure, OTLP
**Source:** https://github.com/open-telemetry/opentelemetry-python/issues/4529

---

## Observation (What Happened)

After a temporary `otel-collector` restart, the `OTLPSpanExporter` (gRPC) never recovered. All subsequent export attempts silently failed even after the collector came back online. The gRPC channel entered `TRANSIENT_FAILURE` state and did not attempt reconnection, causing the application to lose all telemetry for the remainder of its lifetime.

## Analysis (Root Cause — 5 Whys)

The gRPC channel was created once at exporter initialization and reused without state inspection. When the collector became unreachable, the channel moved through `IDLE → CONNECTING → TRANSIENT_FAILURE`. The SDK did not call `channel.channel_ready()` or check channel connectivity state before each export, and the retry logic in the exporter did not distinguish a poisoned channel from a transient error. Channels in `TRANSIENT_FAILURE` require an explicit `wait_for_ready=True` hint or channel recreation to recover.

## Corrective Actions

- On export failure with gRPC status `UNAVAILABLE`, recreate the channel: close the old one and instantiate a new `OTLPSpanExporter`.
- Use `wait_for_ready=True` in gRPC stub calls so the client waits for channel reconnection rather than failing immediately.
- Add a health-check loop: if N consecutive exports fail, log a warning and force channel recreation rather than silently discarding spans.

## Key Takeaway

gRPC channels in `TRANSIENT_FAILURE` do not self-heal under the OTEL SDK — treat consecutive export failures as a signal to recreate the channel.
