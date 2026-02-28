# Lesson: Shared OpenTelemetry Tracer Provider Causes Recursive Span Processor Callbacks Across Autologgers

**Date:** 2026-02-28
**System:** community (mlflow/mlflow)
**Tier:** lesson
**Category:** reliability
**Keywords:** mlflow, autolog, opentelemetry, tracer-provider, span-processor, recursion, shared-state, multi-autologger, infinite-loop
**Source:** https://github.com/mlflow/mlflow/issues/20808

---

## Observation (What Happened)

`mlflow.strands.autolog()` with a shared OpenTelemetry tracer provider (`MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false`) caused `RecursionError: maximum recursion depth exceeded` on any span creation. `StrandsSpanProcessor.on_start()` called `tracer.span_processor.on_start()` on the shared provider — which dispatched to all registered processors, including `StrandsSpanProcessor` itself, creating infinite recursion. The behavior only appeared when both `strands.autolog()` and `bedrock.autolog()` were active simultaneously on the same tracer provider.

## Analysis (Root Cause — 5 Whys)

Each autologger registered its span processor with the shared provider's `add_span_processor()`. The `StrandsSpanProcessor.on_start` implementation then called `tracer.span_processor.on_start()` on the same tracer, which triggered the composite processor to dispatch to all registered processors, including itself. There was no guard to detect that the processor was already in its own callback — no re-entrancy protection, no recursion flag, no "already processing" sentinel. This is a classic listener self-re-enqueue bug applied to the OpenTelemetry span processor chain.

## Corrective Actions

- Any span processor that calls back into the same tracer provider from within `on_start()` or `on_end()` must set a per-call re-entrancy flag (e.g., `threading.local()`) and return early if already processing.
- When registering multiple autologgers, verify they do not share a processor callback path to the same tracer by inspecting `tracer.span_processor._span_processors` before and after each `autolog()` call.
- In ARIA: if OpenTelemetry tracing is ever added alongside mlflow or another observability framework, register each autologger's span processor on a dedicated child tracer, not the global root tracer.

## Key Takeaway

Any span processor that calls back into its own tracer's processor chain must protect against self-re-entrancy with a thread-local flag — failure causes infinite recursion when multiple processors share a tracer provider.
