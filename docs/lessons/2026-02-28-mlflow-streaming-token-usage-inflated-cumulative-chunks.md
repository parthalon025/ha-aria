# Lesson: Streaming LLM Traces Inflate Token Counts by Summing Cumulative Per-Chunk Usage Fields

**Date:** 2026-02-28
**System:** community (mlflow/mlflow)
**Tier:** lesson
**Category:** data-model
**Keywords:** mlflow, tracing, llm, streaming, token-usage, langchain, cumulative-sum, chunk-metadata, inflation, autolog
**Source:** https://github.com/mlflow/mlflow/issues/19649

---

## Observation (What Happened)

MLflow GenAI tracing with LangChain streaming (`ChatOpenAI(streaming=True)`) reported inflated `input_tokens` and `output_tokens` in the trace UI. The root cause: each streaming chunk's `usage_metadata` contains the full prompt token count and a cumulative output token count up to that point. The autologger summed these values across all chunks, producing `input_tokens = actual_prompt_tokens * num_chunks` and `output_tokens = sum(cumulative_output_counts)` instead of using only the final chunk's values.

## Analysis (Root Cause — 5 Whys)

The tracing code treated every chunk's usage metadata as an independent, additive contribution. For non-streaming calls this is correct: each call produces exactly one usage object. For streaming, the provider sends a delta-per-chunk usage pattern OR a cumulative-at-each-chunk pattern (provider-dependent). The autologger did not distinguish between delta and cumulative modes: it always summed, which is only correct for deltas. The final chunk's usage metadata contains the correct total in cumulative mode; summing all chunks multiplied the count by `num_chunks`.

## Corrective Actions

- For streaming LLM calls, always use only the final chunk's usage metadata for token accounting — discard all intermediate chunk usage fields.
- When implementing any metric accumulator for streaming responses: distinguish delta-mode (add all) from cumulative-mode (take last) — check whether `usage_metadata.output_tokens` increases monotonically; if yes, it is cumulative, take the final value only.
- In ARIA's patterns module (LLM call via `ollama_chat`): if streaming is ever enabled, accumulate token counts as a running max (not sum) over the `prompt_eval_count` field in Ollama responses.

## Key Takeaway

Streaming LLM usage metadata may be cumulative-per-chunk, not incremental — sum only delta-mode fields; for cumulative-mode fields, take the final chunk's value.
