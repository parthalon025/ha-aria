# Lesson: Regex Escape Sequences in Pydantic Field Patterns Cause Ollama Structured Output Failures

**Date:** 2026-02-28
**System:** community (ollama/ollama-python)
**Tier:** lesson
**Category:** integration
**Keywords:** ollama, structured-output, pydantic, regex, escape, JSON-schema, format, ResponseError, sampling-context
**Source:** https://github.com/ollama/ollama-python/issues/541

---

## Observation (What Happened)

Using `pydantic` `Field(pattern=...)` with regex patterns containing backslash escapes (e.g., `\._`) in Ollama's `format=` structured output caused `ResponseError: Failed to create new sequence: unable to create sampling context`. The identical pattern without the escape character worked fine. The failure was silent at the Python layer — no validation error, only a server-side crash.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Ollama returned a server-side error when the structured output format was requested.
**Why #2:** The Pydantic model's `model_json_schema()` serialized the escaped regex into the JSON schema passed to Ollama.
**Why #3:** The `\._` escape is redundant in a character class (`.` doesn't need escaping) but legal in Python regex; however, when serialized to JSON and passed to Ollama's GGML grammar compiler, the resulting grammar was invalid.
**Why #4:** Ollama's grammar generation from JSON Schema regex patterns does not normalize or validate escape sequences before compiling the sampling context.
**Why #5:** The error surfaced server-side only, with no diagnostic message indicating which part of the schema was problematic — the caller received a generic `ResponseError` with no actionable detail.

## Corrective Actions

| # | Action | Status | Owner | Evidence |
|---|--------|--------|-------|----------|
| 1 | Normalize regex patterns before passing to Ollama: remove redundant escapes in character classes | proposed | community | issue #541 |
| 2 | Validate JSON Schema regex patterns client-side using `re.compile()` before submitting to Ollama | proposed | community | defensive |
| 3 | Ollama should return a descriptive error identifying which schema field's pattern caused the failure | proposed | community | issue #541 |
| 4 | When structured output fails, log the full schema sent to the server for post-mortem debugging | proposed | community | operational |

## Key Takeaway

Backslash escapes in regex patterns that are valid in Python but redundant in character classes will silently crash Ollama's grammar compiler when used in structured output `format=` schemas — always simplify regex patterns passed to Ollama and validate them with `re.compile()` before submission.
